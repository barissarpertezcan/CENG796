import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from ddpm import DDPMSampler
from pipeline import get_time_embedding
from dataloader import train_dataloader
import model_loader
import time
from config import *
from diffusion import TransformerBlock, UNet_Transformer  # Ensure these are correctly imported

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_file = "./data/v1-5-pruned.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

vae = models['encoder']
text_encoder = models['clip']
decoder = models['decoder']
unet = models['diffusion']
ddpm = DDPMSampler(generator=None)

# Disable gradient computations for the VAE, DDPM, and text_encoder models
for param in vae.parameters():
    param.requires_grad = False

for param in text_encoder.parameters():
    param.requires_grad = False

# Set the VAE and text_encoder to eval mode
vae.eval()
text_encoder.eval()

# Separate parameters for discriminative tasks
discriminative_params = []
non_discriminative_params = []

for name, param in unet.named_parameters():
    if isinstance(getattr(unet, name.split('.')[0], None), (TransformerBlock, UNet_Transformer)):
        discriminative_params.append(param)
    else:
        non_discriminative_params.append(param)

# AdamW optimizer with separate learning rates
optimizer = torch.optim.AdamW([
    {'params': non_discriminative_params, 'lr': learning_rate},
    {'params': discriminative_params, 'lr': discriminative_learning_rate}
], betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon)

# Linear warmup scheduler for non-discriminative parameters
def warmup_lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
    warmup_lr_lambda,  # Apply warmup for non-discriminative params
    lambda step: 1.0  # Keep constant learning rate for discriminative params
])

# EMA setup
ema_unet = torch.optim.swa_utils.AveragedModel(unet, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: ema_decay * averaged_model_parameter + (1 - ema_decay) * model_parameter)

def train(num_train_epochs, device="cuda", save_steps=1000, max_train_steps=10000):
    global_step = 0

    best_loss = float('inf')  # Initialize best loss as infinity
    accumulator = 0

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Move models to the device
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    ema_unet.to(device)

    num_train_epochs = tqdm(range(first_epoch, num_train_epochs), desc="Epoch")
    for epoch in num_train_epochs:
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()

            # Extract images and texts from batch
            images = batch["pixel_values"]
            texts = batch["input_ids"]

            # Normalize pixel values to [-1, 1]
            # images = images / 127.5 - 1.0

            # Move batch to the device
            images = images.to(device)
            texts = texts.to(device)

            # Encode images to latent space
            encoder_noise = torch.randn(images.shape[0], 4, LATENTS_HEIGHT, LATENTS_WIDTH).to(device)  # Shape (BATCH_SIZE, 4, 32, 32)
            latents = vae(images, encoder_noise)

            # Sample noise and timesteps for diffusion process
            bsz = latents.shape[0]
            timesteps = torch.randint(0, ddpm.num_train_timesteps, (bsz,), device=latents.device).long()
            text_timesteps = torch.randint(0, ddpm.num_train_timesteps, (bsz,), device=latents.device).long()

            # Add noise to latents and texts
            noisy_latents, image_noise = ddpm.add_noise(latents, timesteps)
            encoder_hidden_states = text_encoder(texts)
            noisy_text_query, text_noise = ddpm.add_noise(encoder_hidden_states, text_timesteps)

            # Get time embeddings
            image_time_embeddings = get_time_embedding(timesteps, is_image=True).to(device)
            text_time_embeddings = get_time_embedding(timesteps, is_image=False).to(device)
            
            # Average and normalize text time embeddings
            average_noisy_text_query = noisy_text_query.mean(dim=1)
            text_query = F.normalize(average_noisy_text_query, p=2, dim=-1)

            # Randomly drop 10% of text and image conditions: classifier free guidance
            if torch.rand(1).item() < 0.1:
                text_query = torch.zeros_like(text_query)
            if torch.rand(1).item() < 0.1:
                noisy_latents = torch.zeros_like(noisy_latents)

            # Predict the noise residual and compute loss
            image_pred, text_pred = unet(noisy_latents, encoder_hidden_states, image_time_embeddings, text_time_embeddings, text_query)
            image_loss = F.mse_loss(image_pred.float(), image_noise.float(), reduction="mean")
            text_loss = F.mse_loss(text_pred.float(), text_query.float(), reduction="mean")
            
            loss = image_loss + Lambda * text_loss
            train_loss += loss.item()
            accumulator += loss.item()

            # Backpropagate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            ema_unet.update_parameters(unet)

            if global_step % save_steps == 0 and global_step != 0:
                # Save model and optimizer state
                save_path = os.path.join(output_dir, f"last.pt")
                torch.save({
                    'model_state_dict': unet.state_dict(),
                    'ema_state_dict': ema_unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
                print(f"Saved state to {save_path}")

                # Check if the current step's loss is the best
                if accumulator / save_steps < best_loss:
                    best_loss = accumulator / save_steps
                    best_save_path = os.path.join(output_dir, "best.pt")
                    torch.save({
                        'model_state_dict': unet.state_dict(),
                        'ema_state_dict': ema_unet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, best_save_path)
                    print(f"New best model saved to {best_save_path}")

                accumulator = 0.0

            end_time = time.time()
            print(f"step_loss: {loss.item()}, time per step: {(end_time - start_time) / bsz}, step per second: {bsz / (end_time - start_time)}")

            if global_step >= max_train_steps:
                break

            global_step += 1

        print(f"Average loss over epoch: {train_loss / (step + 1)}")


if __name__ == "__main__":
    train(num_train_epochs=num_train_epochs)
