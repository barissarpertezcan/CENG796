import torch
import torch.nn.functional as F
import os
import shutil
from tqdm import tqdm

from encoder import VAE_Encoder
from ddpm import DDPMSampler
from clip import CLIP
from diffusion import Diffusion
from pipeline import get_time_embedding
from dataloader import train_dataloader

# TODO: move these to a config file

# define the constants 
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

first_epoch = 0
num_train_epochs = 10
latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

# optimizer constants
learning_rate = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 0.0
adam_epsilon = 1e-8

checkpoints_total_limit = None
output_dir = "output"

vae = VAE_Encoder()
ddpm = DDPMSampler(generator=None)
text_encoder = CLIP()
unet = Diffusion()

# Disable gradient computations for the VAE, DDPM, and text_encoder models
for param in vae.parameters():
    param.requires_grad = False

for param in text_encoder.parameters():
    param.requires_grad = False

# set the vae and text_encoder to eval mode
vae.eval()
text_encoder.eval()

optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon)

global_step = 0

def train(num_train_epochs, device="cuda"):
    # move models to the device
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    num_train_epochs = tqdm(range(first_epoch, num_train_epochs), desc="Epoch")
    for epoch in num_train_epochs:
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                # batch consists of images and texts, we need to extract the images and texts

                # move batch to the device
                batch["pixel_values"] = batch["pixel_values"].to(device)
                batch["input_ids"] = batch["input_ids"].to(device)

                # (Batch_Size, 4, Latents_Height, Latents_Width)
                encoder_noise = torch.randn(latents_shape, device=device)
                encoder_noise = encoder_noise.to(device)
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                latents = vae(batch["pixel_values"], encoder_noise)

                # Sample noise that we'll add to the latents -> it is done inside the add noise method
                # noise = torch.randn_like(latents)
                
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, ddpm.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents, noise = ddpm.add_noise(latents, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])

                target = noise

                time_embeddings = get_time_embedding(timesteps).to(device)

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, encoder_hidden_states, time_embeddings)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                train_loss += loss.item()

                # Backpropagate
                loss.backward()

                optimizer.zero_grad()
                optimizer.step()
                # lr_scheduler.step() # maybe linear scheduler can be added

                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if checkpoints_total_limit is not None:
                    checkpoints = os.listdir(output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        print(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")

                # Save model state and optimizer state
                torch.save({
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)

                print(f"Saved state to {save_path}")

                print("step_loss:", loss.detach().item())
                
                # if global_step >= max_train_steps:
                #     break

                global_step += 1

            print("Average loss over epoch:", train_loss / (step + 1))


    
if __name__ == "__main__":
    train(num_train_epochs)
