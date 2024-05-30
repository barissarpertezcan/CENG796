import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import os
from ddpm import DDPMSampler
from pipeline import get_time_embedding
import model_loader
import time
from config import *
from transformers import CLIPTokenizer

# Set the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the models
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
ddpm = DDPMSampler(generator=None)

if unet_file is not None:
    # Load the UNet model
    print(f"Loading UNet model from {unet_file}")
    models['diffusion'].load_state_dict(torch.load(unet_file)['model_state_dict'])

# TEXT TO IMAGE
tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")

# Set the models['encoder'], models['clip'], models['diffusion'] to eval mode
models['encoder'].eval()
models['clip'].eval()
models['diffusion'].eval()

def test(device="cuda"):
    # Get the transform for the test data
    transform = transforms.Compose([
        transforms.Resize((WIDTH, HEIGHT), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load the CIFAR-10 dataset
    if test_dataset == "CIFAR10":
        testset = torchvision.datasets.CIFAR10(
            root='../dataset', train=False, download=True, transform=transform)

    elif test_dataset == "CIFAR100":
        testset = torchvision.datasets.CIFAR100(
            root='../dataset', train=False, download=True, transform=transform)

    print(f"Test dataset: {test_dataset} | Number of test samples: {len(testset)}")

    # Load the test data
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Move models to the device
    models['encoder'].to(device)
    models['clip'].to(device)
    models['diffusion'].to(device)

    # Define the class names and tokens
    class_names = testset.classes
    class_tokens = []

    # Tokenize class names
    for class_name in class_names:
        # Tokenize text
        tokens = tokenizer.batch_encode_plus(
            [class_name], padding="max_length", max_length=77
        ).input_ids
        tokens = torch.tensor(tokens, dtype=torch.long).squeeze()
        class_tokens.append(tokens)

    # Convert list of class tokens to a tensor
    class_tokens = torch.stack(class_tokens).to(device)
    print(f"Class tokens shape: {class_tokens.shape}")

    # Encode class tokens with the CLIP model
    with torch.no_grad():
        # Encode class tokens
        encoder_hidden_states = models['clip'](class_tokens)

        # Average and normalize class embeddings
        class_embeddings = encoder_hidden_states.mean(dim=1)
        class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
        print(f"Class embeddings shape: {class_embeddings.shape}\n")
    
    # Start testing
    test_loss = 0.0
    num_test_steps = len(testloader)
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(testloader):
            start_time = time.time()

            # Move batch to the device
            images = images.to(device)
            targets = targets.to(device)
            texts = [class_tokens[target] for target in targets]
            
            # Convert list of class tokens to a tensor
            texts = torch.stack(texts).to(device)

            # Encode images to latent space
            encoder_noise = torch.randn(images.shape[0], 4, LATENTS_HEIGHT, LATENTS_WIDTH).to(device)  # Shape (BATCH_SIZE, 4, 32, 32)
            latents = models['encoder'](images, encoder_noise)

            # Sample noise and timesteps for diffusion process
            bsz = latents.shape[0]
            timesteps = torch.randint(0, ddpm.num_train_timesteps, (bsz,), device=latents.device).long()
            text_timesteps = torch.randint(0, ddpm.num_train_timesteps, (bsz,), device=latents.device).long()

            # Add noise to latents and texts
            noisy_latents, image_noise = ddpm.add_noise(latents, timesteps)
            encoder_hidden_states = models['clip'](texts)
            noisy_text_query, text_noise = ddpm.add_noise(encoder_hidden_states, text_timesteps)

            # Get time embeddings
            image_time_embeddings = get_time_embedding(timesteps, is_image=True).to(device)
            text_time_embeddings = get_time_embedding(timesteps, is_image=False).to(device)
            
            # Average and normalize text time embeddings
            average_noisy_text_query = noisy_text_query.mean(dim=1)
            text_query = F.normalize(average_noisy_text_query, p=2, dim=-1)

            # Randomly drop 10% of text and image conditions: Context Free Guidance
            if torch.rand(1).item() < 0.1:
                text_query = torch.zeros_like(text_query)
            if torch.rand(1).item() < 0.1:
                noisy_latents = torch.zeros_like(noisy_latents)

            # Predict the noise residual and compute loss
            _, text_pred = models['diffusion'](noisy_latents, encoder_hidden_states, image_time_embeddings, text_time_embeddings, text_query)
                
            # Calculate loss
            loss = F.mse_loss(text_pred.float(), text_query.float(), reduction="mean")
            test_loss += loss.item()
            
            # Calculate cosine similarity between the generated text query and class embeddings
            similarities = F.cosine_similarity(text_pred.unsqueeze(1), class_embeddings.unsqueeze(0), dim=-1)
            predicted_classes = similarities.argmax(dim=-1)

            # Compare predictions with actual targets
            correct_predictions += (predicted_classes == targets).sum().item()
            total_predictions += targets.size(0)

            end_time = time.time()

            print(f"Batch {batch_idx + 1}/{num_test_steps} | Loss: {loss:.4f} | Time: {end_time - start_time:.2f}s", end="\r")

    # Calculate total accuracy
    accuracy = correct_predictions / total_predictions
    s = f"Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.4f})\n"
    s += f"\nTest Loss: {test_loss / num_test_steps:.4f}"
    print(s)
    with open(os.path.join(output_dir, 'test_results.txt'), 'a') as f:
        f.write(s)


if __name__ == "__main__":
    s = '==> Training starts..'
    s += f'\nModel file: {model_file}'
    s += f'\nUNet file: {unet_file}'
    s += f'\nBatch size: {BATCH_SIZE}'
    s += f'\nWidth: {WIDTH}'
    s += f'\nHeight: {HEIGHT}'
    s += f'\nLatents width: {LATENTS_WIDTH}'
    s += f'\nLatents height: {LATENTS_HEIGHT}'
    s += f'\nNumber of training epochs: {num_train_epochs}'
    s += f'\nLambda: {Lambda}'
    s += f'\nLearning rate: {learning_rate}'
    s += f'\nDiscriminative learning rate: {discriminative_learning_rate}'
    s += f'\nAdam beta1: {adam_beta1}'
    s += f'\nAdam beta2: {adam_beta2}'
    s += f'\nAdam weight decay: {adam_weight_decay}'
    s += f'\nAdam epsilon: {adam_epsilon}'
    s += f'\nEMA decay: {ema_decay}'
    s += f'\nWarmup steps: {warmup_steps}'
    s += f'\nOutput directory: {output_dir}'
    s += f'\nSave steps: {save_steps}'
    s += f'\nDevice: {DEVICE}'
    s += f'\nSampler: {sampler}'
    s += f'\nNumber of inference steps: {num_inference_steps}'
    s += f'\nSeed: {seed}'
    for i, prompt in enumerate(prompts):
        s += f'\nPrompt {i + 1}: {prompt}'
    s += f'\nUnconditional prompt: {uncond_prompt}'
    s += f'\nDo CFG: {do_cfg}'
    s += f'\nCFG scale: {cfg_scale}'
    s += f'\n\n'
    print(s)

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(s)

    test(device=DEVICE)
