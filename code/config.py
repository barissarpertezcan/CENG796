# define the constants 
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
BATCH_SIZE = 1
root_dir = "../cc3m/train" # TODO: change root_dir with the path to the dataset according to your setup

# training parameters
num_train_epochs = 6
Lambda = 1.0

# optimizer parameters
learning_rate = 1e-5
discriminative_learning_rate = 1e-4  # New learning rate for discriminative tasks
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-4
adam_epsilon = 1e-8

# checkpoint parameters
output_dir = "../output_1_resumed_2"
save_steps = 1000

# Load the models
model_file = "data/v1-5-pruned.ckpt"  
unet_file = "../output_1_resumed/last.pt"  # Set to None to finetune from scratch

# EMA parameters
ema_decay = 0.9999
warmup_steps = 1000

# TEXT TO IMAGE
prompt1 = "A river with boats docked and houses in the background"
prompt2 = "A piece of chocolate swirled cake on a plate"
prompt3 = "A large bed sitting next to a small Christmas Tree surrounded by pictures"
prompt4 = "A bear searching for food near the river"
prompts = [prompt1, prompt2, prompt3, prompt4]
uncond_prompt = ""  # Also known as negative prompt
do_cfg = False
cfg_scale = 8  # min: 1, max: 14

# SAMPLER
sampler = "ddpm"
num_inference_steps = 50
seed = 42