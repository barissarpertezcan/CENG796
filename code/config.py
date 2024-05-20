# define the constants 
WIDTH = 256
HEIGHT = 256
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
BATCH_SIZE = 2
root_dir = "../dataset_deneme" # TODO: change root_dir with the path to the dataset according to your setup

# training parameters
first_epoch = 0
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
output_dir = "output"
save_steps = 10000
max_train_steps = 1000000

# EMA parameters
ema_decay = 0.9999
warmup_steps = 1000
