# Training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 10
CHECKPOINT_EVERY = 5
LOG_EVERY = 10

# Model hyperparameters
TIME_EMB_DIM = 128
CLASS_EMB_DIM = 128
BASE_CHANNELS = 64

# Diffusion hyperparameters
TIMESTEPS = 1000

# Training settings
USE_CLASSIFIER_FREE = True  # Enable classifier-free guidance training
CHECKPOINT_DIR = "./checkpoints"