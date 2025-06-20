# Training settings
LEARNING_RATE = 1e-4
BATCH_SIZE = 16  # Reduced for 128x128 images
NUM_EPOCHS = 50
CHECKPOINT_EVERY = 5
LOG_EVERY = 50

# Model settings
TIME_EMB_DIM = 128
CLASS_EMB_DIM = 128
BASE_CHANNELS = 64
IMAGE_SIZE = 128

# Diffusion settings
TIMESTEPS = 1000

# Paths
CHECKPOINT_DIR = "./checkpoints"
DATA_DIR = "/home/azureuser/Diffusion/data/cifar10-128x128"

# CIFAR-10 classes
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]