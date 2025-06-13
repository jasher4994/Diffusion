# =============================================================================
# TRAINING CONFIGURATION - OPTIMIZED FOR FASTER LEARNING
# =============================================================================

# Training hyperparameters  
BATCH_SIZE = 16             # 🔥 INCREASED for more stable gradients
IMAGE_SIZE = 32            # Keep current
NUM_EPOCHS = 200           # More epochs for better convergence
LEARNING_RATE = 1e-4       # 🔥 REDUCED LR - more stable training

# Dataset configuration
MAX_SAMPLES = None          

# Model configuration - Increased capacity for better quality
TIMESTEPS = 1000           # 🔥 STANDARD timesteps for better denoising
BASE_CHANNELS = 128        # 🔥 INCREASED model capacity
TIME_EMB_DIM = 256         # 🔥 INCREASED for better time conditioning
TEXT_EMB_DIM = 512         # Keep CLIP dimension

# Generation configuration - Proper sampling
NUM_INFERENCE_STEPS = 100  # 🔥 More steps for higher quality

# Logging and checkpoints
CHECKPOINT_EVERY = 10      # 🔥 MORE frequent saves
LOG_EVERY = 100            # 🔥 MORE frequent logging to catch issues
CHECKPOINT_DIR = "./checkpoints"

# Monitoring configuration
GENERATE_SAMPLES_EVERY = 5 # 🔥 MORE frequent samples
SAVE_LOSS_PLOT_EVERY = 1   

# Resume from your current checkpoint
RESUME_FROM_CHECKPOINT = None  # 🔥 START FRESH with optimized settings

# Data paths
CAPTIONS_FILE = "/home/azureuser/Diffusion/data/unconditional/conditional/captions.json"
IMAGES_DIR = "/home/azureuser/Diffusion/data/unconditional/conditional/images/"

DEFAULT_PROMPTS = [
    None,  # Unconditional
    "A train",
    "A car", 
    "A horse",
    "A truck",
]