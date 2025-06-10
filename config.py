# =============================================================================
# TRAINING CONFIGURATION - OPTIMIZED FOR FASTER LEARNING
# =============================================================================

# Training hyperparameters
BATCH_SIZE = 8              # 🔥 DOUBLE batch size - more stable gradients
IMAGE_SIZE = 32            # Keep current
NUM_EPOCHS = 150           # More epochs but faster per epoch
LEARNING_RATE = 5e-5       # 🔥 5x HIGHER LR - aggressive learning

# Dataset configuration
MAX_SAMPLES = None          

# Model configuration - Optimized for speed + capacity
TIMESTEPS = 200            # 🔥 MUCH EASIER denoising problem (was 500)
BASE_CHANNELS = 96         # 🔥 REDUCE slightly - faster training
TIME_EMB_DIM = 192         # 🔥 REDUCE - less computation
TEXT_EMB_DIM = 512         # Keep CLIP dimension

# Generation configuration - Better sampling
NUM_INFERENCE_STEPS = 50   # 🔥 200/50 = 4x ratio (much easier)

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