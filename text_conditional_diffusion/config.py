# Text-Conditional Diffusion Configuration
# Optimized for 4x Tesla V100 (16GB each) - Proof of Concept
IMAGE_SIZE = 64
BATCH_SIZE = 128  # 32 per GPU across 4 GPUs
NUM_EPOCHS = 100  # Longer training for better text conditioning
LEARNING_RATE = 1e-4
TIMESTEPS = 1000

# Model architecture
CHANNELS = 256  # Maximum capacity for best feature learning
TIME_DIM = 128
TEXT_DIM = 512  # CLIP embedding dimension

# Text encoder
CLIP_MODEL = "openai/clip-vit-base-patch32"
FREEZE_CLIP = True  # Freeze CLIP weights during training

# Classifier-free guidance
CFG_DROP_PROB = 0.15  # Increased to 15% for stronger conditioning contrast
CFG_GUIDANCE_SCALE = 5.0  # Higher default guidance scale

# Data
DATASET_NAME = "Xenova/quickdraw-small"
MAX_SAMPLES = None  # Set to None to use all samples, or specify a number
NUM_CLASSES_FILTER = 5  # Small set for clear proof-of-concept
NUM_CLASSES = None  # Will be set after loading dataset