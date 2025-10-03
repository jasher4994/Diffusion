# Text-Conditional Diffusion Configuration
IMAGE_SIZE = 64
BATCH_SIZE = 32  # Increased for faster training
NUM_EPOCHS = 5   # 5 epochs fits ~3 hour training window
LEARNING_RATE = 1e-4
TIMESTEPS = 1000

# Model architecture
CHANNELS = 128
TIME_DIM = 128
TEXT_DIM = 512  # CLIP embedding dimension

# Text encoder
CLIP_MODEL = "openai/clip-vit-base-patch32"
FREEZE_CLIP = True  # Freeze CLIP weights during training

# Data
DATASET_NAME = "Xenova/quickdraw-small"
MAX_SAMPLES = None  # Set to None to use all samples, or specify a number
NUM_CLASSES_FILTER = 10  # Filter to this many classes (set to None for all classes)
NUM_CLASSES = None  # Will be set after loading dataset