.PHONY: train overfit clean clean-all clean-cuda help

# Default target
help:
	@echo "Available commands:"
	@echo "  make train     - Train the model with full dataset"
	@echo "  make overfit   - Train with 1 sample per class (overfitting test)"
	@echo "  make clean     - Clean checkpoints and generated images"
	@echo "  make clean-all - Clean everything including cache and logs"
	@echo "  make clean-cuda- Clear CUDA memory cache"
	@echo "  make help      - Show this help message"

# Train with full dataset
train:
	@echo "🚀 Starting full training..."
	python train.py --full

# Overfit training with 1 sample per class
overfit:
	@echo "🎯 Starting overfit training (1 sample per class)..."
	python train.py --overfit --samples 1

# Clean checkpoints and generated images
clean:
	@echo "🧹 Cleaning checkpoints and generated images..."
	rm -rf checkpoints/
	rm -rf checkpoints_overfit/
	rm -rf generated_images/
	rm -rf generated/
	@echo "✅ Cleaned checkpoints and images"

# Clean everything including cache and temporary files
clean-all: clean
	@echo "🧹 Deep cleaning..."
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	rm -rf *.pyc
	rm -rf */*.pyc
	rm -rf */*/*.pyc
	rm -rf .pytest_cache/
	rm -rf conditional_diffusion/__pycache__/
	rm -rf conditional_diffusion/*.pyc
	rm -rf logs/
	rm -rf .DS_Store
	rm -rf */.DS_Store
	@echo "✅ Deep clean completed"

# Clear CUDA memory cache
clean-cuda:
	@echo "🔧 Clearing CUDA memory cache..."
	python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')" 2>/dev/null || echo "PyTorch not available or no CUDA"
	@echo "✅ CUDA memory cleared"

# Quick overfit test (shorthand)
test: overfit

# Train for longer (you can modify epochs in config.py)
train-long:
	@echo "🚀 Starting long training session..."
	python train.py --full

# Generate samples (assumes you have a checkpoint)
generate:
	@echo "🎨 Generating samples..."
	@if [ -f "checkpoints/model_final.pt" ]; then \
		python generate.py checkpoints/model_final.pt --samples 4; \
	elif [ -f "checkpoints_overfit/model_final.pt" ]; then \
		python generate.py checkpoints_overfit/model_final.pt --samples 4; \
	else \
		echo "❌ No checkpoint found. Train a model first with 'make train' or 'make overfit'"; \
	fi

# Check GPU status
gpu-status:
	@echo "🔍 GPU Status:"
	@nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
	@echo ""
	@python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not available"

# Show training progress (if checkpoints exist)
status:
	@echo "📊 Training Status:"
	@if [ -d "checkpoints" ]; then \
		echo "Regular training checkpoints:"; \
		ls -la checkpoints/*.pt 2>/dev/null || echo "  No regular checkpoints found"; \
	fi
	@if [ -d "checkpoints_overfit" ]; then \
		echo "Overfit training checkpoints:"; \
		ls -la checkpoints_overfit/*.pt 2>/dev/null || echo "  No overfit checkpoints found"; \
	fi
	@if [ -d "generated_images" ] || [ -d "generated" ]; then \
		echo "Generated images:"; \
		ls -la generated_images/ generated/ 2>/dev/null || echo "  No generated images found"; \
	fi

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Quick test to see if everything works
quick-test:
	@echo "🧪 Quick test (2 epochs overfit)..."
	python -c "import config; config.NUM_EPOCHS = 2" 
	python train.py --overfit --samples 1

# Monitor GPU usage while training
monitor:
	@echo "📈 Monitoring GPU usage (Ctrl+C to stop)..."
	@while true; do \
		clear; \
		echo "=== GPU Monitoring ==="; \
		nvidia-smi 2>/dev/null || echo "nvidia-smi not available"; \
		echo ""; \
		echo "Press Ctrl+C to stop monitoring"; \
		sleep 2; \
	done