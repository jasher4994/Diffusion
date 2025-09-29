.PHONY: train overfit clean clean-all clean-cuda help

# Default target
help:
	@echo "Available commands:"
	@echo "  make train     - Train the model with full dataset"
	@echo "  make overfit   - Train with small dataset (quick test)"
	@echo "  make clean     - Clean checkpoints and generated images"
	@echo "  make clean-all - Clean everything including cache and logs"
	@echo "  make clean-cuda- Clear CUDA memory cache"
	@echo "  make generate  - Generate samples from trained model"
	@echo "  make help      - Show this help message"

# Train with full dataset
train:
	@echo "🚀 Starting full training..."
	cd conditional_diffusion && python train.py --epochs $(or $(EPOCHS),50)

# Quick overfit test
overfit:
	@echo "🎯 Starting quick test training..."
	cd conditional_diffusion && python train.py --epochs $(or $(EPOCHS),5)

# Clean checkpoints and generated images
clean:
	@echo "🧹 Cleaning checkpoints and generated images..."
	rm -rf conditional_diffusion/checkpoints/
	rm -rf conditional_diffusion/outputs/
	rm -rf conditional_diffusion/*.png
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

# Generate samples (assumes you have a checkpoint)
generate:
	@echo "🎨 Generating samples..."
	@cd conditional_diffusion && \
	if [ -f "checkpoints/quickdraw_final_epoch_50.pt" ]; then \
		python generate.py --checkpoint checkpoints/quickdraw_final_epoch_50.pt --num-samples $(or $(SAMPLES),8); \
	elif ls checkpoints/quickdraw_final_epoch_*.pt 1> /dev/null 2>&1; then \
		LATEST=$$(ls -t checkpoints/quickdraw_final_epoch_*.pt | head -n1); \
		echo "Using latest checkpoint: $$LATEST"; \
		python generate.py --checkpoint $$LATEST --num-samples $(or $(SAMPLES),8); \
	elif ls checkpoints/quickdraw_step_*.pt 1> /dev/null 2>&1; then \
		LATEST=$$(ls -t checkpoints/quickdraw_step_*.pt | head -n1); \
		echo "Using latest step checkpoint: $$LATEST"; \
		python generate.py --checkpoint $$LATEST --num-samples $(or $(SAMPLES),8); \
	else \
		echo "❌ No checkpoint found. Train a model first with 'make train' or 'make overfit'"; \
	fi

# Generate specific class
generate-class:
	@echo "🎨 Generating samples for class $(or $(CLASS),0)..."
	@cd conditional_diffusion && \
	if [ -f "checkpoints/quickdraw_final_epoch_50.pt" ]; then \
		python generate.py --checkpoint checkpoints/quickdraw_final_epoch_50.pt --class-id $(or $(CLASS),0) --num-samples $(or $(SAMPLES),4); \
	elif ls checkpoints/quickdraw_final_epoch_*.pt 1> /dev/null 2>&1; then \
		LATEST=$$(ls -t checkpoints/quickdraw_final_epoch_*.pt | head -n1); \
		echo "Using latest checkpoint: $$LATEST"; \
		python generate.py --checkpoint $$LATEST --class-id $(or $(CLASS),0) --num-samples $(or $(SAMPLES),4); \
	else \
		echo "❌ No checkpoint found. Train a model first with 'make train' or 'make overfit'"; \
	fi

# List available classes
list-classes:
	@echo "📋 Available classes:"
	cd conditional_diffusion && python generate.py --list-classes

# Check GPU status
gpu-status:
	@echo "🔍 GPU Status:"
	@nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
	@echo ""
	@python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not available"

# Show training status
status:
	@echo "📊 Training Status:"
	@if [ -d "conditional_diffusion/checkpoints" ]; then \
		echo "Checkpoints:"; \
		ls -la conditional_diffusion/checkpoints/*.pt 2>/dev/null || echo "  No checkpoints found"; \
	fi
	@if [ -d "conditional_diffusion/outputs" ]; then \
		echo "Generated images:"; \
		ls -la conditional_diffusion/outputs/*.png 2>/dev/null || echo "  No generated images found"; \
	fi

# Quick test (shorthand for overfit)
test: overfit

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