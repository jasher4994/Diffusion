.PHONY: train overfit clean clean-all clean-cuda help
.PHONY: train-simple train-text overfit-simple overfit-text
.PHONY: generate-simple generate-text clean-simple clean-text

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Simple Class-Conditional Model:"
	@echo "  make train-simple      - Train simple model with full dataset"
	@echo "  make overfit-simple    - Quick test training (simple model)"
	@echo "  make generate-simple   - Generate samples from simple model"
	@echo "  make clean-simple      - Clean simple model checkpoints"
	@echo ""
	@echo "Text-Conditional Model (CLIP):"
	@echo "  make train-text        - Train text-conditional model"
	@echo "  make overfit-text      - Quick test training (text model)"
	@echo "  make generate-text     - Generate samples from text model"
	@echo "  make clean-text        - Clean text model checkpoints"
	@echo ""
	@echo "General:"
	@echo "  make clean-all         - Clean everything including cache"
	@echo "  make clean-cuda        - Clear CUDA memory cache"
	@echo "  make gpu-status        - Check GPU status"
	@echo "  make status            - Show training status"
	@echo ""
	@echo "Backwards compatibility aliases:"
	@echo "  make train             - Alias for train-simple"
	@echo "  make overfit           - Alias for overfit-simple"
	@echo "  make generate          - Alias for generate-simple"

# ============================================
# Simple Class-Conditional Model
# ============================================

# Train simple model with full dataset
train-simple:
	@echo "🚀 Starting simple model training..."
	cd conditional_diffusion && python train.py --epochs $(or $(EPOCHS),50)

# Quick overfit test for simple model
overfit-simple:
	@echo "🎯 Starting simple model quick test..."
	cd conditional_diffusion && python train.py --epochs $(or $(EPOCHS),5)

# Backwards compatibility aliases
train: train-simple
overfit: overfit-simple

# ============================================
# Text-Conditional Model (CLIP)
# ============================================

# Train text model with full dataset
train-text:
	@echo "🚀 Starting text-conditional model training..."
	cd text_conditional_diffusion && python train.py $(if $(EPOCHS),--epochs $(EPOCHS),)

# Quick overfit test for text model
overfit-text:
	@echo "🎯 Starting text model quick test..."
	cd text_conditional_diffusion && python train.py --epochs $(or $(EPOCHS),5)

# ============================================
# Cleaning
# ============================================

# Clean simple model checkpoints
clean-simple:
	@echo "🧹 Cleaning simple model checkpoints..."
	rm -rf conditional_diffusion/checkpoints/
	rm -rf conditional_diffusion/outputs/
	rm -rf conditional_diffusion/*.png
	@echo "✅ Cleaned simple model"

# Clean text model checkpoints
clean-text:
	@echo "🧹 Cleaning text model checkpoints..."
	rm -rf text_conditional_diffusion/checkpoints/
	rm -rf text_conditional_diffusion/outputs/
	rm -rf text_conditional_diffusion/*.png
	@echo "✅ Cleaned text model"

# Clean both models
clean: clean-simple clean-text

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
	rm -rf text_conditional_diffusion/__pycache__/
	rm -rf text_conditional_diffusion/*.pyc
	rm -rf logs/
	rm -rf .DS_Store
	rm -rf */.DS_Store
	@echo "✅ Deep clean completed"

# Clear CUDA memory cache
clean-cuda:
	@echo "🔧 Clearing CUDA memory cache..."
	python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')" 2>/dev/null || echo "PyTorch not available or no CUDA"
	@echo "✅ CUDA memory cleared"

# ============================================
# Generation
# ============================================

# Generate samples from simple model
generate-simple:
	@echo "🎨 Generating samples from simple model..."
	@cd conditional_diffusion && \
	if [ -f "checkpoints/quickdraw_final_epoch_50.pt" ]; then \
		python generate.py --checkpoint checkpoints/quickdraw_final_epoch_50.pt --num-samples $(or $(SAMPLES),8); \
	elif ls checkpoints/quickdraw_final_epoch_*.pt 1> /dev/null 2>&1; then \
		LATEST=$$(ls -t checkpoints/quickdraw_final_epoch_*.pt | head -n1); \
		echo "Using latest checkpoint: $$LATEST"; \
		python generate.py --checkpoint $$LATEST --num-samples $(or $(SAMPLES),8); \
	elif ls checkpoints/quickdraw_epoch_*.pt 1> /dev/null 2>&1; then \
		LATEST=$$(ls -t checkpoints/quickdraw_epoch_*.pt | head -n1); \
		echo "Using latest checkpoint: $$LATEST"; \
		python generate.py --checkpoint $$LATEST --num-samples $(or $(SAMPLES),8); \
	else \
		echo "❌ No checkpoint found. Train a model first with 'make train-simple'"; \
	fi

# Generate samples from text model
generate-text:
	@echo "🎨 Generating samples from text model..."
	@cd text_conditional_diffusion && \
	if ls checkpoints/*_final_epoch_*.pt 1> /dev/null 2>&1; then \
		LATEST=$$(ls -t checkpoints/*_final_epoch_*.pt | head -n1); \
		echo "Using latest checkpoint: $$LATEST"; \
		python generate.py --checkpoint $$LATEST --prompt "$(or $(PROMPT),a cat and dog)" --num-samples $(or $(SAMPLES),4); \
	elif ls checkpoints/*_epoch_*.pt 1> /dev/null 2>&1; then \
		LATEST=$$(ls -t checkpoints/*_epoch_*.pt | head -n1); \
		echo "Using latest checkpoint: $$LATEST"; \
		python generate.py --checkpoint $$LATEST --prompt "$(or $(PROMPT),a cat and dog)" --num-samples $(or $(SAMPLES),4); \
	else \
		echo "❌ No checkpoint found. Train a model first with 'make train-text'"; \
	fi

# Backwards compatibility
generate: generate-simple

# Check GPU status
gpu-status:
	@echo "🔍 GPU Status:"
	@nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
	@echo ""
	@python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not available"

# Show training status
status:
	@echo "📊 Training Status:"
	@echo ""
	@echo "Simple Model:"
	@if [ -d "conditional_diffusion/checkpoints" ]; then \
		echo "  Checkpoints:"; \
		ls -lh conditional_diffusion/checkpoints/*.pt 2>/dev/null | tail -5 || echo "    No checkpoints found"; \
	fi
	@if [ -d "conditional_diffusion/outputs" ]; then \
		echo "  Generated images:"; \
		ls -la conditional_diffusion/outputs/*.png 2>/dev/null || echo "    No generated images found"; \
	fi
	@echo ""
	@echo "Text Model:"
	@if [ -d "text_conditional_diffusion/checkpoints" ]; then \
		echo "  Checkpoints:"; \
		ls -lh text_conditional_diffusion/checkpoints/*.pt 2>/dev/null | tail -5 || echo "    No checkpoints found"; \
	fi
	@if [ -d "text_conditional_diffusion/outputs" ]; then \
		echo "  Generated images:"; \
		ls -la text_conditional_diffusion/outputs/*.png 2>/dev/null || echo "    No generated images found"; \
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