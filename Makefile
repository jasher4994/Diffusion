# Simple Makefile for Conditional Diffusion Training

# Default Python command
PYTHON = python

# Default targets
.PHONY: help train generate quick full clean status

# Show help by default
help:
	@echo "🎨 Conditional Diffusion Makefile"
	@echo "================================="
	@echo ""
	@echo "Quick Commands:"
	@echo "  make quick      - Quick test (2 epochs, 20 samples)"
	@echo "  make train      - Regular training (uses config.py)"
	@echo "  make generate   - Generate images from latest model"
	@echo "  make full       - Full training (100 epochs)"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make status     - Show training progress"
	@echo "  make clean      - Remove checkpoints and generated images"
	@echo "  make config     - Show current config"
	@echo ""
	@echo "Custom Commands:"
	@echo "  make train-custom EPOCHS=50 SAMPLES=500"
	@echo "  make generate-custom CHECKPOINT=model_epoch_10.pt"
	@echo "  make generate-prompt PROMPT='Snoopy dancing'"

# Quick test - modify config for quick run
quick:
	@echo "🚀 Running quick test..."
	@echo "Temporarily setting config for quick test..."
	@sed -i.bak 's/BATCH_SIZE = .*/BATCH_SIZE = 2/' config.py
	@sed -i.bak 's/IMAGE_SIZE = .*/IMAGE_SIZE = 64/' config.py
	@sed -i.bak 's/NUM_EPOCHS = .*/NUM_EPOCHS = 2/' config.py
	@sed -i.bak 's/MAX_SAMPLES = .*/MAX_SAMPLES = 20/' config.py
	@sed -i.bak 's/TIMESTEPS = .*/TIMESTEPS = 100/' config.py
	@sed -i.bak 's/CHECKPOINT_DIR = .*/CHECKPOINT_DIR = ".\/quick_test_checkpoints"/' config.py
	$(PYTHON) train.py
	@echo "Restoring original config..."
	@mv config.py.bak config.py
	@echo "✅ Quick test completed!"

# Regular training using current config.py
train:
	@echo "🏃 Starting training with current config..."
	$(PYTHON) train.py

# Full training - modify config for intensive training
full:
	@echo "🔥 Setting up full training..."
	@sed -i.bak 's/NUM_EPOCHS = .*/NUM_EPOCHS = 100/' config.py
	@sed -i.bak 's/MAX_SAMPLES = .*/MAX_SAMPLES = None/' config.py
	@sed -i.bak 's/IMAGE_SIZE = .*/IMAGE_SIZE = 256/' config.py
	@sed -i.bak 's/CHECKPOINT_EVERY = .*/CHECKPOINT_EVERY = 10/' config.py
	$(PYTHON) train.py
	@echo "Restoring original config..."
	@mv config.py.bak config.py
	@echo "✅ Full training completed!"

# Generate images with latest model
generate:
	@echo "🎨 Generating images..."
	$(PYTHON) generate.py

# Generate with specific checkpoint
generate-custom:
	@echo "🎨 Generating with checkpoint: $(CHECKPOINT)"
	$(PYTHON) generate.py $(CHECKPOINT)

# Generate with custom prompt
generate-prompt:
	@echo "🎨 Generating with prompt: $(PROMPT)"
	$(PYTHON) generate.py model_final.pt $(PROMPT)

# Show current config
config:
	@echo "🔧 Current Configuration:"
	@echo "========================"
	@grep -E "^[A-Z_]+ = " config.py | head -15

# Show training status
status:
	@echo "📊 Training Status:"
	@echo "=================="
	@echo ""
	@echo "Checkpoints:"
	@if [ -d "checkpoints" ]; then \
		ls -la checkpoints/*.pt 2>/dev/null | tail -5 || echo "No checkpoints found"; \
	else \
		echo "No checkpoint directory found"; \
	fi
	@echo ""
	@echo "Generated Images:"
	@ls -la generated_*.png 2>/dev/null | tail -5 || echo "No generated images found"
	@echo ""
	@echo "GPU Status:"
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	@read -p "Delete all checkpoints and generated images? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf checkpoints/ quick_test_checkpoints/ full_train_checkpoints/; \
		rm -f generated_*.png quick_test_*.png; \
		echo "✅ Cleaned up!"; \
	else \
		echo "❌ Cancelled"; \
	fi

# Train with custom parameters
train-custom:
	@echo "🏃 Training with custom parameters..."
	@if [ "$(EPOCHS)" ]; then sed -i.bak 's/NUM_EPOCHS = .*/NUM_EPOCHS = $(EPOCHS)/' config.py; fi
	@if [ "$(SAMPLES)" ]; then sed -i.bak 's/MAX_SAMPLES = .*/MAX_SAMPLES = $(SAMPLES)/' config.py; fi
	@if [ "$(SIZE)" ]; then sed -i.bak 's/IMAGE_SIZE = .*/IMAGE_SIZE = $(SIZE)/' config.py; fi
	$(PYTHON) train.py
	@if [ -f config.py.bak ]; then mv config.py.bak config.py; echo "Config restored"; fi

# Monitor training (if running in background)
monitor:
	@echo "📈 Monitoring training progress..."
	@tail -f nohup.out 2>/dev/null || echo "No training log found (nohup.out)"

# Background training
train-bg:
	@echo "🏃 Starting background training..."
	nohup $(PYTHON) train.py > training.log 2>&1 &
	@echo "Training started in background. Check with 'make monitor' or 'tail -f training.log'"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install torch torchvision matplotlib pillow clip-by-openai tqdm

# Show disk usage
disk:
	@echo "💾 Disk Usage:"
	@echo "=============="
	@du -sh checkpoints/ 2>/dev/null || echo "No checkpoints directory"
	@du -sh data/ 2>/dev/null || echo "No data directory"
	@du -sh *.png 2>/dev/null || echo "No PNG files"

# Add monitoring commands
monitor:
	@echo "📈 Opening training monitor..."
	@if [ -d "checkpoints/monitoring" ]; then \
		echo "📊 Loss curve:"; \
		ls -la checkpoints/monitoring/loss_curve.png 2>/dev/null || echo "No loss curve yet"; \
		echo "🎨 Sample images:"; \
		ls checkpoints/monitoring/*.png 2>/dev/null | tail -10 || echo "No samples yet"; \
	else \
		echo "No monitoring data found"; \
	fi

# View latest samples
samples:
	@echo "🎨 Latest training samples:"
	@find checkpoints/monitoring -name "*.png" -type f | sort | tail -6