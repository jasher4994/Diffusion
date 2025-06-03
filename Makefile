.PHONY: help train test clean status

BATCH_SIZE := 16
EPOCHS := 50

help:
	@echo "Simple Diffusion Training Commands:"
	@echo "  make train    - Train the model"
	@echo "  make test     - Test the model" 
	@echo "  make status   - Check system status"
	@echo "  make clean    - Clean outputs"

train:
	python src/train_text_cond.py --batch_size $(BATCH_SIZE) --epochs $(EPOCHS)

test:
	python src/test_text_cond.py --checkpoint outputs/conditional/final_model.pt

status:
	@echo "Disk space:"
	@df -h | head -2
	@echo "GPU status:"
	@python -c "import torch; print('GPU available:', torch.cuda.is_available())"

clean:
	rm -rf outputs/conditional/*
	@echo "Outputs cleaned"