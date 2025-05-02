# Stable Diffusion Learning

This repository is dedicated to learning about Stable Diffusion, a powerful deep learning model for generating images from text descriptions. The project includes notebooks for hands-on learning, scripts for setting up the environment, and source code for training and inference.

## Project Structure

- **notebooks/**: Contains Jupyter notebooks for interactive learning.
  - `01_introduction_to_stable_diffusion.ipynb`: Introduction to Stable Diffusion concepts and applications.
  - `02_basic_inference.ipynb`: Demonstrates basic inference using a pre-trained model.
  - `03_fine_tuning.ipynb`: Explains the fine-tuning process on a specific dataset.
  - `04_custom_model_training.ipynb`: Guides through training a custom model from scratch.

- **src/**: Contains source code for data preparation, model training, and inference.
  - `data_preparation.py`: Functions for preparing and preprocessing datasets.
  - `model_training.py`: Classes and functions for training models.
  - `utils.py`: Utility functions for logging and configuration.
  - `inference.py`: Functions for running inference with trained models.

- **configs/**: Configuration files for training and inference.
  - `training_config.yaml`: Settings for training, including hyperparameters.
  - `inference_config.yaml`: Settings for inference, including model paths.

- **scripts/**: Shell scripts for setup and model downloading.
  - `download_pretrained_models.sh`: Automates downloading of pre-trained models.
  - `setup_environment.sh`: Sets up the project environment and installs dependencies.

- **requirements.txt**: Lists required Python packages for the project.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stable-diffusion-learning.git
   cd stable-diffusion-learning
   ```

2. Set up the environment:
   ```
   bash scripts/setup_environment.sh
   ```

3. Download pre-trained models:
   ```
   bash scripts/download_pretrained_models.sh
   ```

4. Open the Jupyter notebooks in the `notebooks/` directory to start learning.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.