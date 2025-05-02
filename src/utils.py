def load_config(config_path):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_file):
    import logging
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

def save_model(model, model_path):
    import torch
    torch.save(model.state_dict(), model_path)

def load_model(model_class, model_path):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def print_progress(epoch, total_epochs):
    print(f'Epoch {epoch + 1}/{total_epochs}')