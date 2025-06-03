import torch
from tqdm import tqdm

def sample_and_show(model, diffusion, num_images=4, image_size=128, channels=3, device="cuda", cond=None):
    """
    Generate samples from the model and display them
    """
    from .visualization import show_tensor_image
    import matplotlib.pyplot as plt
    
    # Run in eval mode
    model.eval()
    
    # Create figure
    fig, axs = plt.subplots(1, num_images, figsize=(12, 3))
    
    with torch.no_grad():
        # Generate samples
        shape = (num_images, channels, image_size, image_size)
        samples = diffusion.sample(model, shape, device, cond)
        
        # Display samples
        for i, ax in enumerate(axs):
            if num_images > 1:
                ax.imshow(show_tensor_image(samples[i]))
                ax.axis('off')
            else:
                plt.imshow(show_tensor_image(samples[0]))
                plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Return to train mode
    model.train()
    
    return samples

def generate_grid_samples(model, diffusion, num_rows=4, num_cols=4, image_size=128, channels=3, device="cuda", cond=None):
    """
    Generate a grid of samples from the model
    """
    from .visualization import show_tensor_image
    import matplotlib.pyplot as plt
    
    # Run in eval mode
    model.eval()
    
    # Create figure
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))
    
    with torch.no_grad():
        # Generate samples
        shape = (num_rows * num_cols, channels, image_size, image_size)
        samples = diffusion.sample(model, shape, device, cond)
        
        # Display samples
        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j
                ax = axs[i, j] if num_rows > 1 else axs[j]
                ax.imshow(show_tensor_image(samples[idx]))
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Return to train mode
    model.train()
    
    return samples