import sys
import os
from conditional_diffusion.noise_scheduler import CosineNoiseScheduler

import torch
import pdb 


def main():
    pdb.set_trace()    
    # define the schedular                
    scheduler = CosineNoiseScheduler(num_timesteps=10, s=0.01)

    # simulating batch sizes, channels and timesteps etc.
    batch_size = 4
    C, H, W = 3, 32, 32
    images = torch.randn(batch_size, C, H, W) # torch.Size([4, 3, 32, 32])
    timesteps = torch.tensor([0, 3, 5, 9], dtype=torch.long) # self-explanatory

    pdb.set_trace()                 
    noisy_images, noise = scheduler.add_noise(images, timesteps)

    pdb.set_trace()                    
    print("Done.")

if __name__ == "__main__":
    main()
