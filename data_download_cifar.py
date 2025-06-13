import kagglehub

# Download latest version
path = kagglehub.dataset_download("joaopauloschuler/cifar10-128x128-resized-via-cai-super-resolution")

print("Path to dataset files:", path)