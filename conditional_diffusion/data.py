import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import io
import config

class QuickDrawDataset(Dataset):
    def __init__(self):
        print("Loading Quick Draw dataset...")

        # Load data
        df = pd.read_parquet("hf://datasets/nateraw/quickdraw-sample/data/train-00000-of-00001.parquet")

        # Balance classes
        samples_per_class = config.MAX_SAMPLES // df['label'].nunique()
        balanced_dfs = []

        for class_id in sorted(df['label'].unique()):
            class_df = df[df['label'] == class_id].head(samples_per_class)
            balanced_dfs.append(class_df)
            print(f"Class {class_id}: {len(class_df)} samples")

        self.df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
        self.num_classes = df['label'].nunique()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])

        print(f"Dataset ready: {len(self.df)} samples, {self.num_classes} classes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(row['label'], dtype=torch.long)

def get_dataloader():
    dataset = QuickDrawDataset()
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2), dataset.num_classes