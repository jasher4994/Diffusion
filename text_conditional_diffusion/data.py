"""Dataset loader for text-conditional Quick Draw."""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import config


class QuickDrawTextDataset(Dataset):
    """
    This is a PyTorch Dataset that loads QuickDraw images and converts class labels into text
    prompts for CLIP.
    """

    def __init__(self, split="train", max_samples=None, num_classes=None):
        print(f"Loading Quick Draw dataset ({split} split)...")

        dataset = load_dataset(config.DATASET_NAME)
        self.dataset = dataset[split]

        self.label_feature = self.dataset.features['label']
        self.all_class_names = self.label_feature.names

        if num_classes is not None:
            selected_label_ids = list(range(num_classes))
            selected_class_names = [self.all_class_names[i] for i in selected_label_ids]
            print(f"   Filtering to {num_classes} classes: {selected_class_names}")

            self.dataset = self.dataset.filter(lambda x: x['label'] in selected_label_ids)

        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        unique_label_ids = sorted(set(self.dataset['label']))
        self.class_names = [self.all_class_names[i] for i in unique_label_ids]
        self.num_classes = len(self.class_names)

        config.NUM_CLASSES = self.num_classes

        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # output = [-1, 1] (0.5 mean, 0.5 std)
        ])

        print(f"✅ Dataset loaded: {len(self.dataset)} samples, {self.num_classes} classes")
        print(f"   Sample classes: {self.class_names[:10]}{'...' if len(self.class_names) > 10 else ''}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if image.mode != 'L':
            image = image.convert('L')

        if self.transform:
            image = self.transform(image)

        label_id = item['label']
        label_name = self.all_class_names[label_id]

        text_prompt = f"a drawing of a {label_name}"

        return image, text_prompt

    def get_class_names(self):
        """Return list of all class names."""
        return self.class_names

