from datasets import load_dataset
import json
import os
from pathlib import Path
import requests
from PIL import Image
import time

def download_hf_dataset():
    """Download dataset from huggingface and convert to correct format."""
    
    # Set up directories
    base_dir = Path("/home/azureuser/Diffusion/data/unconditional/conditional")
    images_dir = base_dir / "images"
    
    # Create directories
    base_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    print("📦 Loading dataset from HuggingFace...")
    
    # Load the dataset
    try:
        ds = load_dataset("MatthewWaller/cifar_stable_diffusion")
        print(f"✅ Dataset loaded successfully!")
        
        # Usually datasets have train/test splits, let's check
        print(f"Available splits: {list(ds.keys())}")
        
        # Use the train split (or whichever split exists)
        if 'train' in ds:
            data = ds['train']
        else:
            # Take the first available split
            split_name = list(ds.keys())[0]
            data = ds[split_name]
            print(f"Using '{split_name}' split")
        
        print(f"📊 Dataset contains {len(data)} samples")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Examine the first sample to understand the structure
    print("\n🔍 Examining dataset structure...")
    sample = data[0]
    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if isinstance(value, str):
            print(f"  {key}: {value[:100]}..." if len(str(value)) > 100 else f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value)} - {value}")
    
    # Download images and create captions
    captions = {}
    failed_downloads = 0
    
    print(f"\n📥 Downloading images and preparing captions...")
    
    for i, sample in enumerate(data):
        try:
            # Extract image and text (adjust these keys based on actual dataset structure)
            # Common keys: 'image', 'text', 'caption', 'description'
            
            # You'll need to adjust these based on what you see in the sample structure above
            if 'image' in sample:
                image = sample['image']  # PIL Image
                caption = sample.get('text', sample.get('caption', sample.get('description', f"image captions {i}")))
            elif 'url' in sample:
                # If images are URLs, download them
                image_url = sample['url']
                caption = sample.get('text', sample.get('caption', f"image captions {i}"))
                
                # Download image
                try:
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    print(f"Failed to download image {i}: {e}")
                    failed_downloads += 1
                    continue
            else:
                print(f"⚠️  Couldn't find image data in sample {i}")
                continue
            
            # Save image
            filename = f"image_{i:05d}.jpg"
            image_path = images_dir / filename
            
            # Convert PIL image to RGB if needed and save
            if hasattr(image, 'save'):
                image.convert('RGB').save(image_path, 'JPEG', quality=95)
            else:
                print(f"⚠️  Sample {i} image is not a PIL Image: {type(image)}")
                continue
            
            # Add to captions
            captions[filename] = str(caption).strip()
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(data)} images...")
            
            # Small delay to be nice to servers (if downloading from URLs)
            if 'url' in sample:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
            failed_downloads += 1
            continue
    
    # Save captions.json
    captions_file = base_dir / "captions.json"
    with open(captions_file, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Download complete!")
    print(f"📊 Successfully processed: {len(captions)} images")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📁 Images saved to: {images_dir}")
    print(f"📄 Captions saved to: {captions_file}")
    
    # Print some example captions
    print(f"\n📝 Sample captions:")
    for i, (filename, caption) in enumerate(list(captions.items())[:3]):
        preview = caption[:100] + "..." if len(caption) > 100 else caption
        print(f"  {filename}: {preview}")

if __name__ == "__main__":
    download_hf_dataset()