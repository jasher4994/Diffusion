# debug_text_encoder.py
from src.models.text_encoder import TextEncoder
import torch

text_encoder = TextEncoder()
checkpoint = torch.load("outputs/conditional/final_model.pt", map_location='cpu')
text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

# Test text encoding
with torch.no_grad():
    embeddings = text_encoder(["a forest scene", "a castle", ""])
    print("Text embeddings shape:", embeddings.shape)
    print("Text embeddings range:", embeddings.min(), "to", embeddings.max())
    print("Text embeddings mean:", embeddings.mean())
    
    # Check if all embeddings are the same
    if torch.allclose(embeddings[0], embeddings[1], atol=1e-6):
        print("❌ All text embeddings are identical!")
    else:
        print("✅ Text embeddings are different")