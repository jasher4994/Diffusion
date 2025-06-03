# check_model_params.py
import torch
from src.models.text_model import SimpleTextConditionedUNet

model = SimpleTextConditionedUNet()
checkpoint = torch.load("outputs/conditional/final_model.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

print("🔍 CHECKING MODEL PARAMETERS")
print("=" * 40)

# Check if parameters are reasonable
total_params = 0
zero_params = 0
very_small_params = 0

for name, param in model.named_parameters():
    param_count = param.numel()
    total_params += param_count
    
    # Check for problematic values
    zeros = (param.abs() < 1e-8).sum().item()
    very_small = (param.abs() < 1e-6).sum().item()
    
    zero_params += zeros
    very_small_params += very_small
    
    print(f"{name}: {param.shape}")
    print(f"  Range: {param.min():.6f} to {param.max():.6f}")
    print(f"  Mean: {param.mean():.6f}, Std: {param.std():.6f}")
    print(f"  Zeros: {zeros}/{param_count}, Very small: {very_small}/{param_count}")
    print()

print(f"SUMMARY:")
print(f"Total parameters: {total_params:,}")
print(f"Zero parameters: {zero_params:,} ({100*zero_params/total_params:.1f}%)")
print(f"Very small parameters: {very_small_params:,} ({100*very_small_params/total_params:.1f}%)")

if zero_params / total_params > 0.5:
    print("❌ More than 50% of parameters are zero - model likely broken!")
elif very_small_params / total_params > 0.8:
    print("⚠️  More than 80% of parameters are very small - possible vanishing gradients!")
else:
    print("✅ Parameter distribution looks reasonable")