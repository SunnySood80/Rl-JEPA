from rl_agent import MaskingAgentTrainer
import torch

print("Testing RL components...")

# Small test to avoid memory conflicts
fi1_shape = (1, 384, 64, 64)
trainer = MaskingAgentTrainer(fi1_shape, mask_ratio=0.5, patch_size=8, device='cpu')

print(f"Created trainer with fi1_shape: {fi1_shape}")

# Test episode collection
print("Testing episode collection...")
masks, episodes = trainer.generate_masks_for_batch(batch_size=2)

print(f"✅ Generated {len(masks)} masks")
print(f"✅ Mask shape: {masks[0].shape}")
print(f"✅ Episode count: {len(episodes)}")
print(f"✅ Actions per episode: {len(episodes[0]['actions'])}")

print("RL components working correctly!")