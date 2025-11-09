from torch.utils.data import DataLoader
from Dataset import JEPADataset, ADE20KDataset, jepa_collate, ade_collate

# =============================================================================
# Create Datasets and DataLoaders
# =============================================================================

# Training configuration
# A100 (49GB VRAM) optimized settings:
# - batch_size_pretrain: 32-40 recommended (was 24, H200 used 100)
# - With RL: 32 is safe, 40 may work with monitoring
# - Without RL: can go up to 48
batch_size_pretrain = 24  # Optimized for A100 49GB with RL masking
batch_size_downstream = 16 # 24 for the supervised, 20 for H200

# Create dataset instances
jepa_dataset = JEPADataset()
ade_train_dataset = ADE20KDataset(split="training")
ade_val_dataset = ADE20KDataset(split="validation")

print(f"Dataset sizes:")
print(f"JEPA (ImageNet): {len(jepa_dataset):,} samples")
print(f"ADE20K train: {len(ade_train_dataset):,} samples")
print(f"ADE20K val: {len(ade_val_dataset):,} samples")

# JEPA pretraining loader
pretrain_loader = DataLoader(
    jepa_dataset,
    batch_size=batch_size_pretrain,
    shuffle=True,
    num_workers=2,  # Reduced from 4 to avoid worker overload
    pin_memory=True,
    collate_fn=jepa_collate,
    persistent_workers=True
)

# Downstream fine-tuning loaders
downstream_train_loader = DataLoader(
    ade_train_dataset,
    batch_size=batch_size_downstream,
    shuffle=True,
    num_workers=2,  # Reduced from 8 to avoid worker overload
    pin_memory=True,
    collate_fn=ade_collate,
    persistent_workers=True
)

downstream_val_loader = DataLoader(
    ade_val_dataset,
    batch_size=batch_size_downstream,
    shuffle=False,
    num_workers=2,  # Reduced from 8 to avoid worker overload
    pin_memory=True,
    collate_fn=ade_collate,
    persistent_workers=True
)

print(f"\nDataLoader info:")
print(f"Pretrain: {len(pretrain_loader)} batches")
print(f"Train: {len(downstream_train_loader)} batches")
print(f"Val: {len(downstream_val_loader)} batches")