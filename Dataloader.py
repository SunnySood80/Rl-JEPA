from torch.utils.data import DataLoader
from Dataset import JEPADataset, ADE20KDataset, jepa_collate, ade_collate

# =============================================================================
# Create Datasets and DataLoaders
# =============================================================================

# Training configuration
batch_size_pretrain =  30 # was 32 for no RL
batch_size_downstream = 20 # 24 for the supervised

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
    num_workers=4,
    pin_memory=True,
    collate_fn=jepa_collate
)

# Downstream fine-tuning loaders
downstream_train_loader = DataLoader(
    ade_train_dataset,
    batch_size=batch_size_downstream,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=ade_collate
)

downstream_val_loader = DataLoader(
    ade_val_dataset,
    batch_size=batch_size_downstream,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    collate_fn=ade_collate
)

print(f"\nDataLoader info:")
print(f"Pretrain: {len(pretrain_loader)} batches")
print(f"Train: {len(downstream_train_loader)} batches")
print(f"Val: {len(downstream_val_loader)} batches")