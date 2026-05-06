"""Factory de DataLoaders pra treino e validação."""

from torch.utils.data import DataLoader

from .dataset import DetectIADataset
from .transforms import get_train_transforms, get_eval_transforms


def make_loaders(
    source: str = "hemg",
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples_train: int = None,
    max_samples_val: int = None,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Cria DataLoaders pra treino e validação.
    
    Args:
        source: Fonte do dataset.
        batch_size: Quantas amostras por batch. Limitado pela VRAM.
        num_workers: Processos paralelos pra carregar dados.
            0 = single-process (debug). 4 = bom default.
            No WSL, valores altos podem dar problema. Reduza se houver erro.
        max_samples_*: Limites de amostras (debug rápido).
        pin_memory: True quando treinando em GPU. Acelera CPU→GPU.
    
    Returns:
        (train_loader, val_loader)
    """
    train_ds = DetectIADataset(
        source=source,
        split="train",
        transform=get_train_transforms(),
        max_samples=max_samples_train,
    )
    val_ds = DetectIADataset(
        source=source,
        split="val",
        transform=get_eval_transforms(),
        max_samples=max_samples_val,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,           # Embaralha a cada época
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # Descarta último batch incompleto (estabilidade)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,          # Validação NÃO embaralha
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,        # Validação usa todas as amostras
    )
    
    return train_loader, val_loader