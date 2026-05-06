"""Sanity check do data pipeline.

Roda DEPOIS de explore_dataset.py. Verifica:
1. Os DataLoaders funcionam.
2. Shapes batem com o esperado.
3. Distribuição de labels nos batches é razoável.
4. Imagens visualmente OK após transforms.
"""

import torch
import matplotlib.pyplot as plt
from collections import Counter

from detectia.data.loaders import make_loaders
from detectia.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Inverte a normalização ImageNet pra visualização."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def main():
    print("=== Criando DataLoaders ===")
    train_loader, val_loader = make_loaders(
        source="hemg",
        batch_size=8,
        num_workers=0,              # 0 pra debug
        max_samples_train=200,      # Subset pequeno pra ser rápido
        max_samples_val=50,
    )
    
    print(f"Tamanho do train_loader: {len(train_loader)} batches")
    print(f"Tamanho do val_loader: {len(val_loader)} batches")
    
    print("\n=== Pegando primeiro batch de treino ===")
    images, labels = next(iter(train_loader))
    
    print(f"Shape das imagens: {images.shape}")
    print(f"Tipo: {images.dtype}")
    print(f"Range dos pixels (após normalização): [{images.min():.3f}, {images.max():.3f}]")
    print(f"Labels do batch: {labels.tolist()}")
    print(f"Distribuição: {Counter(labels.tolist())}")
    
    # Visualizar
    print("\n=== Visualizando batch (salvando em sanity_check.png) ===")
    images_denorm = denormalize(images)
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, ax in enumerate(axes.flat):
        img = images_denorm[i].permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
        ax.imshow(img)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=100, bbox_inches="tight")
    print("Salvo em sanity_check.png — abra e VERIFIQUE VISUALMENTE.")
    
    print("\n=== Distribuição de labels no train completo (200 amostras) ===")
    all_labels = []
    for _, lbls in train_loader:
        all_labels.extend(lbls.tolist())
    print(Counter(all_labels))


if __name__ == "__main__":
    main()