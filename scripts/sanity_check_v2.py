"""Sanity check v2: estratificado por classe.

Mostra 16 imagens de CADA classe (em arquivos separados) pra ter
uma noção melhor da qualidade do dataset.
Inclui o tamanho original no título — ajuda a detectar se uma classe
é dominada por imagens minúsculas que viraram lixo no resize.
"""

import torch
import matplotlib.pyplot as plt
from datasets import load_dataset

from detectia.data.transforms import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Inverte normalização ImageNet pra visualização."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def plot_samples(samples, transform, title, output_path):
    """Plota grid 4x4 de amostras transformadas (como o modelo vai ver)."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i, (idx, item) in enumerate(samples):
        original_size = item["image"].size
        image_tensor = transform(item["image"].convert("RGB"))
        image_denorm = denormalize(image_tensor)
        img = image_denorm.permute(1, 2, 0).numpy()
        
        ax = axes[i // 4, i % 4]
        ax.imshow(img)
        ax.set_title(
            f"idx={idx} | orig={original_size[0]}×{original_size[1]}",
            fontsize=9,
        )
        ax.axis("off")
    
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=80, bbox_inches="tight")
    print(f"Salvo: {output_path}")
    plt.close()


def main():
    print("Carregando dataset...")
    ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")
    
    print("Indexando por classe...")
    all_labels = ds["label"]
    indices_label_0 = [i for i, lbl in enumerate(all_labels) if lbl == 0]
    indices_label_1 = [i for i, lbl in enumerate(all_labels) if lbl == 1]
    
    print(f"Total label 0 (AiArtData / IA): {len(indices_label_0)}")
    print(f"Total label 1 (RealArt / Real): {len(indices_label_1)}")
    
    # Amostragem aleatória com seed fixa
    generator = torch.Generator().manual_seed(123)
    perm_0 = torch.randperm(len(indices_label_0), generator=generator).tolist()
    perm_1 = torch.randperm(len(indices_label_1), generator=generator).tolist()
    
    sample_indices_0 = [indices_label_0[i] for i in perm_0[:16]]
    sample_indices_1 = [indices_label_1[i] for i in perm_1[:16]]
    
    print("Carregando 16 amostras de cada classe...")
    samples_0 = [(idx, ds[idx]) for idx in sample_indices_0]
    samples_1 = [(idx, ds[idx]) for idx in sample_indices_1]
    
    transform = get_eval_transforms()
    
    print("\nPlotando...")
    plot_samples(
        samples_0, transform,
        "Label 0 (AiArtData) — supostamente IA-gerada",
        "ai_samples.png",
    )
    plot_samples(
        samples_1, transform,
        "Label 1 (RealArt) — supostamente Real",
        "real_samples.png",
    )
    
    print("\nAbra ai_samples.png e real_samples.png pra avaliação.")


if __name__ == "__main__":
    main()