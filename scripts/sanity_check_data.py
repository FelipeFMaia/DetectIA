"""Sanity check do dataset e do data pipeline.

Faz duas verificações DIFERENTES e COMPLEMENTARES:

1. check_pipeline()
   Testa se o DataLoader funciona ponta-a-ponta:
     - Cria os DataLoaders de treino e validação
     - Pega o primeiro batch de treino
     - Verifica shapes, dtype e range de pixels (após normalização)
     - Visualiza 8 imagens DEPOIS dos transforms (com data augmentation)
     - Imprime distribuição de labels num subset
   Output: sanity_pipeline.png

2. check_per_class()
   Inspeção visual estratificada do dataset (sem augmentation):
     - Carrega o dataset cru, sem DataLoader
     - Amostra N imagens de cada classe (default 16)
     - Mostra cada imagem com seu tamanho ORIGINAL no título
     - Útil pra detectar lixo: imagens minúsculas, miniaturas, label noise
   Output: sanity_class_ai.png e sanity_class_real.png

Como rodar (sempre do raiz do projeto):
    python scripts/sanity_check_data.py

Esse script NÃO tem flags por enquanto — roda os dois checks com defaults
sensatos. Se um dia precisar parametrizar (mudar n_samples, mudar source,
pasta de saída), basta adicionar argparse no main() e passar pros checks.

Saídas no diretório atual:
    sanity_pipeline.png   — 8 imagens do batch após transforms+augmentation
    sanity_class_ai.png   — 16 amostras de label 0 (IA), tamanho original no título
    sanity_class_real.png — 16 amostras de label 1 (Real), tamanho original no título

DEPOIS DE RODAR: abra os 3 PNGs e confira a olho. Sanity check só funciona
se um humano olhar pra saída.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from detectia.data.loaders import make_loaders
from detectia.data.dataset import DetectIADataset
from detectia.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms



def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Inverte a normalização ImageNet pra visualização.

    O modelo recebe imagens normalizadas (mean/std do ImageNet), mas
    matplotlib precisa de [0, 1]. Esta função desfaz a normalização.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def check_pipeline(
    source: str = "hemg",
    n_samples_train: int = 200,
    n_samples_val: int = 50,
    output_path: Path = Path("sanity_pipeline.png"),
) -> None:
    """Smoke test do data pipeline (DataLoader + transforms + augmentation).

    Args:
        source: nome da fonte registrada em data/sources/. Default 'hemg'.
        n_samples_train: subset pequeno pra ser rápido (default 200).
        n_samples_val: subset pequeno pra validação (default 50).
        output_path: caminho do PNG com 8 imagens do primeiro batch.

    O que valida:
        - DataLoaders se criam sem erro
        - Shape do batch é [B, 3, 224, 224]
        - Range de pixels após normalização está em uma faixa razoável
          (algo como [-2.1, 2.6] devido à normalização ImageNet)
        - Distribuição de labels é mista (não pegou só uma classe)
        - Imagens visualmente OK — você ABRE O PNG e CONFERE A OLHO
    """
    print("=" * 60)
    print("CHECK 1: PIPELINE (DataLoader + transforms + augmentation)")
    print("=" * 60)

    print(f"\nCriando DataLoaders (source={source})...")
    train_loader, val_loader = make_loaders(
        source=source,
        batch_size=8,
        num_workers=0,                 # 0 = single-process (mais simples pra debug)
        max_samples_train=n_samples_train,
        max_samples_val=n_samples_val,
    )
    print(f"  train_loader: {len(train_loader)} batches")
    print(f"  val_loader:   {len(val_loader)} batches")

    print("\nPegando primeiro batch de treino...")
    images, labels = next(iter(train_loader))

    print(f"  Shape:  {images.shape}")
    print(f"  Dtype:  {images.dtype}")
    print(f"  Range pixels (após normalização): "
          f"[{images.min():.3f}, {images.max():.3f}]")
    print(f"  Labels: {labels.tolist()}")
    print(f"  Distribuição: {Counter(labels.tolist())}")

    print(f"\nSalvando visualização em {output_path}...")
    images_denorm = denormalize(images)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, ax in enumerate(axes.flat):
        img = images_denorm[i].permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
        ax.imshow(img)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  → {output_path} (ABRA E CONFIRA visualmente)")

    print(f"\nDistribuição de labels no subset completo de treino "
          f"({n_samples_train} amostras):")
    all_labels = []
    for _, lbls in train_loader:
        all_labels.extend(lbls.tolist())
    print(f"  {Counter(all_labels)}")


def check_per_class(
    source: str = "hemg",
    split: str = "train",
    n_per_class: int = 16,
    output_dir: Path = Path("."),
    seed: int = 123,
) -> None:
    """Inspeção visual estratificada DO SUBSET QUE VAI PRO TREINO.

    Args:
        source: nome da fonte registrada em data/sources/. Default 'hemg'.
        split: 'train', 'val' ou 'test'. Default 'train' (o que mais importa).
        n_per_class: quantas amostras mostrar de cada classe (default 16).
        output_dir: pasta onde salvar os 2 PNGs (default: pasta atual).
        seed: semente da amostragem aleatória (reprodutível).

    Diferença pra antes: usa DetectIADataset (igual ao treino), então as
    amostras vêm DEPOIS do filtro de tamanho e do split. Antes a função
    estava amostrando do dataset cru, o que dava falsa impressão de qualidade
    (a maioria das amostras eram thumbnails 32×32 que nem entram no treino).

    O que valida:
        - As imagens que o modelo treina são realmente IA-geradas (label 0)
          ou reais (label 1)
        - Não há label noise no subset filtrado
        - Os tamanhos ORIGINAIS estão no range esperado (deveriam ser todas
          do tamanho do filtro — ex: 256×256)
    """
    import random

    print("\n" + "=" * 60)
    print(f"CHECK 2: AMOSTRAS POR CLASSE DO SUBSET DE TREINO")
    print(f"(source={source}, split={split}, {n_per_class} de cada classe)")
    print("=" * 60)

    print("\nCarregando subset (mesma pipeline do treino, sem transforms)...")
    dataset = DetectIADataset(source=source, split=split, transform=None)
    print(f"Tamanho do subset {split}: {len(dataset)}")

    # Amostragem aleatória até preencher 16 de cada classe.
    # Mais eficiente que iterar tudo: como o subset é balanceado (~50/50),
    # 32 amostras aleatórias quase sempre bastam pra preencher as duas cotas.
    print(f"\nSorteando {n_per_class} amostras de cada classe...")
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    samples_0, samples_1 = [], []
    for idx in indices:
        if len(samples_0) >= n_per_class and len(samples_1) >= n_per_class:
            break
        image, label = dataset[idx]
        if label == 0 and len(samples_0) < n_per_class:
            samples_0.append((idx, image, image.size))
        elif label == 1 and len(samples_1) < n_per_class:
            samples_1.append((idx, image, image.size))

    print(f"  Coletadas {len(samples_0)} de IA e {len(samples_1)} de Real")

    transform = get_eval_transforms()

    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nPlotando...")
    _plot_class_grid(
        samples_0, transform,
        title=f"Label 0 (IA) — subset {split} filtrado, sem augmentation",
        output_path=output_dir / "sanity_class_ai.png",
    )
    _plot_class_grid(
        samples_1, transform,
        title=f"Label 1 (Real) — subset {split} filtrado, sem augmentation",
        output_path=output_dir / "sanity_class_real.png",
    )
    """Inspeção visual estratificada por classe (sem augmentation).

    Args:
        n_per_class: quantas amostras mostrar de cada classe (default 16).
        output_dir: pasta onde salvar os 2 PNGs (default: pasta atual).
        seed: semente da amostragem aleatória (reprodutível).

    O que valida:
        - As imagens de label 0 são de fato IA-geradas (qualidade, estilo)
        - As imagens de label 1 são de fato fotos/arte real
        - Não há label noise óbvio (foto entre as IAs, ou IA entre as reais)
        - Os tamanhos ORIGINAIS são razoáveis (não há dezenas de miniaturas)

    O título de cada thumbnail inclui idx e tamanho ORIGINAL (antes do resize),
    pra detectar se uma classe é dominada por imagens minúsculas.
    """
    print("\n" + "=" * 60)
    print(f"CHECK 2: AMOSTRAS POR CLASSE ({n_per_class} de cada)")
    print("=" * 60)

    print("\nCarregando dataset cru (sem filtros, sem transforms)...")
    ds = load_dataset(HEMG_HF_NAME, split="train")

    print("Indexando por classe...")
    all_labels = ds["label"]
    indices_label_0 = [i for i, lbl in enumerate(all_labels) if lbl == 0]
    indices_label_1 = [i for i, lbl in enumerate(all_labels) if lbl == 1]
    print(f"  Total label 0 (IA):   {len(indices_label_0)}")
    print(f"  Total label 1 (Real): {len(indices_label_1)}")

    # Amostragem reprodutível com seed fixa
    generator = torch.Generator().manual_seed(seed)
    perm_0 = torch.randperm(len(indices_label_0), generator=generator).tolist()
    perm_1 = torch.randperm(len(indices_label_1), generator=generator).tolist()
    sample_indices_0 = [indices_label_0[i] for i in perm_0[:n_per_class]]
    sample_indices_1 = [indices_label_1[i] for i in perm_1[:n_per_class]]

    print(f"\nCarregando {n_per_class} amostras de cada classe...")
    samples_0 = [(idx, ds[idx]) for idx in sample_indices_0]
    samples_1 = [(idx, ds[idx]) for idx in sample_indices_1]

    # Aplica eval_transforms (SEM augmentation) — vê como o modelo veria a imagem
    transform = get_eval_transforms()

    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nPlotando...")
    _plot_class_grid(
        samples_0, transform,
        title="Label 0 (AiArtData) — supostamente IA-gerada",
        output_path=output_dir / "sanity_class_ai.png",
    )
    _plot_class_grid(
        samples_1, transform,
        title="Label 1 (RealArt) — supostamente Real",
        output_path=output_dir / "sanity_class_real.png",
    )


def _plot_class_grid(samples, transform, title: str, output_path: Path) -> None:
    """Helper: plota grid 4×4 de amostras COM tamanho original no título.

    Aplica eval_transforms (resize + normalize, SEM augmentation) pra mostrar
    exatamente o que o modelo veria na hora da inferência.
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i, (idx, image, original_size) in enumerate(samples):
        image_tensor = transform(image.convert("RGB"))
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
    plt.close()
    print(f"  → {output_path}")


def main():
    """Roda os dois sanity checks em sequência."""
    check_pipeline()
    check_per_class()

    print("\n" + "=" * 60)
    print("Sanity check completo. Abra os 3 PNGs e confira a olho:")
    print("  - sanity_pipeline.png")
    print("  - sanity_class_ai.png")
    print("  - sanity_class_real.png")
    print("=" * 60)


if __name__ == "__main__":
    main()