"""Análise exploratória de resoluções do dataset Hemg.

Pega o tamanho ORIGINAL de cada imagem (sem decodificar pixels — só header),
agrupa por classe, e gera visualizações pra fundamentar a decisão de filtro.
"""

from collections import Counter
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from datasets import load_dataset, Image
from tqdm import tqdm


def main():
    print("Carregando dataset com decode=False (rápido, só metadados)...")
    ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")
    ds = ds.cast_column("image", Image(decode=False))
    
    print(f"Total: {len(ds)} imagens\n")
    
    sizes_label_0 = []  # IA
    sizes_label_1 = []  # Real
    
    print("Coletando tamanhos (lê só o header de cada arquivo)...")
    for i in tqdm(range(len(ds))):
        item = ds[i]
        with PILImage.open(BytesIO(item["image"]["bytes"])) as img:
            size = img.size  # (width, height) — não decodifica pixels
        
        if item["label"] == 0:
            sizes_label_0.append(size)
        else:
            sizes_label_1.append(size)
    
    print(f"\nLabel 0 (IA): {len(sizes_label_0)}")
    print(f"Label 1 (Real): {len(sizes_label_1)}")
    
    # Estatísticas e top tamanhos por classe
    for name, sizes in [("IA (label 0)", sizes_label_0), ("Real (label 1)", sizes_label_1)]:
        widths = np.array([s[0] for s in sizes])
        heights = np.array([s[1] for s in sizes])
        
        print(f"\n=== {name} ===")
        print(f"Largura  | min={widths.min()}, max={widths.max()}, mediana={int(np.median(widths))}, média={int(widths.mean())}")
        print(f"Altura   | min={heights.min()}, max={heights.max()}, mediana={int(np.median(heights))}, média={int(heights.mean())}")
        
        size_counter = Counter(sizes)
        print(f"\nTop 10 tamanhos mais frequentes:")
        for size, count in size_counter.most_common(10):
            pct = 100 * count / len(sizes)
            print(f"  {size[0]}×{size[1]}: {count} ({pct:.1f}%)")
    
    # Buckets por menor dimensão (mais útil que largura ou altura sozinhas)
    buckets = [0, 64, 128, 224, 512, 1024, 100000]
    bucket_labels = ["<64", "64-127", "128-223", "224-511", "512-1023", "≥1024"]
    
    min_dims_0 = np.array([min(s) for s in sizes_label_0])
    min_dims_1 = np.array([min(s) for s in sizes_label_1])
    
    counts_0 = np.histogram(min_dims_0, bins=buckets)[0]
    counts_1 = np.histogram(min_dims_1, bins=buckets)[0]
    
    print("\n" + "=" * 60)
    print("DISTRIBUIÇÃO POR MENOR DIMENSÃO (largura ou altura)")
    print("=" * 60)
    print(f"{'Faixa':<12} {'IA (0)':<22} {'Real (1)':<22}")
    print("-" * 60)
    for label, c0, c1 in zip(bucket_labels, counts_0, counts_1):
        pct_0 = 100 * c0 / len(sizes_label_0)
        pct_1 = 100 * c1 / len(sizes_label_1)
        print(f"{label:<12} {c0:>8} ({pct_0:>5.1f}%)      {c1:>8} ({pct_1:>5.1f}%)")
    
    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histograma de menor dimensão (log)
    bins_log = np.logspace(np.log10(10), np.log10(5000), 50)
    axes[0, 0].hist(min_dims_0, bins=bins_log, alpha=0.6, color="red", label="IA (0)")
    axes[0, 0].hist(min_dims_1, bins=bins_log, alpha=0.6, color="blue", label="Real (1)")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Menor dimensão (px, log)")
    axes[0, 0].set_ylabel("Frequência")
    axes[0, 0].set_title("Distribuição de menor dimensão")
    axes[0, 0].axvline(224, color="black", linestyle="--", label="224 (alvo)")
    axes[0, 0].legend()
    
    # 2. Bar chart por bucket
    x = np.arange(len(bucket_labels))
    w = 0.35
    axes[0, 1].bar(x - w/2, counts_0, w, label="IA (0)", color="red", alpha=0.7)
    axes[0, 1].bar(x + w/2, counts_1, w, label="Real (1)", color="blue", alpha=0.7)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(bucket_labels, rotation=15)
    axes[0, 1].set_xlabel("Faixa de resolução mínima")
    axes[0, 1].set_ylabel("Quantidade")
    axes[0, 1].set_title("Imagens por faixa de resolução")
    axes[0, 1].legend()
    
    # 3. Aspect ratio
    aspects_0 = np.array([s[0]/s[1] for s in sizes_label_0])
    aspects_1 = np.array([s[0]/s[1] for s in sizes_label_1])
    axes[1, 0].hist(aspects_0, bins=50, alpha=0.6, color="red", label="IA (0)", range=(0, 4))
    axes[1, 0].hist(aspects_1, bins=50, alpha=0.6, color="blue", label="Real (1)", range=(0, 4))
    axes[1, 0].set_xlabel("Aspect ratio (largura/altura)")
    axes[1, 0].set_ylabel("Frequência")
    axes[1, 0].set_title("Distribuição de aspect ratio")
    axes[1, 0].legend()
    
    # 4. Scatter largura x altura (subset pra não poluir)
    n_sample = min(3000, len(sizes_label_0), len(sizes_label_1))
    idx_0 = np.random.RandomState(42).choice(len(sizes_label_0), n_sample, replace=False)
    idx_1 = np.random.RandomState(42).choice(len(sizes_label_1), n_sample, replace=False)
    w_0 = [sizes_label_0[i][0] for i in idx_0]
    h_0 = [sizes_label_0[i][1] for i in idx_0]
    w_1 = [sizes_label_1[i][0] for i in idx_1]
    h_1 = [sizes_label_1[i][1] for i in idx_1]
    axes[1, 1].scatter(w_0, h_0, s=2, alpha=0.3, c="red", label="IA (0)")
    axes[1, 1].scatter(w_1, h_1, s=2, alpha=0.3, c="blue", label="Real (1)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Largura (log)")
    axes[1, 1].set_ylabel("Altura (log)")
    axes[1, 1].set_title(f"Largura × Altura ({n_sample} amostras por classe)")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("dataset_analysis.png", dpi=100, bbox_inches="tight")
    print("\nSalvo: dataset_analysis.png")


if __name__ == "__main__":
    main()