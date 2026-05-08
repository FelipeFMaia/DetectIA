"""Análise exploratória do dataset (EDA).

Faz três coisas, na ordem:
    1. Estrutura: imprime colunas, features e a primeira amostra do dataset.
    2. Labels:    imprime a distribuição global de labels (IA vs Real).
    3. Resoluções: lê o tamanho ORIGINAL de cada imagem (sem decodificar
       pixels — só o header), agrupa por classe e gera estatísticas + um PNG
       com 4 plots.

Por que ler só o header? Porque o Hemg tem ~152K imagens. Decodificar todas
levaria muitos minutos. Lendo só o header (PIL.Image.size), em ~20s você tem
largura×altura de tudo.

Como rodar (sempre do raiz do projeto):
    python scripts/eda_dataset.py
    python scripts/eda_dataset.py --output runs/eda/analise.png

Flags:
    --output  Caminho do PNG com os 4 plots de resoluções.
              Default: dataset_analysis.png (no diretório atual).
              Se passar um caminho dentro de uma pasta inexistente, ela é
              criada automaticamente.

Saídas no terminal:
    - Estrutura do dataset (colunas, features, primeira amostra)
    - Distribuição global de labels
    - Por classe: estatísticas de largura/altura, top 10 tamanhos
    - Distribuição por faixa de menor dimensão (útil pra escolher filtro)

Saída em arquivo:
    PNG com 4 subplots:
        (0,0) Histograma da menor dimensão (escala log)
        (0,1) Bar chart por faixa de resolução
        (1,0) Distribuição de aspect ratio
        (1,1) Scatter largura × altura (subset de 3000 por classe)
"""

import argparse
from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from datasets import Image, load_dataset
from tqdm import tqdm

HEMG_HF_NAME = "Hemg/AI-Generated-vs-Real-Images-Datasets"


def report_structure(ds) -> None:
    """Imprime a estrutura do dataset: colunas, features e primeira amostra.

    Útil pra membro novo entender o formato sem ter que ler doc do HuggingFace.
    """
    print("=" * 60)
    print("ESTRUTURA DO DATASET")
    print("=" * 60)
    print(f"Tamanho total: {len(ds)}")
    print(f"Colunas: {ds.column_names}")
    print(f"Features: {ds.features}")

    print("\n--- Primeira amostra ---")
    sample = ds[0]
    for k, v in sample.items():
        if hasattr(v, "size"):  # PIL.Image
            print(f"  {k}: PIL Image {v.size}, modo={v.mode}")
        else:
            print(f"  {k}: {v}")


def report_label_distribution(ds) -> None:
    """Imprime a distribuição global de labels.

    Convenção do Hemg: 0 = IA-gerada, 1 = Real.
    Pega a coluna inteira de uma vez (rápido — não itera amostra a amostra).
    """
    print("\n" + "=" * 60)
    print("DISTRIBUIÇÃO DE LABELS")
    print("=" * 60)
    all_labels = ds["label"]
    counter = Counter(all_labels)
    total = len(all_labels)
    print(f"Total: {total}")
    for label, count in sorted(counter.items()):
        nome = "IA-gerada" if label == 0 else "Real"
        pct = 100 * count / total
        print(f"  Label {label} ({nome}): {count} ({pct:.1f}%)")


def collect_sizes_per_class(ds) -> tuple[list, list]:
    """Coleta tamanhos (W, H) de cada imagem agrupados por classe.

    Usa decode=False: lê só o header de cada arquivo, não decodifica pixels.
    Pra dataset grande (152K imagens) isso vira segundos em vez de minutos.

    Returns:
        (sizes_label_0, sizes_label_1) — listas de tuplas (W, H).
    """
    print("\n" + "=" * 60)
    print("COLETANDO RESOLUÇÕES (lendo só o header de cada imagem)")
    print("=" * 60)

    # cast_column reconfigura a coluna 'image' pra retornar bytes brutos.
    ds_meta = ds.cast_column("image", Image(decode=False))

    sizes_label_0 = []  # IA
    sizes_label_1 = []  # Real

    for i in tqdm(range(len(ds_meta)), desc="Lendo headers"):
        item = ds_meta[i]
        with PILImage.open(BytesIO(item["image"]["bytes"])) as img:
            size = img.size  # (largura, altura) — não decodifica pixels
        if item["label"] == 0:
            sizes_label_0.append(size)
        else:
            sizes_label_1.append(size)

    print(f"\nTotal label 0 (IA):   {len(sizes_label_0)}")
    print(f"Total label 1 (Real): {len(sizes_label_1)}")
    return sizes_label_0, sizes_label_1


def report_resolution_stats(sizes_label_0: list, sizes_label_1: list) -> None:
    """Imprime estatísticas de resolução por classe + top 10 tamanhos + buckets.

    Estatísticas (por classe):
        - min/max/mediana/média de largura e altura
        - 10 tamanhos exatos mais frequentes (W×H + porcentagem)

    Buckets (combinando as duas classes lado a lado):
        - distribuição por faixa de menor dimensão (ex: <64, 64-127, ...)
        - útil pra decidir filtro de resolução pro treino
    """
    # Estatísticas por classe
    for nome, sizes in [("IA (label 0)", sizes_label_0),
                        ("Real (label 1)", sizes_label_1)]:
        widths = np.array([s[0] for s in sizes])
        heights = np.array([s[1] for s in sizes])

        print(f"\n=== {nome} ===")
        print(f"Largura  | min={widths.min()}, max={widths.max()}, "
              f"mediana={int(np.median(widths))}, média={int(widths.mean())}")
        print(f"Altura   | min={heights.min()}, max={heights.max()}, "
              f"mediana={int(np.median(heights))}, média={int(heights.mean())}")

        size_counter = Counter(sizes)
        print("\nTop 10 tamanhos mais frequentes:")
        for size, count in size_counter.most_common(10):
            pct = 100 * count / len(sizes)
            print(f"  {size[0]}×{size[1]}: {count} ({pct:.1f}%)")

    # Distribuição por menor dimensão (combina classes em um único relatório)
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
        print(f"{label:<12} {c0:>8} ({pct_0:>5.1f}%)      "
              f"{c1:>8} ({pct_1:>5.1f}%)")

def report_filtered_subset(
    ds,
    target_size: tuple[int, int] = (256, 256),
    cache_dir: str = "cache",
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> None:
    """Mostra estatísticas do SUBSET QUE VAI PRO TREINO (filtrado por tamanho).

    Esta é a única função do EDA que se importa com o que de fato vira treino.
    Reaproveita _get_filtered_indices de data/sources/hemg.py — então usa o
    mesmo cache que o pipeline de treino usa.

    Imprime: total filtrado, % do dataset original, balanço de classes
    no subset, e os tamanhos dos splits train/val/test.
    """
    from detectia.data.sources.hemg import _get_filtered_indices

    print("\n" + "=" * 60)
    print(f"SUBSET FILTRADO ({target_size[0]}×{target_size[1]}) "
          f"— O QUE VAI PRO TREINO")
    print("=" * 60)

    indices = _get_filtered_indices(target_size, Path(cache_dir))
    n_filtered = len(indices)
    pct_total = 100 * n_filtered / len(ds)

    # ds.select pega só os índices pedidos sem decodificar imagens
    labels_filtered = ds.select(indices)["label"]
    counter = Counter(labels_filtered)

    print(f"\nTotal após filtro: {n_filtered} imagens "
          f"({pct_total:.1f}% do dataset original de {len(ds)})")
    print(f"  Label 0 (IA):   {counter[0]} "
          f"({100 * counter[0] / n_filtered:.1f}%)")
    print(f"  Label 1 (Real): {counter[1]} "
          f"({100 * counter[1] / n_filtered:.1f}%)")

    n_train = int(split_ratios[0] * n_filtered)
    n_val = int(split_ratios[1] * n_filtered)
    n_test = n_filtered - n_train - n_val
    print(f"\nSplits {split_ratios[0]:.0%}/{split_ratios[1]:.0%}/"
          f"{split_ratios[2]:.0%} (mesmos de load_hemg):")
    print(f"  Train: {n_train}")
    print(f"  Val:   {n_val}")
    print(f"  Test:  {n_test}")

def plot_resolutions(sizes_label_0: list, sizes_label_1: list,
                     output_path: Path) -> None:
    """Gera 4 subplots num único PNG.

    (0,0) Histograma da menor dimensão em escala log — vê se o dataset tem
          imagens minúsculas (problemáticas pra resize 224×224).
    (0,1) Bar chart por faixa de resolução — comparação direta IA vs Real.
    (1,0) Distribuição de aspect ratio — vê se há viés (ex: IA tende ao quadrado).
    (1,1) Scatter largura × altura (subset 3000/classe) — padrões geométricos.
    """
    # Reaproveita os mesmos buckets da função de stats
    buckets = [0, 64, 128, 224, 512, 1024, 100000]
    bucket_labels = ["<64", "64-127", "128-223", "224-511", "512-1023", "≥1024"]
    min_dims_0 = np.array([min(s) for s in sizes_label_0])
    min_dims_1 = np.array([min(s) for s in sizes_label_1])
    counts_0 = np.histogram(min_dims_0, bins=buckets)[0]
    counts_1 = np.histogram(min_dims_1, bins=buckets)[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Histograma de menor dimensão (escala log)
    bins_log = np.logspace(np.log10(10), np.log10(5000), 50)
    axes[0, 0].hist(min_dims_0, bins=bins_log, alpha=0.6, color="red", label="IA (0)")
    axes[0, 0].hist(min_dims_1, bins=bins_log, alpha=0.6, color="blue", label="Real (1)")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Menor dimensão (px, log)")
    axes[0, 0].set_ylabel("Frequência")
    axes[0, 0].set_title("Distribuição de menor dimensão")
    axes[0, 0].axvline(224, color="black", linestyle="--", label="224 (alvo)")
    axes[0, 0].legend()

    # (0,1) Bar chart por bucket
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

    # (1,0) Aspect ratio
    aspects_0 = np.array([s[0]/s[1] for s in sizes_label_0])
    aspects_1 = np.array([s[0]/s[1] for s in sizes_label_1])
    axes[1, 0].hist(aspects_0, bins=50, alpha=0.6, color="red",
                    label="IA (0)", range=(0, 4))
    axes[1, 0].hist(aspects_1, bins=50, alpha=0.6, color="blue",
                    label="Real (1)", range=(0, 4))
    axes[1, 0].set_xlabel("Aspect ratio (largura/altura)")
    axes[1, 0].set_ylabel("Frequência")
    axes[1, 0].set_title("Distribuição de aspect ratio")
    axes[1, 0].legend()

    # (1,1) Scatter largura × altura (subset pra não poluir)
    n_sample = min(3000, len(sizes_label_0), len(sizes_label_1))
    rng = np.random.RandomState(42)  # seed fixa pra reprodutibilidade
    idx_0 = rng.choice(len(sizes_label_0), n_sample, replace=False)
    idx_1 = rng.choice(len(sizes_label_1), n_sample, replace=False)
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
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nPlot salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="EDA do dataset Hemg: estrutura, labels e resoluções.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_analysis.png",
        help=("Caminho do PNG com os 4 plots de resoluções. "
              "Default: dataset_analysis.png na pasta atual. "
              "Pastas inexistentes são criadas automaticamente."),
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Carregando dataset Hemg do HuggingFace...")
    print("(Primeira execução baixa ~851MB. Próximas usam o cache local.)\n")
    ds = load_dataset(HEMG_HF_NAME, split="train")

    report_structure(ds)
    report_label_distribution(ds)
    sizes_0, sizes_1 = collect_sizes_per_class(ds)
    report_resolution_stats(sizes_0, sizes_1)
    report_filtered_subset(ds)
    plot_resolutions(sizes_0, sizes_1, output_path)


if __name__ == "__main__":
    main()