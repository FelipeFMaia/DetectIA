"""CLI de avaliação completa: métricas, plots e análise de erros."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from detectia.data.dataset import DetectIADataset
from detectia.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from detectia.evaluation.evaluator import evaluate_model, load_checkpoint
from detectia.evaluation.plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_probability_histogram,
    plot_roc_curve,
)
from detectia.models.classifier import build_model


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Inverte normalização ImageNet pra visualização."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def plot_image_grid(dataset, indices, probs, labels, title, output_path, cols=4):
    """Plota grid de imagens com probabilidade e classe verdadeira no título."""
    n = len(indices)
    if n == 0:
        print(f"  [aviso] Sem imagens pra '{title}', pulando")
        return
    
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        idx_int = idx.item()
        image_tensor, _ = dataset[idx_int]
        image_denorm = denormalize(image_tensor)
        img = image_denorm.permute(1, 2, 0).numpy()
        
        prob = probs[idx_int].item()
        true_label = labels[idx_int].item()
        true_str = "Real" if true_label == 1 else "IA"
        
        ax = axes[i // cols, i % cols]
        ax.imshow(img)
        ax.set_title(f"P[Real]={prob:.3f}\nVerdadeiro: {true_str}", fontsize=10)
        ax.axis("off")
    
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def visualize_errors_and_uncertain(dataset, results, output_dir, n_per_category=8):
    """Visualiza FP/FN mais confiantes e casos próximos do threshold."""
    probs = results["probs"]
    labels = results["labels"]
    preds = results["preds"]
    
    # Falsos Positivos (modelo previu Real, mas era IA) — ordenados por prob descendente
    fp_mask = (preds == 1) & (labels == 0)
    fp_indices = torch.nonzero(fp_mask).squeeze(-1)
    if len(fp_indices) > 0:
        sorted_idx = torch.argsort(probs[fp_indices], descending=True)
        plot_image_grid(
            dataset, fp_indices[sorted_idx][:n_per_category], probs, labels,
            f"Falsos Positivos mais confiantes (modelo: Real, real: IA) — total: {len(fp_indices)}",
            output_dir / "errors_fp_confident.png",
        )
    print(f"  Falsos Positivos: {len(fp_indices)}")
    
    # Falsos Negativos (modelo previu IA, mas era Real) — ordenados por prob ascendente
    fn_mask = (preds == 0) & (labels == 1)
    fn_indices = torch.nonzero(fn_mask).squeeze(-1)
    if len(fn_indices) > 0:
        sorted_idx = torch.argsort(probs[fn_indices], descending=False)
        plot_image_grid(
            dataset, fn_indices[sorted_idx][:n_per_category], probs, labels,
            f"Falsos Negativos mais confiantes (modelo: IA, real: Real) — total: {len(fn_indices)}",
            output_dir / "errors_fn_confident.png",
        )
    print(f"  Falsos Negativos: {len(fn_indices)}")
    
    # Casos mais incertos (prob próxima de 0.5)
    uncertainty = (probs - 0.5).abs()
    uncertain_indices = torch.argsort(uncertainty)[:n_per_category]
    plot_image_grid(
        dataset, uncertain_indices, probs, labels,
        "Casos mais incertos (prob ≈ 0.5) — modelo em dúvida",
        output_dir / "uncertain_cases.png",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full_finetune.yaml",
                       help="Config usado no treino")
    parser.add_argument("--checkpoint", type=str, default="runs/full_finetune/best.pt",
                       help="Caminho do checkpoint")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir or
                      Path(cfg["logging"]["output_dir"]) / cfg["experiment"]["name"] / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Dataset (mesmas transforms de eval)
    print(f"=== Carregando dataset (split={args.split}) ===")
    dataset = DetectIADataset(
        source=cfg["data"]["source"],
        split=args.split,
        transform=get_eval_transforms(),
    )
    print(f"Tamanho: {len(dataset)}\n")
    
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    
    # Modelo
    print("=== Carregando modelo ===")
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
    )
    model = load_checkpoint(model, args.checkpoint, device)
    
    # Avaliação
    print(f"\n=== Avaliando no split '{args.split}' ===")
    results = evaluate_model(model, loader, device=device)
    
    # Métricas
    print("\n=== Classification Report ===")
    report = classification_report(
        results["labels"], results["preds"],
        target_names=["IA (0)", "Real (1)"], digits=4,
    )
    print(report)
    
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(f"Split: {args.split}\nCheckpoint: {args.checkpoint}\n\n{report}")
    
    # Plots de métricas
    print("\n=== Gerando plots ===")
    plot_confusion_matrix(results["labels"], results["preds"],
                         output_dir / "confusion_matrix.png")
    plot_roc_curve(results["labels"], results["probs"],
                  output_dir / "roc_curve.png")
    plot_pr_curve(results["labels"], results["probs"],
                 output_dir / "pr_curve.png")
    plot_probability_histogram(results["labels"], results["probs"],
                              output_dir / "probability_histogram.png")
    print(f"  → {output_dir}/[confusion_matrix, roc_curve, pr_curve, probability_histogram].png")
    
    # Análise de erros
    print("\n=== Análise de erros ===")
    visualize_errors_and_uncertain(dataset, results, output_dir)
    
    print(f"\n=== Avaliação completa. Resultados em: {output_dir} ===")


if __name__ == "__main__":
    main()