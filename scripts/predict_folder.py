"""Inferência em uma pasta de imagens (sem labels — teste OOD).

Carrega imagens, roda modelo, gera tabela no terminal + grid visual.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image

from detectia.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from detectia.evaluation.evaluator import load_checkpoint
from detectia.models.classifier import build_model

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def predict_image(model, image_path, transform, device):
    """Carrega uma imagem, roda modelo, retorna (tensor_preprocessado, prob_real)."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img)
    
    with torch.no_grad():
        logit = model(tensor.unsqueeze(0).to(device))
        prob = torch.sigmoid(logit).item()
    
    return tensor, prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full_finetune.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/full_finetune/best.pt")
    parser.add_argument("--folder", type=str, required=True,
                       help="Pasta com imagens (jpg, png, webp, etc.)")
    parser.add_argument("--output", type=str, default="ood_predictions.png")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Modelo + checkpoint
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
    )
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()
    
    # Lista imagens
    folder = Path(args.folder)
    image_paths = sorted([p for p in folder.iterdir()
                          if p.suffix.lower() in VALID_EXTENSIONS])
    
    if not image_paths:
        print(f"Nenhuma imagem encontrada em {folder}")
        return
    
    print(f"\n=== {len(image_paths)} imagens em {folder} ===\n")
    
    transform = get_eval_transforms()
    
    # Predições
    print(f"{'Arquivo':<45} {'P[Real]':>10} {'Predição':>10} {'Confiança':>12}")
    print("-" * 80)
    
    results = []
    for img_path in image_paths:
        tensor, prob = predict_image(model, img_path, transform, device)
        prediction = "Real" if prob > 0.5 else "IA"
        confidence = max(prob, 1 - prob)
        
        results.append({
            "path": img_path, "tensor": tensor, "prob": prob,
            "prediction": prediction, "confidence": confidence,
        })
        
        print(f"{img_path.name:<45} {prob:>10.4f} {prediction:>10} {confidence:>11.0%}")
    
    print()
    
    # Visualização em grid
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, r in enumerate(results):
        img_denorm = denormalize(r["tensor"])
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Cor do título: verde se muito confiante, laranja se médio, vermelho se em dúvida
        if r["confidence"] > 0.85:
            color = "darkgreen"
        elif r["confidence"] > 0.65:
            color = "darkorange"
        else:
            color = "red"
        
        ax = axes[i // cols, i % cols]
        ax.imshow(img_np)
        ax.set_title(
            f"{r['path'].name}\nP[Real]={r['prob']:.3f} → {r['prediction']} ({r['confidence']:.0%})",
            fontsize=10, color=color,
        )
        ax.axis("off")
    
    # Esconde axes vazias
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")
    
    fig.suptitle(
        "Predições em imagens fora da distribuição (OOD)\n"
        "Verde = alta confiança | Laranja = média | Vermelho = baixa",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=100, bbox_inches="tight")
    print(f"Grid salvo em: {args.output}")


if __name__ == "__main__":
    main()