"""Visualização Grad-CAM em uma pasta de imagens.

Pra cada imagem, gera um trio: original, heatmap, overlay.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from detectia.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from detectia.evaluation.evaluator import load_checkpoint
from detectia.evaluation.gradcam import GradCAM
from detectia.models.classifier import build_model

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def overlay_heatmap(image_np, heatmap, alpha=0.45):
    """Sobrepoe heatmap colorizado (jet) sobre imagem normalizada [0,1]."""
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap)[:, :, :3]  # RGB, descarta alpha
    overlay = (1 - alpha) * image_np + alpha * heatmap_colored
    return overlay.clip(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full_finetune.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/full_finetune/best.pt")
    parser.add_argument("--folder", type=str, required=True,
                       help="Pasta com imagens pra visualizar")
    parser.add_argument("--output", type=str, default="gradcam_grid.png")
    parser.add_argument("--target", type=str, default="predicted",
                       choices=["predicted", "real", "ai"],
                       help="Em qual classe focar o heatmap")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Modelo
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
    )
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()
    
    # Camada alvo: layer4 (último bloco conv da ResNet18)
    target_layer = model.layer4
    
    # Imagens
    folder = Path(args.folder)
    image_paths = sorted([p for p in folder.iterdir()
                          if p.suffix.lower() in VALID_EXTENSIONS])
    if not image_paths:
        print(f"Nenhuma imagem em {folder}")
        return
    
    print(f"\n=== {len(image_paths)} imagens em {folder} ===")
    print(f"Target: {args.target}\n")
    
    transform = get_eval_transforms()
    
    # Aplica Grad-CAM em cada imagem
    results = []
    with GradCAM(model, target_layer) as gradcam:
        for img_path in image_paths:
            img_pil = Image.open(img_path).convert("RGB")
            tensor = transform(img_pil).unsqueeze(0).to(device)
            
            heatmap, prob = gradcam(tensor, target_class=args.target)
            
            # Imagem desnormalizada pra display
            img_np = denormalize(tensor.squeeze().cpu()).permute(1, 2, 0).numpy()
            overlay = overlay_heatmap(img_np, heatmap)
            
            results.append({
                "path": img_path,
                "image": img_np,
                "heatmap": heatmap,
                "overlay": overlay,
                "prob": prob,
            })
            
            pred = "Real" if prob > 0.5 else "IA"
            print(f"{img_path.name:<35} P[Real]={prob:.4f} → {pred}")
    
    # Grid de visualização: 3 colunas (original, heatmap, overlay) × N linhas
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, r in enumerate(results):
        pred = "Real" if r["prob"] > 0.5 else "IA"
        title_color = "darkblue" if pred == "Real" else "darkred"
        
        axes[i, 0].imshow(r["image"])
        axes[i, 0].set_title(f"{r['path'].name}\nP[Real]={r['prob']:.3f} → {pred}",
                             fontsize=10, color=title_color)
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(r["heatmap"], cmap="jet")
        axes[i, 1].set_title("Heatmap (Grad-CAM)", fontsize=10)
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(r["overlay"])
        axes[i, 2].set_title("Overlay", fontsize=10)
        axes[i, 2].axis("off")
    
    fig.suptitle(
        f"Grad-CAM em layer4 — Target: {args.target}\n"
        f"(Vermelho/quente = região contribuiu mais pra a predição)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=100, bbox_inches="tight")
    print(f"\nSalvo: {args.output}")


if __name__ == "__main__":
    main()