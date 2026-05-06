"""Extrai imagens curadas do test set pra análise visual.

Salva 4 categorias na pasta de saída (default: samples_test_curated/):
    tp_*  → Real corretamente classificado (alta P[Real])
    tn_*  → IA corretamente classificada (baixa P[Real])
    fp_*  → Modelo previu Real, era IA (mais confiantes)
    fn_*  → Modelo previu IA, era Real (mais confiantes)

Nome do arquivo embute idx do dataset e prob, pra rastreabilidade.
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from detectia.data.dataset import DetectIADataset
from detectia.data.transforms import get_eval_transforms
from detectia.evaluation.evaluator import evaluate_model, load_checkpoint
from detectia.models.classifier import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full_finetune.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/full_finetune/best.pt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="samples_test_curated")
    parser.add_argument("--per-category", type=int, default=2,
                       help="Quantas amostras por categoria (TP, TN, FP, FN)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset COM transform pra inferência
    print(f"=== Carregando dataset (split={args.split}) ===")
    dataset_eval = DetectIADataset(
        source=cfg["data"]["source"],
        split=args.split,
        transform=get_eval_transforms(),
    )
    print(f"Tamanho: {len(dataset_eval)}\n")
    
    loader = DataLoader(
        dataset_eval,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    
    # Modelo
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
    )
    model = load_checkpoint(model, args.checkpoint, device)
    
    # Inferência
    print("=== Avaliando ===")
    results = evaluate_model(model, loader, device=device)
    probs = results["probs"]
    labels = results["labels"]
    preds = results["preds"]
    
    # Seleção por categoria
    n = args.per_category
    
    tp_mask = (preds == 1) & (labels == 1)
    tp_idx = torch.nonzero(tp_mask).squeeze(-1)
    tp_top = tp_idx[torch.argsort(probs[tp_idx], descending=True)[:n]]
    
    tn_mask = (preds == 0) & (labels == 0)
    tn_idx = torch.nonzero(tn_mask).squeeze(-1)
    tn_top = tn_idx[torch.argsort(probs[tn_idx], descending=False)[:n]]
    
    fp_mask = (preds == 1) & (labels == 0)
    fp_idx = torch.nonzero(fp_mask).squeeze(-1)
    fp_top = fp_idx[torch.argsort(probs[fp_idx], descending=True)[:n]]
    
    fn_mask = (preds == 0) & (labels == 1)
    fn_idx = torch.nonzero(fn_mask).squeeze(-1)
    fn_top = fn_idx[torch.argsort(probs[fn_idx], descending=False)[:n]]
    
    # Re-cria dataset SEM transform pra salvar imagens originais (PIL)
    dataset_raw = DetectIADataset(
        source=cfg["data"]["source"],
        split=args.split,
        transform=None,
    )
    
    print(f"\n=== Salvando em {output_dir}/ ===")
    
    categories = [
        ("tp", tp_top, "TP — Real corretamente classificado"),
        ("tn", tn_top, "TN — IA corretamente classificada"),
        ("fp", fp_top, "FP — Modelo: Real | Verdadeiro: IA"),
        ("fn", fn_top, "FN — Modelo: IA | Verdadeiro: Real"),
    ]
    
    total = 0
    for prefix, indices, desc in categories:
        print(f"\n{desc}:")
        for idx in indices:
            idx_int = idx.item()
            img_pil, _ = dataset_raw[idx_int]
            prob = probs[idx_int].item()
            
            filename = f"{prefix}_idx{idx_int:05d}_p{prob:.3f}.png"
            img_pil.save(output_dir / filename)
            print(f"  → {filename}")
            total += 1
    
    print(f"\n=== {total} imagens salvas em {output_dir}/ ===")
    print(f"\nAgora roda:")
    print(f"  python scripts/gradcam_visualize.py --folder {output_dir} --output gradcam_test.png")


if __name__ == "__main__":
    main()