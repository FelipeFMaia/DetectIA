"""Avaliação de modelo: roda inferência, retorna logits/labels/probs."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    use_amp: bool = True,
) -> dict:
    """
    Roda o modelo no loader inteiro e coleta predições.
    
    Returns:
        dict com:
            'logits': tensor [N] — saída crua do modelo
            'probs':  tensor [N] — sigmoid(logits) = P[classe=1=Real]
            'preds':  tensor [N] — 0 ou 1 (threshold 0.5)
            'labels': tensor [N] — ground truth
    """
    model.eval()
    model = model.to(device)
    
    all_logits, all_labels = [], []
    
    for images, labels in tqdm(loader, desc="Avaliando"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device == "cuda"):
            logits = model(images)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
    
    logits = torch.cat(all_logits).squeeze()
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    
    return {"logits": logits, "probs": probs, "preds": preds, "labels": labels}


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cuda") -> nn.Module:
    """Carrega pesos de um checkpoint salvo pelo Trainer."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint carregado: {checkpoint_path}")
    print(f"  Época salva: {ckpt['epoch']}")
    print(f"  {ckpt['monitor']}: {ckpt['metric']:.4f}")
    return model