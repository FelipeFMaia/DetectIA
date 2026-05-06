"""Sanity check do modelo: forward pass com batch dummy.

Verifica:
1. Modelo é construído sem erro.
2. Forward pass roda e produz output de shape esperada.
3. Contagem de parâmetros bate com o esperado.
4. Comparação visual entre fine-tuning completo vs feature extraction.
"""

import torch

from detectia.models.classifier import build_model, count_parameters


def report(model_name, model):
    """Imprime contagem de parâmetros formatada."""
    p = count_parameters(model)
    print(f"  Parâmetros totais:     {p['total']:>14,}")
    print(f"  Parâmetros treináveis: {p['trainable']:>14,}")
    print(f"  Parâmetros congelados: {p['frozen']:>14,}")
    pct = 100 * p['trainable'] / p['total']
    print(f"  % treinável: {pct:.2f}%")


def main():
    # === Modelo 1: Fine-tuning completo (default) ===
    print("=" * 60)
    print("MODELO 1: Fine-tuning completo (freeze_backbone=False)")
    print("=" * 60)
    model_full = build_model(
        num_classes=1,
        pretrained=True,
        freeze_backbone=False,
    )
    report("full", model_full)
    
    # === Modelo 2: Feature extraction ===
    print("\n" + "=" * 60)
    print("MODELO 2: Feature extraction (freeze_backbone=True)")
    print("=" * 60)
    model_frozen = build_model(
        num_classes=1,
        pretrained=True,
        freeze_backbone=True,
    )
    report("frozen", model_frozen)
    
    # === Forward pass (com Modelo 1) ===
    print("\n" + "=" * 60)
    print("FORWARD PASS")
    print("=" * 60)
    dummy_batch = torch.randn(4, 3, 224, 224)
    
    model_full.eval()
    with torch.no_grad():
        logits = model_full(dummy_batch)
        probs = torch.sigmoid(logits)
    
    print(f"  Input shape:  {tuple(dummy_batch.shape)}")
    print(f"  Output shape: {tuple(logits.shape)}")
    print(f"  Logits crus (4 amostras): {logits.flatten().tolist()}")
    print(f"  Após sigmoid (P[Real]): {[round(p, 4) for p in probs.flatten().tolist()]}")
    
    print("\nNota: o modelo NÃO foi treinado ainda. Outputs são essencialmente")
    print("aleatórios — o que importa é que o forward pass FUNCIONA e produz")
    print("a shape correta [batch_size, 1].")


if __name__ == "__main__":
    main()