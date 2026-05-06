"""ResNet18 classifier para detecção de IA-gerada vs Real.

Convenção de output:
    Saída é UM logit (pré-sigmoid).
    output > 0  → modelo prevê classe 1 (Real)
    output < 0  → modelo prevê classe 0 (IA)
    sigmoid(output) ∈ [0, 1] = probabilidade de ser Real
    
    Loss apropriada: BCEWithLogitsLoss (aplica sigmoid + BCE de forma estável)
"""

import torch.nn as nn
from torchvision import models


def build_model(
    num_classes: int = 1,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Constroi ResNet18 com cabeça customizada pra classificação binária.
    
    Args:
        num_classes: Número de logits de saída.
            1 = binário com BCEWithLogitsLoss (recomendado, mais eficiente)
            2 = multi-classe com CrossEntropyLoss (mais flexível pra extender depois)
        pretrained: Se True, usa pesos do ImageNet. Se False, inicia aleatório.
        freeze_backbone: Se True, congela todas as camadas exceto a nova head
            (Opção 1 — feature extraction). Se False, treina tudo (Opção 2 —
            fine-tuning completo).
        dropout: Probabilidade de dropout antes da camada final.
            0.5 é padrão e ajuda contra overfitting.
    
    Returns:
        Modelo PyTorch pronto pra treino/inferência.
    """
    # Carrega backbone com ou sem pesos pré-treinados
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Congela backbone se for feature extraction
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Substitui a cabeça (FC final).
    # ResNet18 termina com `model.fc = Linear(512, 1000)` (1000 classes ImageNet).
    # Trocamos por: Dropout + Linear(512, num_classes)
    in_features = model.fc.in_features  # 512 pra ResNet18
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    # IMPORTANTE: Linear é criada DEPOIS do freeze, então requires_grad=True por
    # padrão. A nova head sempre é treinada, mesmo com freeze_backbone=True.
    
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Conta parâmetros treináveis vs congelados."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }