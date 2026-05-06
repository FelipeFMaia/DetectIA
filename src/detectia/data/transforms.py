"""Transformações de imagem para treino e avaliação.

IMPORTANTE: Este módulo é a fonte única de verdade para normalização.
Qualquer pré-processamento (treino, validação, inferência) DEVE importar daqui.
"""

import torchvision.transforms as T

# Constantes de normalização do ImageNet.
# Por que ImageNet? Porque vamos usar ResNet18 pré-treinada nele —
# os pesos esperam input normalizado dessa forma.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Tamanho de entrada padrão para ResNet (224x224)
IMAGE_SIZE = 224


def get_train_transforms() -> T.Compose:
    """
    Transformações para TREINO (com data augmentation).
    
    Augmentation aumenta a diversidade do dataset e reduz overfitting.
    Cuidado: nem toda augmentation faz sentido pra detecção de IA.
    Por exemplo, rotações fortes podem destruir artefatos típicos de geradores.
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),                         # Espelhamento horizontal
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Pequena variação de cor
        T.ToTensor(),                                          # PIL → Tensor [0, 1]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),     # Normaliza para distribuição do ImageNet
    ])


def get_eval_transforms() -> T.Compose:
    """
    Transformações para VALIDAÇÃO/TESTE/INFERÊNCIA (SEM augmentation).
    
    Avaliação tem que ser determinística: mesma imagem → mesma predição.
    Se tiver random crop ou flip aqui, suas métricas viram ruído.
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])