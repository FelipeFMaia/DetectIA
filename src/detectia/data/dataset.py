"""Dataset customizado para detecção de imagens reais vs IA-geradas."""

from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from .sources.hemg import load_hemg

# Mapeia nome da source → função que carrega
SOURCE_LOADERS = {
    "hemg": load_hemg,
    # "glide": load_glide,  # futuro
}


class DetectIADataset(Dataset):
    """
    Dataset wrapper para imagens reais vs IA-geradas.
    
    Convenções de label:
        0 = IA-gerada
        1 = Real
    
    A lógica de carregamento por fonte vive em `sources/`.
    Adicionar uma fonte nova = criar `sources/<nome>.py` e registrar acima.
    """
    
    def __init__(
        self,
        source: str = "hemg",
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.source = source
        self.split = split
        self.transform = transform
        
        if source not in SOURCE_LOADERS:
            raise ValueError(
                f"Fonte desconhecida: {source}. Disponíveis: {list(SOURCE_LOADERS)}"
            )
        
        # Cada loader retorna (lista_de_amostras, getter)
        # samples = índices ou paths; getter(sample) → (PIL.Image, label_int)
        self.samples, self._getter = SOURCE_LOADERS[source](
            split=split, max_samples=max_samples, seed=seed
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self._getter(self.samples[idx])
        image = image.convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label