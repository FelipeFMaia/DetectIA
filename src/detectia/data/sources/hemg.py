"""Loader para o dataset Hemg/AI-Generated-vs-Real-Images-Datasets."""

import json
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import torch
from datasets import Image, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm

HEMG_HF_NAME = "Hemg/AI-Generated-vs-Real-Images-Datasets"


def _get_filtered_indices(
    target_size: tuple[int, int],
    cache_dir: Path,
) -> list[int]:
    """
    Retorna índices das imagens cujo tamanho ORIGINAL bate com target_size.
    
    Tamanho segue convenção PIL: (largura, altura).
    
    Cache em disco evita re-iteração:
        - 1ª chamada: lê headers das 152K imagens (~20s)
        - Próximas: leitura do JSON (~ms)
    """
    cache_file = cache_dir / f"hemg_indices_{target_size[0]}x{target_size[1]}.json"
    
    if cache_file.exists():
        print(f"  → Carregando índices filtrados do cache: {cache_file}")
        with open(cache_file) as f:
            return json.load(f)
    
    print(f"  → Cache não encontrado. Computando índices para target_size={target_size}...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Carrega o dataset SEM decodificar imagens (rápido, só metadados)
    ds = load_dataset(HEMG_HF_NAME, split="train")
    ds_meta = ds.cast_column("image", Image(decode=False))
    
    valid_indices = []
    for i in tqdm(range(len(ds_meta)), desc="Filtrando por tamanho"):
        item = ds_meta[i]
        with PILImage.open(BytesIO(item["image"]["bytes"])) as img:
            if img.size == target_size:
                valid_indices.append(i)
    
    print(f"  → Encontradas {len(valid_indices)} imagens com tamanho {target_size}")
    
    with open(cache_file, "w") as f:
        json.dump(valid_indices, f)
    print(f"  → Cache salvo em: {cache_file}")
    
    return valid_indices


def load_hemg(
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
    target_size: Optional[tuple[int, int]] = (256, 256),
    cache_dir: str = "cache",
) -> tuple[list[int], Callable]:
    """
    Carrega o Hemg do HuggingFace.
    
    O Hemg vem com split único 'train' e ordenado por classe (todas IA, depois reais).
    Embaralhamos com seed fixa antes de splittar.
    
    Convenção de labels do Hemg:
        0 = AiArtData (IA-gerada)
        1 = RealArt (real)
    
    Args:
        split: "train", "val" ou "test".
        max_samples: Limita o número de amostras (útil pra debug).
        seed: Semente para divisão reprodutível dos splits.
        target_size: Filtra apenas imagens com este tamanho ORIGINAL exato (W, H).
            Default: (256, 256). Se None, usa todas as imagens (sem filtro).
        cache_dir: Diretório onde guardar o cache de índices filtrados.
    
    Returns:
        (samples, getter):
            samples: lista de índices nos dados originais
            getter: função idx -> (PIL.Image, label_int)
    """
    ds = load_dataset(HEMG_HF_NAME, split="train")
    
    # Filtro de tamanho (com cache)
    if target_size is not None:
        cache_path = Path(cache_dir)
        all_indices = _get_filtered_indices(target_size, cache_path)
    else:
        all_indices = list(range(len(ds)))
    
    # Embaralhamento reprodutível (Hemg vem ordenado por classe!)
    n = len(all_indices)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    shuffled = [all_indices[i] for i in perm]
    
    # Split 80/10/10
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    if split == "train":
        selected = shuffled[:n_train]
    elif split == "val":
        selected = shuffled[n_train : n_train + n_val]
    elif split == "test":
        selected = shuffled[n_train + n_val :]
    else:
        raise ValueError(f"Split inválido: {split}. Use 'train', 'val' ou 'test'.")
    
    if max_samples is not None:
        selected = selected[:max_samples]
    
    # Getter usa closure pra manter referência ao ds carregado
    def getter(real_idx: int) -> tuple:
        item = ds[real_idx]
        return item["image"], item["label"]
    
    return selected, getter