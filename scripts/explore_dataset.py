import os
from collections import Counter
from datasets import load_dataset

print("Baixando/carregando Hemg... (851MB no primeiro uso, depois cacheado)")

# Recupera o token da variável de ambiente
hf_token = os.environ.get("HF_TOKEN")

# Adicionado o parâmetro token
ds = load_dataset(
    "Hemg/AI-Generated-vs-Real-Images-Datasets", 
    split="train",
    token=hf_token
)

print(f"\n=== Estrutura ===")
print(f"Tamanho total: {len(ds)}")
print(f"Colunas: {ds.column_names}")
print(f"Features: {ds.features}")

print(f"\n=== Primeira amostra ===")
sample = ds[0]
print(f"Chaves: {list(sample.keys())}")
for k, v in sample.items():
    if hasattr(v, 'size'):  # PIL Image
        print(f"  {k}: PIL Image {v.size}, modo={v.mode}")
    else:
        print(f"  {k}: {v}")

print(f"\n=== Distribuição de labels (amostra de 1000) ===")
sample_size = min(1000, len(ds))
labels = [ds[i]['label'] for i in range(sample_size)]
print(Counter(labels))

print(f"\n=== Variação de tamanhos (amostra de 50) ===")
sizes = [ds[i]['image'].size for i in range(50)]
print(f"Tamanhos únicos: {set(sizes)}")

#---------------------------------------------------------------------------------------

# Pega a coluna inteira de uma vez (rápido, não itera amostra por amostra)
all_labels = ds["label"]
print(f"Total: {len(all_labels)}")
print(f"Distribuição global: {Counter(all_labels)}")