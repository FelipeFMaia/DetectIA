# DetectIA

Detector de imagens geradas por IA. Projeto da extensão **Corvus AI** / USP São Carlos.

Pipeline completo de detecção binária (IA vs Real) usando transfer learning com ResNet18, treinado no dataset Hemg/AI-Generated-vs-Real-Images-Datasets (filtrado para 256×256). Acompanha um site (React/Vite) com galeria curada para apresentação ao público leigo.

## Resultados (test set, 3944 amostras)

| Métrica | Valor |
|---------|-------|
| Accuracy | 97.84% |
| ROC AUC | 0.9980 |
| PR AUC | 0.9981 |

## Setup

### Modelo (Python 3.11+)

Requer [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ".[dev]"
```

### Site (Node 20+)

```bash
cd webapp
npm install
```

Detalhes do frontend em [`webapp/README.md`](webapp/README.md).

## Estrutura

```
detectIA/
├── configs/                    # Configs YAML de experimentos
│   ├── baseline.yaml           # Feature extraction (Opção 1)
│   ├── full_finetune.yaml      # Fine-tuning completo (Opção 2) — modelo final
│   └── sanity.yaml             # Treino curtíssimo pra debug
├── scripts/                    # CLIs (executados do raiz do projeto)
│   ├── eda_dataset.py          # EDA: estrutura, labels, resoluções, subset filtrado
│   ├── sanity_check_data.py    # Sanity check: pipeline + amostras por classe
│   ├── sanity_check_model.py   # Sanity check do modelo (forward pass)
│   ├── train.py                # Treino
│   ├── evaluate.py             # Avaliação completa (métricas, plots, erros)
│   ├── predict_folder.py       # Inferência em pasta de imagens
│   ├── extract_curated_samples.py  # Extrai amostras TP/TN/FP/FN
│   └── gradcam_visualize.py    # Visualização Grad-CAM
├── src/detectia/               # Código do modelo
│   ├── data/
│   │   ├── transforms.py       # Normalização e augmentation
│   │   ├── dataset.py          # DetectIADataset (orquestrador)
│   │   ├── loaders.py          # Factory de DataLoaders
│   │   └── sources/            # Loaders por fonte (uma fonte = um arquivo)
│   │       └── hemg.py         # Hemg HuggingFace + filtro 256x256 + cache
│   ├── models/
│   │   └── classifier.py       # ResNet18 com transfer learning
│   ├── training/
│   │   ├── trainer.py          # Loop completo (AMP, scheduler, early stop, TB)
│   │   └── metrics.py          # AverageMeter
│   └── evaluation/
│       ├── evaluator.py        # Inferência + carregamento de checkpoint
│       ├── plots.py            # Matriz, ROC, PR, histograma
│       └── gradcam.py          # Grad-CAM via hooks
└── webapp/                     # Frontend React + Vite + Tailwind
    └── README.md               # Detalhes do site (uso, estrutura, deploy)
```

## Uso (modelo)

### Análise exploratória do dataset

```bash
python scripts/eda_dataset.py
```

Imprime estrutura, distribuição de labels e estatísticas de resolução. Salva `dataset_analysis.png` com 4 plots.

### Sanity check do pipeline de dados

```bash
python scripts/sanity_check_data.py
```

Verifica que o DataLoader funciona, transforms são aplicados corretamente, e amostras por classe representam o subset usado no treino. Salva 3 PNGs pra inspeção visual.

### Sanity check do modelo

```bash
python scripts/sanity_check_model.py
```

Verifica que o modelo carrega, conta parâmetros, e roda um forward pass com batch dummy.

### Treinar

```bash
python scripts/train.py --config configs/full_finetune.yaml
```

### Avaliar

```bash
python scripts/evaluate.py --config configs/full_finetune.yaml \
    --checkpoint runs/full_finetune/best.pt --split test
```

### Inferência em uma pasta

```bash
python scripts/predict_folder.py --folder samples_ood
```

### Visualização Grad-CAM

```bash
python scripts/gradcam_visualize.py --folder samples_test_curated --output gradcam.png
```

## Site (webapp)

```bash
cd webapp
npm run dev
```

Abre em `http://localhost:5173`. Modo atual: galeria com gabaritos pré-computados (não depende de backend). Detalhes em [`webapp/README.md`](webapp/README.md).

## Decisões arquiteturais

- **`src/` layout** + instalação editable (`pip install -e .`)
- **Data layer modular**: cada fonte de dados em arquivo próprio em `data/sources/`. Migrar pra GLIDE/GenImage = adicionar `sources/glide.py` sem tocar no resto.
- **Configs YAML centralizadas** em `configs/`. Não há números mágicos no código.
- **Sempre executar do raiz do projeto.** Todos os paths são relativos à raiz.
- **Frontend desacoplado** em `webapp/`. Versão atual usa galeria curada com gabaritos pré-computados; backend FastAPI é trabalho futuro.

## Limitações conhecidas

1. **Domain shift**: modelo foi treinado em dataset com geradores ~2023 (SD 1.4, GANs antigos). Em geradores SOTA modernos (Flux, Midjourney v6+), a performance cai significativamente. A magnitude exata desse drop ainda não foi medida sistematicamente.
2. **Atalho parcial**: o modelo associa "qualidade de imagem incomum" com IA. Fotos reais nítidas e modernas podem ser classificadas erroneamente.
3. **Label noise**: o dataset Hemg contém algumas imagens com rotulagem incorreta.

Para uso em produção, considerar dataset moderno e técnicas adicionais.