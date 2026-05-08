# DetectIA

Detector de imagens geradas por IA. Projeto da extensão **Corvus AI** / USP São Carlos.

Pipeline completo de detecção binária (IA vs Real) usando transfer learning com ResNet18, treinado no dataset Hemg/AI-Generated-vs-Real-Images-Datasets (filtrado para 256×256).

## Resultados (test set, 3944 amostras)

| Métrica | Valor |
|---------|-------|
| Accuracy | 97.84% |
| ROC AUC | 0.9980 |
| PR AUC | 0.9981 |

## Setup

Requer Python 3.11+ e [uv](https://docs.astral.sh/uv/).

\`\`\`bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ".[dev]"
\`\`\`

## Estrutura

\`\`\`
detectIA/
├── configs/                    # Configs YAML de experimentos
│   ├── baseline.yaml           # Feature extraction (Opção 1)
│   ├── full_finetune.yaml      # Fine-tuning completo (Opção 2) — modelo final
│   └── sanity.yaml             # Treino curtíssimo pra debug
├── scripts/                    # CLIs (executados do raiz do projeto)
│   ├── eda_resolutions.py      # EDA: estrutura, labels, resoluções, subset filtrado
│   ├── sanity_check_data.py    # Sanity check: pipeline + amostras por classe (do subset de treino)
│   ├── sanity_check_model.py   # Sanity check do modelo
│   ├── train.py                # Treino
│   ├── evaluate.py             # Avaliação completa (métricas, plots, erros)
│   ├── predict_folder.py       # Inferência em pasta de imagens
│   ├── extract_curated_samples.py  # Extrai amostras TP/TN/FP/FN
│   └── gradcam_visualize.py    # Visualização Grad-CAM
└── src/detectia/
    ├── data/
    │   ├── transforms.py       # Normalização e augmentation
    │   ├── dataset.py          # DetectIADataset (orquestrador)
    │   ├── loaders.py          # Factory de DataLoaders
    │   └── sources/            # Loaders por fonte (uma fonte = um arquivo)
    │       └── hemg.py         # Hemg HuggingFace + filtro 256x256 + cache
    ├── models/
    │   └── classifier.py       # ResNet18 com transfer learning
    ├── training/
    │   ├── trainer.py          # Loop completo (AMP, scheduler, early stop, TB)
    │   └── metrics.py          # AverageMeter
    └── evaluation/
        ├── evaluator.py        # Inferência + carregamento de checkpoint
        ├── plots.py            # Matriz, ROC, PR, histograma
        └── gradcam.py          # Grad-CAM via hooks
\`\`\`

## Uso

### Treinar

\`\`\`bash
python scripts/train.py --config configs/full_finetune.yaml
\`\`\`

### Avaliar

\`\`\`bash
python scripts/evaluate.py --config configs/full_finetune.yaml \\
    --checkpoint runs/full_finetune/best.pt --split test
\`\`\`

### Inferência em uma pasta

\`\`\`bash
python scripts/predict_folder.py --folder samples_ood
\`\`\`

### Visualização Grad-CAM

\`\`\`bash
python scripts/gradcam_visualize.py --folder samples_test_curated --output gradcam.png
\`\`\`

## Decisões arquiteturais

- **`src/` layout** + instalação editable (`pip install -e .`)
- **Data layer modular**: cada fonte de dados em arquivo próprio em `data/sources/`. Migrar pra GLIDE/GenImage = adicionar `sources/glide.py` sem tocar no resto.
- **Configs YAML centralizadas** em `configs/`. Não há números mágicos no código.
- **Sempre executar do raiz do projeto.** Todos os paths são relativos à raiz.

## Limitações conhecidas

1. **Domain shift**: modelo foi treinado em dataset com geradores ~2023 (SD 1.4, GANs antigos). Em geradores SOTA modernos (Flux, Midjourney v6+), performance cai significativamente (~30% acc em teste OOD).
2. **Atalho parcial**: o modelo associa "qualidade de imagem incomum" com IA. Fotos reais nítidas e modernas podem ser classificadas erroneamente.
3. **Label noise**: o dataset Hemg contém algumas imagens com rotulagem incorreta.

Para uso em produção, considerar dataset moderno e técnicas adicionais (ver issues abertas).