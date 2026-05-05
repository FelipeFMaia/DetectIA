# DetectIA

Detector de imagens geradas por IA. Projeto da extensão Corvus AI / USP São Carlos.

## Setup

\`\`\`bash
uv venv --python 3.11
uv pip install -e ".[dev]"
\`\`\`

## Estrutura

- `src/detectia/data/` — carregamento e transforms
- `src/detectia/models/` — arquitetura
- `src/detectia/training/` — loop de treino
- `src/detectia/evaluation/` — métricas e Grad-CAM
- `scripts/` — CLIs (train, evaluate)