# DetectIA — Frontend

Site React (Vite) do projeto DetectIA, do grupo de extensão Corvus AI · USP São Carlos.

## Setup local (3 minutos)

```bash
cd web
npm install
npm run dev
```

Abre em `http://localhost:5173`.

Por padrão, o frontend tenta falar com o backend em `http://localhost:8000`. Suba o FastAPI da pasta `api/` em paralelo, ou use uma URL remota via env:

```bash
VITE_API_URL=https://seu-backend.hf.space npm run dev
```

## Deploy

Plug o repositório no Vercel. Configure:

- **Root directory**: `web`
- **Framework**: Vite (auto-detectado)
- **Environment Variable**: `VITE_API_URL` apontando pro backend de produção

Cada push na `main` faz deploy automático.

## O que vai pra `web/public/gallery/`

A equipe de curadoria coloca aqui as imagens finais (depois do modelo congelado):
- `img-01.jpg` até `img-NN.jpg` (mistura de reais e geradas por IA)
- Cada uma deve ter sido pré-validada no modelo final
- O array `GALLERY` em `src/lib/constants.js` lista a ordem e os títulos

Enquanto não houver imagens reais, o frontend cai num fallback do `picsum.photos` automaticamente — o site não quebra em dev.

## Mapa de arquivos

```
web/
  index.html             # lang pt-BR, OG tags, favicon SVG inline
  public/
    gallery/             # imagens curadas (a equipe preenche)
    og-image.png         # imagem 1200x630 pra preview no WhatsApp (a equipe gera)
  src/
    main.jsx             # entry point
    App.jsx              # tela única — todo o UI vive aqui
    api.js               # cliente HTTP (predict, healthcheck, validações)
    index.css            # Tailwind + ajustes de mobile
    lib/
      constants.js       # URL da API, paleta, galeria, limites — TROCAR AQUI
```

## Estados da aplicação

A análise de uma imagem passa por uma máquina de estados explícita em `App.jsx`:

```
idle  →  analyzing  →  ┌─ success
                       ├─ cold_start  →  success
                       └─ error
```

`cold_start` é disparado quando a request passa de 3s sem responder — útil pra avisar o idoso que o servidor está acordando do sleep do HF Spaces grátis.

## Pendências conhecidas

- Imagens reais da galeria (curadoria entrega após o modelo final)
- `og-image.png` em `public/` (gerar em qualquer ferramenta de design, 1200x630)
- Página "Sobre o projeto" (decisão pendente: vale a pena ou minimalismo basta?)
