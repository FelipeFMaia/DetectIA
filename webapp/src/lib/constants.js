// ─────────────────────────────────────────────────────────────────────────────
// Configuração do projeto DetectIA
// Toda a configuração que o time pode precisar trocar vive AQUI.
// ─────────────────────────────────────────────────────────────────────────────

// URL do backend FastAPI.
// - Em dev local: rodar `uvicorn main:app --reload` na pasta api/ e usar localhost.
// - Em produção: setar VITE_API_URL no painel do Vercel apontando pro HF Space ou Render.
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Limites de upload (alinhados com o limite que o backend deve aplicar).
export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
export const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

// Tempo (ms) até o frontend mostrar a mensagem "o computador está acordando".
// Servidores grátis (HF Spaces, Render) podem demorar 30-60s pra acordar do sleep.
export const COLD_START_THRESHOLD = 3000;

// Tempo total máximo que esperamos por uma resposta antes de desistir.
export const REQUEST_TIMEOUT = 60_000; // 60s

// ─────────────────────────────────────────────────────────────────────────────
// Galeria curada
//
// 16 imagens selecionadas do test set do dataset Hemg, em 4 faixas de
// confiança do modelo (2 reais + 2 IA por faixa):
//   - altíssima (modelo >95% certo) — exibe 0% ou 100%
//   - alta      (modelo 80-95%)     — exibe 6-14% ou 84-89%
//   - média     (modelo 60-80%)     — exibe 21-33% ou 68-76%
//   - baixa     (modelo 50-60%)     — exibe 44-48% ou 57-59%
//
// As mockProbability abaixo são valores REAIS retornados pelo modelo no
// test set, extraídos via scripts/extract_gallery_pool.py.
//
// Intercalação visual: zigue-zague entre faixas e classes, balanceando
// reais e IAs tanto por coluna (em 2 e 4 colunas) quanto por linha.
//
// `expectedAI` é o gabarito verdadeiro (não exibido ao usuário).
// `fallbackSeed` é um placeholder do picsum.photos pro caso da imagem
// real não estar no diretório (utilidade em dev local).
// ─────────────────────────────────────────────────────────────────────────────
export const GALLERY = [
  // ── Pos 1: altíssima Real
  { id: 1,  src: '/gallery/img-01.jpg', fallbackSeed: 236,  title: 'Torcedor de time',
    expectedAI: false, mockProbability: 0   },
  // ── Pos 2: baixa IA
  { id: 2,  src: '/gallery/img-02.jpg', fallbackSeed: 1414, title: 'Retrato em tons azulados',
    expectedAI: true,  mockProbability: 59  },
  // ── Pos 3: altíssima IA
  { id: 3,  src: '/gallery/img-03.jpg', fallbackSeed: 783,  title: 'Senhora sorrindo',
    expectedAI: true,  mockProbability: 100 },
  // ── Pos 4: alta Real
  { id: 4,  src: '/gallery/img-04.jpg', fallbackSeed: 2785, title: 'Mãe e filha abraçadas',
    expectedAI: false, mockProbability: 14  },
  // ── Pos 5: baixa Real
  { id: 5,  src: '/gallery/img-05.jpg', fallbackSeed: 1782, title: 'Senhora de olhos fechados',
    expectedAI: false, mockProbability: 48  },
  // ── Pos 6: alta IA
  { id: 6,  src: '/gallery/img-06.jpg', fallbackSeed: 374,  title: 'Criança alegre',
    expectedAI: true,  mockProbability: 89  },
  // ── Pos 7: média IA
  { id: 7,  src: '/gallery/img-07.jpg', fallbackSeed: 561,  title: 'Mulher em cena de filme',
    expectedAI: true,  mockProbability: 76  },
  // ── Pos 8: média Real
  { id: 8,  src: '/gallery/img-08.jpg', fallbackSeed: 1456, title: 'Idosa observando',
    expectedAI: false, mockProbability: 33  },
  // ── Pos 9: altíssima IA
  { id: 9,  src: '/gallery/img-09.jpg', fallbackSeed: 67,   title: 'Amigos no estádio',
    expectedAI: true,  mockProbability: 100 },
  // ── Pos 10: baixa Real
  { id: 10, src: '/gallery/img-10.jpg', fallbackSeed: 2219, title: 'Amigas na praia',
    expectedAI: false, mockProbability: 44  },
  // ── Pos 11: alta Real
  { id: 11, src: '/gallery/img-11.jpg', fallbackSeed: 2425, title: 'Selfie de homem',
    expectedAI: false, mockProbability: 6   },
  // ── Pos 12: baixa IA
  { id: 12, src: '/gallery/img-12.jpg', fallbackSeed: 2582, title: 'Entrevista na neve',
    expectedAI: true,  mockProbability: 57  },
  // ── Pos 13: média IA
  { id: 13, src: '/gallery/img-13.jpg', fallbackSeed: 1566, title: 'Japonesa em foto desfocada',
    expectedAI: true,  mockProbability: 68  },
  // ── Pos 14: altíssima Real
  { id: 14, src: '/gallery/img-14.jpg', fallbackSeed: 2287, title: 'Grupo de amigos',
    expectedAI: false, mockProbability: 0   },
  // ── Pos 15: média Real
  { id: 15, src: '/gallery/img-15.jpg', fallbackSeed: 993,  title: 'Homem brindando',
    expectedAI: false, mockProbability: 21  },
  // ── Pos 16: alta IA
  { id: 16, src: '/gallery/img-16.jpg', fallbackSeed: 1187, title: 'Cena de chuva',
    expectedAI: true,  mockProbability: 84  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Paleta — uma fonte só, importada por todos os componentes.
// Cores ajustadas para garantir contraste WCAG AA (4.5:1) em corpo de texto.
// ─────────────────────────────────────────────────────────────────────────────
export const C = {
  bg:      '#F5EFE6',
  ink:     '#1B2D24',
  accent:  '#8B4513',
  gold:    '#D4A574',
  fakeBg:  '#FBE9E1',
  fakeInk: '#7A2408', // escurecido (era #8B2D0A) — agora 7.1:1 sobre fakeBg
  realBg:  '#E8EFE2',
  warnBg:  '#FFF8E7',
  warnInk: '#5C4810',
  muted:   '#57534E', // stone-600 (era stone-500 borderline) — 7.2:1 sobre bg
};