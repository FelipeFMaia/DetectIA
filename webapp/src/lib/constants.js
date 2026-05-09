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
// A equipe de curadoria troca este array depois que o modelo final estiver pronto:
//   1. Selecionam ~60 candidatas (30 reais, 30 IA)
//   2. Rodam todas no modelo final
//   3. Mantêm só as que classificam corretamente com >75% de confiança
//   4. Colocam os arquivos em web/public/gallery/ e atualizam este array
// `expectedAI` é o gabarito conhecido (não exibido pro usuário, mas útil pra QA).
// `fallbackSeed` é só pra dev: enquanto não houver imagem real, usa picsum.photos.
// ─────────────────────────────────────────────────────────────────────────────
export const GALLERY = [
  { id: 1, src: '/gallery/img-01.jpg', fallbackSeed: 1018, title: 'Lago ao amanhecer',
    expectedAI: false, mockProbability: 7  },
  { id: 2, src: '/gallery/img-02.jpg', fallbackSeed: 837,  title: 'Retrato em estúdio',
    expectedAI: true,  mockProbability: 92 },
  { id: 3, src: '/gallery/img-03.jpg', fallbackSeed: 1025, title: 'Cachorro no parque',
    expectedAI: false, mockProbability: 11 },
  { id: 4, src: '/gallery/img-04.jpg', fallbackSeed: 433,  title: 'Cidade ao entardecer',
    expectedAI: true,  mockProbability: 88 },
  { id: 5, src: '/gallery/img-05.jpg', fallbackSeed: 312,  title: 'Café da manhã',
    expectedAI: false, mockProbability: 14 },
  { id: 6, src: '/gallery/img-06.jpg', fallbackSeed: 219,  title: 'Paisagem onírica',
    expectedAI: true,  mockProbability: 95 },
  { id: 7, src: '/gallery/img-07.jpg', fallbackSeed: 156,  title: 'Rua histórica',
    expectedAI: false, mockProbability: 9  },
  { id: 8, src: '/gallery/img-08.jpg', fallbackSeed: 595,  title: 'Animal fantástico',
    expectedAI: true,  mockProbability: 89 },
  { id: 9, src: '/gallery/img-09.jpg', fallbackSeed: 421,  title: 'Praia tranquila',
    expectedAI: false, mockProbability: 13 },
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
