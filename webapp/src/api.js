// ─────────────────────────────────────────────────────────────────────────────
// Cliente HTTP do DetectIA
//
// Responsabilidades:
//   1. Validar arquivo antes de enviar (tipo, tamanho)
//   2. Enviar imagem pro backend e parsear resposta
//   3. Detectar cold-start (servidor demorando pra acordar) e sinalizar pro UI
//   4. Traduzir erros de rede/servidor em mensagens amigáveis pro usuário
// ─────────────────────────────────────────────────────────────────────────────

import {
  API_URL,
  MAX_FILE_SIZE,
  ALLOWED_MIME_TYPES,
  COLD_START_THRESHOLD,
  REQUEST_TIMEOUT,
} from './lib/constants';

// Erro tipado — o componente que captura usa `err.code` pra escolher
// qual mensagem amigável mostrar ao usuário.
export class APIError extends Error {
  constructor(message, code) {
    super(message);
    this.name = 'APIError';
    this.code = code; // 'invalid_file' | 'invalid_type' | 'too_large' | 'network'
                      // | 'timeout' | 'server' | 'parse' | 'gallery_missing' | 'unknown'
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validação de arquivo (executada ANTES de qualquer upload)
// ─────────────────────────────────────────────────────────────────────────────
export function validateImageFile(file) {
  if (!file) {
    throw new APIError('Nenhum arquivo selecionado.', 'invalid_file');
  }
  if (!ALLOWED_MIME_TYPES.includes(file.type)) {
    throw new APIError('Formato não suportado. Use JPG, PNG ou WEBP.', 'invalid_type');
  }
  if (file.size > MAX_FILE_SIZE) {
    throw new APIError('Imagem muito grande. Máximo de 10 MB.', 'too_large');
  }
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Healthcheck — usado também como wake-up call em background.
// Retorna true/false sem nunca lançar (não queremos quebrar a Home se o
// servidor estiver dormindo).
// ─────────────────────────────────────────────────────────────────────────────
export async function checkHealth() {
  try {
    const res = await fetch(`${API_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

// Wake-up: o componente Home chama isso ao montar pra acordar o servidor
// enquanto o usuário lê a tela inicial. Reduz a chance de cold-start no
// primeiro `predict`.
export function wakeUpServer() {
  return checkHealth().catch(() => false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Análise principal — File → { probability: 0-100, gradcamBase64: string|null }
//
// Aceita callback opcional onProgress(phase) que recebe:
//   - 'cold_start': se a request passou de COLD_START_THRESHOLD sem responder
//
// O componente usa esse sinal pra trocar a mensagem do loading de
// "Analisando..." pra "O computador está acordando, só um instante..."
// ─────────────────────────────────────────────────────────────────────────────
export async function predictImage(file, { onProgress } = {}) {
  validateImageFile(file);

  const formData = new FormData();
  formData.append('file', file);

  // Timer dispara o callback se a request demorar muito.
  // Tem que ser cancelado em TODOS os caminhos abaixo (sucesso, erro, timeout).
  const coldStartTimer = setTimeout(() => {
    onProgress?.('cold_start');
  }, COLD_START_THRESHOLD);

  let response;
  try {
    response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(REQUEST_TIMEOUT),
    });
  } catch (err) {
    clearTimeout(coldStartTimer);
    if (err.name === 'TimeoutError' || err.name === 'AbortError') {
      throw new APIError('A análise demorou demais. Tente de novo.', 'timeout');
    }
    throw new APIError('Não foi possível conectar ao servidor.', 'network');
  }
  clearTimeout(coldStartTimer);

  if (!response.ok) {
    if (response.status === 413) throw new APIError('Imagem muito grande.', 'too_large');
    if (response.status === 415) throw new APIError('Formato não suportado.', 'invalid_type');
    if (response.status >= 500) throw new APIError('O servidor teve um problema. Tente de novo.', 'server');
    throw new APIError(`Erro inesperado (código ${response.status}).`, 'unknown');
  }

  let data;
  try {
    data = await response.json();
  } catch {
    throw new APIError('Resposta do servidor em formato inválido.', 'parse');
  }

  // Validação leve do contrato: o backend deve retornar `probability` em [0, 1].
  // Se o time mudar o contrato, atualiza aqui ou desencana com TypeScript depois.
  if (typeof data.probability !== 'number' || data.probability < 0 || data.probability > 1) {
    throw new APIError('Resposta do servidor incompleta.', 'parse');
  }

  return {
    probability: Math.round(data.probability * 100),
    gradcamBase64: data.gradcam_base64 || null,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: baixa uma imagem da galeria como File pra mandar pro backend.
// As imagens da galeria são servidas estaticamente pelo Vercel a partir de
// web/public/gallery/, então este fetch não toca o backend FastAPI.
// ─────────────────────────────────────────────────────────────────────────────
export async function fetchGalleryAsFile(galleryItem) {
  const res = await fetch(galleryItem.src);
  if (!res.ok) {
    throw new APIError('Esta imagem da galeria está indisponível.', 'gallery_missing');
  }
  const blob = await res.blob();
  return new File([blob], `gallery-${galleryItem.id}.jpg`, { type: blob.type });
}
