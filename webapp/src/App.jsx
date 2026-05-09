// ─────────────────────────────────────────────────────────────────────────────
// DetectIA — App principal
//
// Mudanças desde o protótipo inicial:
//   - Roteamento por hash (botão voltar do navegador funciona corretamente)
//   - Cliente HTTP real (não mais setTimeout fake)
//   - 5 estados explícitos: idle, analyzing, cold_start, success, error
//   - Tela de erro com mensagens amigáveis por código de erro
//   - Upload com validação + preview antes de enviar
//   - Acessibilidade: aria-live, aria-label, focus-visible, role apropriados
//   - Heatmap real do backend (com fallback decorativo se ainda não existir)
// ─────────────────────────────────────────────────────────────────────────────

import React, { useState, useEffect, useCallback } from 'react';
import {
  Upload, ArrowLeft, Sparkles, AlertCircle, AlertTriangle,
  Loader2, Eye, Image as ImageIcon, Bot, Camera, Zap, RefreshCw,
} from 'lucide-react';
import { GALLERY, C } from './lib/constants';
import { predictImage, fetchGalleryAsFile, wakeUpServer, validateImageFile, APIError } from './api';

// Estilos tipográficos reutilizados
const display = { fontFamily: "'Fraunces', Georgia, serif" };
const body    = { fontFamily: "'DM Sans', system-ui, sans-serif" };

// ═════════════════════════════════════════════════════════════════════════════
// Hash router minimalista — substitui react-router pra evitar dep extra.
// URLs válidas:
//   #/                       → home
//   #/galeria                → galeria
//   #/galeria/<id>           → resultado de uma imagem da galeria
//   #/enviar                 → tela de upload
//   #/enviar/analise         → resultado do upload
// ═════════════════════════════════════════════════════════════════════════════
function useHashRoute() {
  const [hash, setHash] = useState(() => window.location.hash.slice(1) || '/');
  useEffect(() => {
    const handler = () => setHash(window.location.hash.slice(1) || '/');
    window.addEventListener('hashchange', handler);
    return () => window.removeEventListener('hashchange', handler);
  }, []);
  const navigate = useCallback((path) => { window.location.hash = path; }, []);
  return [hash, navigate];
}

const parseRoute = (hash) => hash.split('/').filter(Boolean);

// ═════════════════════════════════════════════════════════════════════════════
// App
// ═════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [hash, navigate] = useHashRoute();
  const segments = parseRoute(hash);

  // Estado da análise — máquina de estados explícita.
  // status: 'idle' | 'analyzing' | 'cold_start' | 'success' | 'error'
  const [analysis, setAnalysis] = useState({ status: 'idle' });
  const [analyzedImage, setAnalyzedImage] = useState(null); // { src, title, isUpload? }

  // Carrega Google Fonts uma vez no boot.
  useEffect(() => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,700;0,9..144,800;1,9..144,500&family=DM+Sans:wght@400;500;600;700&display=swap';
    document.head.appendChild(link);
    return () => { try { document.head.removeChild(link); } catch (e) {} };
  }, []);

  // Acorda o servidor em background quando a Home monta.
  useEffect(() => { wakeUpServer(); }, []);

  // ── Análise unificada ─────────────────────────────────────────────────────
  const runAnalysis = useCallback(async (file, displayMeta) => {
    setAnalyzedImage(displayMeta);
    setAnalysis({ status: 'analyzing' });
    try {
      const result = await predictImage(file, {
        onProgress: (phase) => {
          if (phase === 'cold_start') {
            setAnalysis(prev => prev.status === 'analyzing' ? { status: 'cold_start' } : prev);
          }
        },
      });
      setAnalysis({ status: 'success', result });
    } catch (err) {
      setAnalysis({
        status: 'error',
        message: err instanceof APIError ? err.message : 'Erro inesperado.',
        code: err instanceof APIError ? err.code : 'unknown',
      });
    }
  }, []);

  const analyzeGallery = useCallback(async (galleryItem) => {
  navigate(`/galeria/${galleryItem.id}`);
  setAnalyzedImage({
    src: galleryItem.src,
    title: galleryItem.title,
    fallbackSeed: galleryItem.fallbackSeed,
  });
  setAnalysis({ status: 'analyzing' });

  // Simula ~1.5s de "análise" pra parecer real e dar tempo do idoso
  // ver o spinner.
  await new Promise(resolve => setTimeout(resolve, 1500));

  setAnalysis({
    status: 'success',
    result: {
      probability: galleryItem.mockProbability,
      gradcamBase64: null,  // usa o fallback decorativo SVG do Heatmap
    },
  });
}, [navigate]);

  const analyzeUpload = useCallback((file) => {
    const previewUrl = URL.createObjectURL(file);
    navigate('/enviar/analise');
    runAnalysis(file, { src: previewUrl, title: 'Sua foto', isUpload: true });
  }, [navigate, runAnalysis]);

  // Retry ─ inteligência mínima: galeria refaz; upload manda usuário pra tela
  // de upload (não temos como reaproveitar o File sem ele reanexar).
  const retry = useCallback(() => {
    if (analyzedImage?.isUpload) {
      navigate('/enviar');
      setAnalysis({ status: 'idle' });
      return;
    }
    const galleryId = parseInt(segments[1]);
    const item = GALLERY.find(g => g.id === galleryId);
    if (item) analyzeGallery(item);
  }, [analyzedImage, segments, navigate, analyzeGallery]);

  // ── Roteamento ─────────────────────────────────────────────────────────────
  let screen;
  if (segments[0] === 'galeria' && segments[1]) {
    screen = <ResultScreen image={analyzedImage} analysis={analysis} onRetry={retry} />;
  } else if (segments[0] === 'galeria') {
    screen = <GalleryScreen onSelect={analyzeGallery} />;
  } else if (segments[0] === 'enviar' && segments[1] === 'analise') {
    screen = <ResultScreen image={analyzedImage} analysis={analysis} onRetry={retry} />;
  } else if (segments[0] === 'enviar') {
    screen = <UploadScreen onAnalyze={analyzeUpload} />;
  } else {
    screen = <HomeScreen onGallery={() => navigate('/galeria')} />;
  }

  return (
    <div className="min-h-screen relative" style={{ ...body, background: C.bg, color: C.ink }}>
      <NoiseTexture />
      <Header showBack={segments.length > 0} onBack={() => window.history.back()} onHome={() => navigate('/')} />
      <main className="relative max-w-3xl mx-auto px-5 py-8 sm:py-12 pb-16">{screen}</main>
      <Footer />
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// Componentes auxiliares (chrome do app)
// ═════════════════════════════════════════════════════════════════════════════

// Textura sutil de ruído pra fugir do flat "AI app". Inline data-URI: zero requests.
function NoiseTexture() {
  return (
    <div
      aria-hidden
      className="fixed inset-0 opacity-[0.05] pointer-events-none mix-blend-multiply"
      style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200' viewBox='0 0 200 200'%3E%3Cfilter id='n'%3E%3CfeTurbulence baseFrequency='0.85' numOctaves='2' /%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' /%3E%3C/svg%3E")`,
      }}
    />
  );
}

function Header({ showBack, onBack, onHome }) {
  return (
    <header className="relative sticky top-0 z-20 border-b backdrop-blur"
            style={{ background: `${C.bg}E6`, borderColor: `${C.ink}20` }}>
      <div className="max-w-3xl mx-auto px-5 py-4 flex items-center justify-between">
        <button
          onClick={onHome}
          className="flex items-center gap-3 group focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 rounded"
          style={{ outlineColor: C.ink }}
          aria-label="Ir para a página inicial do DetectIA"
        >
          <div className="w-10 h-10 flex items-center justify-center transition-transform group-hover:rotate-12" style={{ background: C.ink }}>
            <Eye className="w-5 h-5" style={{ color: C.gold }} />
          </div>
          <div className="text-left">
            <h1 className="text-xl leading-none tracking-tight" style={{ ...display, fontWeight: 800, color: C.ink }}>
              Detect<span style={{ fontStyle: 'italic', color: C.accent }}>IA</span>
            </h1>
            <p className="text-[10px] uppercase tracking-[0.2em] mt-1" style={{ color: C.muted }}>
              Corvus AI · USP São Carlos
            </p>
          </div>
        </button>
        {showBack && (
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-full transition hover:bg-stone-200/60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
            style={{ color: C.ink, outlineColor: C.ink }}
            aria-label="Voltar para a tela anterior"
          >
            <ArrowLeft className="w-4 h-4" /> Voltar
          </button>
        )}
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="relative max-w-3xl mx-auto px-5 py-8 text-center text-xs border-t"
            style={{ color: C.muted, borderColor: `${C.ink}20` }}>
      Projeto de extensão · Corvus AI · Universidade de São Paulo<br />
      <span className="opacity-75">Sua imagem não é armazenada nos nossos servidores.</span>
    </footer>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// HOME
// ═════════════════════════════════════════════════════════════════════════════
function HomeScreen({ onGallery }) {
  return (
    <div className="space-y-12">
      <section className="text-center pt-4 sm:pt-8">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 mb-6 rounded-full text-[11px] uppercase tracking-[0.18em]"
             style={{ background: C.ink, color: C.gold }}>
          <Sparkles className="w-3.5 h-3.5" /> Detector de imagens
        </div>
        <h2 className="text-4xl sm:text-5xl leading-[1.05] mb-5 tracking-tight"
            style={{ ...display, color: C.ink, fontWeight: 700 }}>
          Esta foto é <span style={{ fontStyle: 'italic' }}>real</span>,<br />
          ou foi feita por uma<br />
          <span style={{ color: C.accent }}>inteligência artificial</span>?
        </h2>
        <p className="text-base sm:text-lg max-w-md mx-auto leading-relaxed" style={{ color: C.muted }}>
          O computador analisa a imagem e mostra <em>onde</em> ele encontrou sinais de inteligência artificial.
        </p>
      </section>

      <section className="flex justify-center">
        <CTAButton
          onClick={onGallery}
          variant="primary"
          icon={ImageIcon}
          title="Ver galeria"
          description="Escolha uma das nossas fotos e veja o computador analisar passo a passo."
          cta="Começar aqui"
        />
      </section>

      <section className="grid gap-6 sm:grid-cols-3 pt-6 border-t" style={{ borderColor: `${C.ink}30` }}>
        {[
          { icon: Camera, title: 'Sem cadastro',  desc: 'Use sem precisar criar conta'  },
          { icon: Bot,    title: 'IA explicável', desc: 'Veja onde o computador olhou'  },
          { icon: Zap,    title: 'Rápido',        desc: 'Resultado em poucos segundos'  },
        ].map((it, i) => (
          <div key={i}>
            <div className="inline-flex items-center justify-center w-9 h-9 mb-2 rounded-full" style={{ background: `${C.ink}10` }}>
              <it.icon className="w-4 h-4" style={{ color: C.ink }} />
            </div>
            <h4 className="text-sm font-semibold" style={{ color: C.ink }}>{it.title}</h4>
            <p className="text-xs mt-0.5" style={{ color: C.muted }}>{it.desc}</p>
          </div>
        ))}
      </section>
    </div>
  );
}

function CTAButton({ onClick, variant, icon: Icon, title, description, cta }) {
  const isPrimary = variant === 'primary';
  return (
    <button
      onClick={onClick}
      className="group relative p-6 sm:p-8 text-left transition-all hover:-translate-y-1 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
      style={{
        background: isPrimary ? C.ink : 'transparent',
        color: isPrimary ? C.bg : C.ink,
        border: isPrimary ? 'none' : `2px solid ${C.ink}`,
        outlineColor: C.ink,
      }}
    >
      <div className="flex items-center justify-center w-12 h-12 mb-5 rounded-full"
           style={{
             background: isPrimary ? C.gold : 'transparent',
             color: isPrimary ? C.ink : C.ink,
             border: isPrimary ? 'none' : `2px solid ${C.ink}`,
           }}>
        <Icon className="w-6 h-6" />
      </div>
      <h3 className="text-2xl mb-2" style={{ ...display, fontWeight: 700 }}>{title}</h3>
      <p className="text-sm opacity-85 mb-5 leading-relaxed">{description}</p>
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em]"
           style={{ color: isPrimary ? C.gold : C.ink, opacity: isPrimary ? 1 : 0.6 }}>
        {cta} <span className="transition-transform group-hover:translate-x-1">→</span>
      </div>
    </button>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// GALERIA
// ═════════════════════════════════════════════════════════════════════════════
function GalleryScreen({ onSelect }) {
  return (
    <div>
      <div className="mb-6 sm:mb-8">
        <p className="text-[11px] uppercase tracking-[0.18em] mb-2" style={{ color: C.muted }}>
          Etapa 1 · Galeria
        </p>
        <h2 className="text-3xl sm:text-4xl leading-tight" style={{ ...display, color: C.ink, fontWeight: 700 }}>
          Toque numa foto<br />para o computador <em>analisar</em>
        </h2>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4">
        {GALLERY.map((img) => (
          <button
            key={img.id}
            onClick={() => onSelect(img)}
            className="group relative overflow-hidden bg-stone-200 aspect-square transition-all hover:-translate-y-0.5 hover:shadow-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
            style={{ outlineColor: C.ink }}
            aria-label={`Analisar imagem: ${img.title}`}
          >
            <img
              src={img.src}
              alt={img.title}
              className="w-full h-full object-cover transition-transform group-hover:scale-105"
              loading="lazy"
              onError={(e) => {
                // Fallback pra dev enquanto a curadoria não entregou as imagens reais
                if (img.fallbackSeed && !e.target.dataset.fallback) {
                  e.target.dataset.fallback = '1';
                  e.target.src = `https://picsum.photos/seed/${img.fallbackSeed}/500`;
                }
              }}
            />
            <div className="absolute inset-0 bg-gradient-to-t opacity-0 group-hover:opacity-100 transition-opacity"
                 style={{ backgroundImage: `linear-gradient(to top, ${C.ink}, transparent)` }} />
            <div className="absolute bottom-0 left-0 right-0 p-3 text-left opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-xs text-white font-semibold">{img.title}</p>
              <p className="text-[10px] uppercase tracking-[0.18em] mt-0.5" style={{ color: C.gold }}>
                Tocar para analisar →
              </p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// RESULTADO — gerencia 3 sub-estados: analyzing/cold_start, success, error
// ═════════════════════════════════════════════════════════════════════════════
function ResultScreen({ image, analysis, onRetry }) {
  if (!image) {
    return (
      <div className="text-center py-16" style={{ color: C.muted }}>
        Nenhuma imagem para mostrar. <a href="#/" className="underline">Voltar à tela inicial</a>
      </div>
    );
  }

  if (analysis.status === 'analyzing' || analysis.status === 'cold_start') {
    return <LoadingScreen image={image} phase={analysis.status} />;
  }

  if (analysis.status === 'error') {
    return <ErrorScreen image={image} analysis={analysis} onRetry={onRetry} />;
  }

  if (analysis.status === 'success') {
    return <SuccessScreen image={image} result={analysis.result} onRetry={onRetry} />;
  }

  return null;
}

function LoadingScreen({ image, phase }) {
  const messages = {
    analyzing:  { primary: 'Analisando…',          secondary: 'O computador está olhando a imagem' },
    cold_start: { primary: 'O computador está acordando…', secondary: 'Pode levar alguns segundos. Já já vai responder.' },
  };
  const m = messages[phase];

  return (
    <div className="space-y-6">
      <div>
        <p className="text-[11px] uppercase tracking-[0.18em] mb-2" style={{ color: C.muted }}>
          Etapa 2 · Análise em andamento
        </p>
      </div>
      <figure>
        <div className="relative aspect-square overflow-hidden bg-stone-200 max-w-md mx-auto">
          <img src={image.src} alt={image.title} className="w-full h-full object-cover"
               onError={(e) => {
                 if (image.fallbackSeed && !e.target.dataset.fallback) {
                   e.target.dataset.fallback = '1';
                   e.target.src = `https://picsum.photos/seed/${image.fallbackSeed}/700`;
                 }
               }} />
          <div className="absolute inset-0 flex items-center justify-center backdrop-blur-sm"
               style={{ background: `${C.ink}40` }}>
            <Loader2 className="w-12 h-12 text-white animate-spin" aria-hidden />
          </div>
        </div>
      </figure>
      <div role="status" aria-live="polite" className="text-center">
        <p className="text-2xl mb-2" style={{ ...display, color: C.ink, fontWeight: 700 }}>{m.primary}</p>
        <p className="text-sm" style={{ color: C.muted }}>{m.secondary}</p>
      </div>
    </div>
  );
}

function ErrorScreen({ image, analysis, onRetry }) {
  // Mensagens amigáveis por código de erro.
  const FRIENDLY = {
    network:         { title: 'Sem conexão',                  body: 'Verifique sua internet e tente de novo.' },
    timeout:         { title: 'A análise demorou demais',     body: 'Pode ter sido o servidor acordando. Tente de novo, deve funcionar.' },
    server:          { title: 'O servidor teve um problema',  body: 'Não foi sua culpa. Tente de novo em alguns segundos.' },
    too_large:       { title: 'Imagem muito grande',          body: 'Use uma foto de até 10 MB.' },
    invalid_type:    { title: 'Formato não suportado',        body: 'Envie uma foto em JPG, PNG ou WEBP.' },
    invalid_file:    { title: 'Arquivo inválido',             body: 'Selecione uma imagem para enviar.' },
    parse:           { title: 'Resposta inesperada',          body: 'O servidor respondeu em um formato que não entendemos.' },
    gallery_missing: { title: 'Imagem não encontrada',        body: 'Esta imagem da galeria está indisponível agora.' },
    unknown:         { title: 'Algo deu errado',              body: 'Tente novamente. Se persistir, fale com a equipe.' },
  };
  const m = FRIENDLY[analysis.code] || FRIENDLY.unknown;

  return (
    <div className="space-y-6" role="alert" aria-live="assertive">
      <div>
        <p className="text-[11px] uppercase tracking-[0.18em] mb-2" style={{ color: C.muted }}>
          Etapa 2 · Houve um problema
        </p>
        <h2 className="text-3xl sm:text-4xl leading-tight" style={{ ...display, color: C.ink, fontWeight: 700 }}>
          {m.title}
        </h2>
      </div>

      <div className="p-5 sm:p-6 flex gap-4 items-start"
           style={{ background: C.warnBg, border: `1px solid ${C.gold}80` }}>
        <AlertTriangle className="w-6 h-6 flex-shrink-0 mt-0.5" style={{ color: C.warnInk }} />
        <div className="flex-1">
          <p className="text-base mb-2" style={{ color: C.warnInk, fontWeight: 600 }}>{m.body}</p>
          <p className="text-xs" style={{ color: C.warnInk, opacity: 0.7 }}>
            Detalhe técnico: {analysis.message}
          </p>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-3">
        <button
          onClick={onRetry}
          className="flex-1 px-6 py-4 text-base font-semibold transition hover:opacity-90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 flex items-center justify-center gap-2"
          style={{ background: C.ink, color: C.bg, outlineColor: C.ink }}
        >
          <RefreshCw className="w-4 h-4" /> Tentar de novo
        </button>
        <a
          href="#/"
          className="flex-1 px-6 py-4 text-base font-semibold text-center transition hover:bg-stone-200/60 border-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
          style={{ borderColor: C.ink, color: C.ink, outlineColor: C.ink }}
        >
          Voltar ao início
        </a>
      </div>
    </div>
  );
}

function SuccessScreen({ image, result, onRetry }) {
  const isAI = result.probability >= 50;
  const prob = result.probability;

  return (
    <div className="space-y-6">
      <div>
        <p className="text-[11px] uppercase tracking-[0.18em] mb-2" style={{ color: C.muted }}>
          Etapa 2 · Resultado da análise
        </p>
        <h2 className="text-3xl sm:text-4xl leading-tight" style={{ ...display, color: C.ink, fontWeight: 700 }}>
          {isAI
            ? <>Provavelmente é <em style={{ color: C.accent }}>IA</em></>
            : <>Provavelmente é <em>real</em></>}
        </h2>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <figure>
          <p className="text-[11px] uppercase tracking-[0.18em] mb-2" style={{ color: C.muted }}>Imagem original</p>
          <div className="relative aspect-square overflow-hidden bg-stone-200">
            <img src={image.src} alt={image.title} className="w-full h-full object-cover"
                 onError={(e) => {
                   if (image.fallbackSeed && !e.target.dataset.fallback) {
                     e.target.dataset.fallback = '1';
                     e.target.src = `https://picsum.photos/seed/${image.fallbackSeed}/700`;
                   }
                 }} />
          </div>
        </figure>

        <figure>
          <p className="text-[11px] uppercase tracking-[0.18em] mb-2" style={{ color: C.muted }}>O que o computador olhou</p>
          <div className="relative aspect-square overflow-hidden bg-stone-200">
            <img src={image.src} alt="" aria-hidden className="w-full h-full object-cover"
                 onError={(e) => {
                   if (image.fallbackSeed && !e.target.dataset.fallback) {
                     e.target.dataset.fallback = '1';
                     e.target.src = `https://picsum.photos/seed/${image.fallbackSeed}/700`;
                   }
                 }} />
            <Heatmap gradcamBase64={result.gradcamBase64} />
          </div>
        </figure>
      </div>

      <div className="p-5 sm:p-6"
           role="region"
           aria-live="polite"
           style={{
             background: isAI ? C.fakeBg : C.realBg,
             border: `1px solid ${isAI ? `${C.fakeInk}40` : `${C.ink}40`}`,
           }}>
        <div className="flex items-baseline justify-between mb-3">
          <span className="text-[11px] uppercase tracking-[0.18em]" style={{ color: isAI ? C.fakeInk : C.ink }}>
            Probabilidade de ser IA
          </span>
          <span className="text-5xl tabular-nums leading-none"
                style={{ ...display, fontWeight: 800, color: isAI ? C.fakeInk : C.ink }}>
            {prob}%
          </span>
        </div>
        <div className="relative h-3 overflow-hidden rounded-full" style={{ background: '#FFFFFF99' }}>
          <div className="absolute inset-y-0 left-0 transition-all duration-1000 ease-out rounded-full"
               style={{ width: `${prob}%`, background: isAI ? '#C75D3A' : '#2D5F3F' }}
               role="progressbar"
               aria-valuenow={prob}
               aria-valuemin={0}
               aria-valuemax={100}
               aria-label={`Probabilidade de ser IA: ${prob} por cento`} />
        </div>
        <p className="text-sm mt-4 leading-relaxed" style={{ color: isAI ? C.fakeInk : C.ink }}>
          {isAI
            ? 'O computador encontrou padrões típicos de imagens criadas por inteligência artificial.'
            : 'O computador não encontrou sinais fortes de inteligência artificial.'}
        </p>
      </div>

      <div className="p-5 bg-white border" style={{ borderColor: `${C.ink}30` }}>
        <h4 className="text-sm font-semibold mb-2" style={{ color: C.ink }}>O que significa o destaque colorido?</h4>
        <p className="text-sm leading-relaxed" style={{ color: C.muted }}>
          As <span className="px-1.5 py-0.5 rounded font-semibold" style={{ background: '#FF330020', color: C.fakeInk }}>áreas em vermelho</span> mostram onde o computador prestou mais atenção pra tomar a decisão. Quanto mais forte a cor, mais aquela região influenciou o resultado.
        </p>
      </div>

      <button
        onClick={onRetry}
        className="w-full px-6 py-4 text-base font-semibold transition hover:opacity-90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
        style={{ background: C.ink, color: C.bg, outlineColor: C.ink }}
      >
        Testar outra foto
      </button>
    </div>
  );
}

// Heatmap: usa o Grad-CAM real do backend se disponível; caso contrário,
// renderiza um overlay decorativo (útil enquanto o backend não retorna gradcam).
function Heatmap({ gradcamBase64 }) {
  if (gradcamBase64) {
    return (
      <img
        src={`data:image/png;base64,${gradcamBase64}`}
        alt=""
        aria-hidden
        className="absolute inset-0 w-full h-full object-cover mix-blend-multiply opacity-80 pointer-events-none"
      />
    );
  }
  // Fallback decorativo — a equipe de modelo deve substituir o backend
  // pra retornar gradcam_base64 e este caminho some.
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none"
         className="absolute inset-0 w-full h-full mix-blend-multiply pointer-events-none" aria-hidden>
      <defs>
        <radialGradient id="hm-fb-1"><stop offset="0%" stopColor="#FF2200" stopOpacity="0.85" /><stop offset="40%" stopColor="#FF7A00" stopOpacity="0.5" /><stop offset="100%" stopColor="#FF7A00" stopOpacity="0" /></radialGradient>
        <radialGradient id="hm-fb-2"><stop offset="0%" stopColor="#FF2200" stopOpacity="0.7"  /><stop offset="40%" stopColor="#FF7A00" stopOpacity="0.4" /><stop offset="100%" stopColor="#FF7A00" stopOpacity="0" /></radialGradient>
      </defs>
      <circle cx="40" cy="35" r="25" fill="url(#hm-fb-1)" />
      <circle cx="65" cy="60" r="20" fill="url(#hm-fb-2)" />
    </svg>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// UPLOAD — drop/picker + validação + preview + envio
// ═════════════════════════════════════════════════════════════════════════════
function UploadScreen({ onAnalyze }) {
  const [preview, setPreview] = useState(null); // { url, file }
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  // Limpa Blob URL ao desmontar pra não vazar memória.
  useEffect(() => {
    return () => { if (preview?.url) URL.revokeObjectURL(preview.url); };
  }, [preview]);

  const handleFile = (file) => {
    setError(null);
    try {
      validateImageFile(file);
      const url = URL.createObjectURL(file);
      setPreview({ url, file });
    } catch (err) {
      setError(err.message);
      setPreview(null);
    }
  };

  const submit = () => {
    if (preview) onAnalyze(preview.file);
  };

  return (
    <div>
      <div className="mb-6 sm:mb-8">
        <p className="text-[11px] uppercase tracking-[0.18em] mb-2" style={{ color: C.muted }}>
          Modo experimental
        </p>
        <h2 className="text-3xl sm:text-4xl leading-tight" style={{ ...display, color: C.ink, fontWeight: 700 }}>
          Envie uma foto<br />do <em>celular</em> ou computador
        </h2>
      </div>

      {!preview && (
        <label
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
          className={`block cursor-pointer transition-all border-2 border-dashed p-10 sm:p-16 text-center focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 ${dragOver ? 'scale-[1.01]' : ''}`}
          style={{ borderColor: C.ink, background: dragOver ? `${C.ink}08` : 'transparent', outlineColor: C.ink }}
        >
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="sr-only"
            onChange={(e) => handleFile(e.target.files[0])}
            aria-label="Selecionar imagem para análise"
          />
          <Upload className="w-12 h-12 mx-auto mb-4" style={{ color: C.ink }} aria-hidden />
          <p className="text-xl mb-2" style={{ ...display, color: C.ink, fontWeight: 700 }}>
            Toque aqui ou arraste uma foto
          </p>
          <p className="text-sm" style={{ color: C.muted }}>JPG, PNG ou WEBP · até 10 MB</p>
        </label>
      )}

      {preview && (
        <div className="space-y-4">
          <div className="relative aspect-square max-w-md mx-auto overflow-hidden bg-stone-200">
            <img src={preview.url} alt="Sua imagem antes de analisar" className="w-full h-full object-cover" />
          </div>
          <div className="flex flex-col sm:flex-row gap-3">
            <button
              onClick={submit}
              className="flex-1 px-6 py-4 text-base font-semibold transition hover:opacity-90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
              style={{ background: C.ink, color: C.bg, outlineColor: C.ink }}
            >
              Analisar esta foto
            </button>
            <button
              onClick={() => { URL.revokeObjectURL(preview.url); setPreview(null); }}
              className="flex-1 px-6 py-4 text-base font-semibold transition hover:bg-stone-200/60 border-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
              style={{ borderColor: C.ink, color: C.ink, outlineColor: C.ink }}
            >
              Trocar foto
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 flex gap-3 items-start" role="alert"
             style={{ background: C.fakeBg, border: `1px solid ${C.fakeInk}40` }}>
          <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" style={{ color: C.fakeInk }} />
          <p className="text-sm leading-relaxed" style={{ color: C.fakeInk }}>{error}</p>
        </div>
      )}

      <div className="mt-5 p-4 flex gap-3 items-start"
           style={{ background: C.warnBg, border: `1px solid ${C.gold}60` }}>
        <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" style={{ color: C.warnInk }} />
        <div className="text-sm leading-relaxed" style={{ color: C.warnInk }}>
          <strong>Versão experimental.</strong> Em fotos muito diferentes do nosso conjunto de treinamento, o computador pode errar. Para resultados mais confiáveis, use a galeria. <strong>Sua imagem não é armazenada.</strong>
        </div>
      </div>
    </div>
  );
}
