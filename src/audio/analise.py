"""Detecção de alterações vocais em áudios de consulta.

Os achados são de TRIAGEM: indicam que vale uma avaliação dirigida, não diagnosticam.
Além da acústica, o áudio é transcrito (Azure Speech to Text) e o texto é analisado
(Azure Text Analytics + léxico de termos críticos) — ver `azure_ia.py` e `texto.py`.
Na parte acústica, duas estratégias complementares:
* limiares populacionais (referências clássicas de análise acústica da voz);
* desvio em relação às gravações anteriores do próprio paciente (linha de base).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config
from ..schemas import Anomalia, Severidade
from . import azure_ia
from .azure_ia import AnaliseTexto, Transcricao, Trecho
from .features import FeaturesVoz, extrair_de_arquivo
from .texto import anomalias_do_texto

# atributos comparados com a linha de base: (nome legível, escala mínima do desvio)
ATRIBUTOS_BASE = {
    "f0_media": ("frequência fundamental", 6.0),
    "jitter_pct": ("jitter", 0.25),
    "shimmer_pct": ("shimmer", 0.8),
    "hnr_db": ("relação harmônico-ruído", 1.5),
    "fracao_pausa": ("proporção de pausas", 0.06),
    "f0_cv": ("variação de entonação", 0.02),
}

def analisar_features(f: FeaturesVoz, paciente_id: str, timestamp: datetime | None = None,
                      baseline: list[FeaturesVoz] | None = None) -> list[Anomalia]:
    cfg = config.AUDIO
    ts = timestamp or datetime.now()
    anomalias: list[Anomalia] = []

    def novo(tipo, sev, desc, detector="limiar acústico de referência", **ev):
        anomalias.append(Anomalia(paciente_id, "audio", tipo, sev, desc, ts, ev, detector))

    if f.fracao_vozeada < 0.05:
        novo("audio_sem_voz", Severidade.INFO, "Áudio sem voz suficiente para análise.", "controle de qualidade")
        return anomalias

    # --- qualidade vocal: perturbação e ruído ----------------------------------------
    sinais = []
    if f.jitter_pct > cfg["jitter_max_pct"]:
        sinais.append(f"jitter {f.jitter_pct:.2f}% (ref. < {cfg['jitter_max_pct']}%)")
    if f.shimmer_pct > cfg["shimmer_max_pct"]:
        sinais.append(f"shimmer {f.shimmer_pct:.1f}% (ref. < {cfg['shimmer_max_pct']}%)")
    if f.hnr_db < cfg["hnr_min_db"]:
        sinais.append(f"HNR {f.hnr_db:.1f} dB (ref. > {cfg['hnr_min_db']} dB)")
    if sinais:
        novo("disfonia", Severidade.MEDIA if len(sinais) >= 2 else Severidade.BAIXA,
             "Voz com instabilidade/soprosidade (" + "; ".join(sinais) + ") — padrão de disfonia; "
             "considerar avaliação laríngea (ex.: pós-extubação, paralisia de prega vocal, laringite).",
             jitter_pct=f.jitter_pct, shimmer_pct=f.shimmer_pct, hnr_db=f.hnr_db)

    # --- tremor vocal ------------------------------------------------------------------
    if f.tremor_indice > cfg["tremor_indice_max"]:
        novo("tremor_vocal", Severidade.MEDIA,
             f"Modulação rítmica da voz em ~{f.tremor_hz:.1f} Hz (índice {f.tremor_indice:.2f}) — tremor vocal; "
             "considerar avaliação neurológica (ex.: tremor essencial, doença de Parkinson).",
             tremor_indice=f.tremor_indice, tremor_hz=f.tremor_hz)

    # --- prosódia ----------------------------------------------------------------------
    if 0 < f.f0_cv < cfg["f0_cv_min"] and f.tremor_indice <= cfg["tremor_indice_max"]:
        novo("fala_monotona", Severidade.BAIXA,
             f"Entonação muito reduzida (variação de F0 de {100 * f.f0_cv:.1f}%) — fala monótona; pode acompanhar "
             "hipocinesia, sedação ou humor deprimido.", f0_cv=f.f0_cv, f0_media=f.f0_media)

    # --- pausas / fôlego -------------------------------------------------------------
    if f.fracao_pausa > cfg["pausa_frac_max"]:
        novo("pausas_excessivas", Severidade.MEDIA,
             f"{100 * f.fracao_pausa:.0f}% do tempo de fala em pausas ({f.n_pausas_longas} pausas longas) — "
             "fala entrecortada; avaliar dispneia, fadiga ou lentificação.",
             fracao_pausa=f.fracao_pausa, n_pausas_longas=f.n_pausas_longas)

    if f.tosses_por_min > cfg["tosse_por_min_max"]:
        novo("tosse_frequente", Severidade.BAIXA,
             f"Eventos de tosse frequentes (~{f.tosses_por_min:.0f}/min) durante a consulta.",
             tosses_por_min=f.tosses_por_min)

    # --- mudança em relação ao próprio paciente --------------------------------------
    if baseline and len(baseline) >= 3:
        desvios = []
        for attr, (nome, escala_min) in ATRIBUTOS_BASE.items():
            hist = np.array([getattr(b, attr) for b in baseline], dtype=float)
            med = float(np.median(hist))
            escala = max(float(np.median(np.abs(hist - med)) * 1.4826), escala_min)
            z = (getattr(f, attr) - med) / escala
            if abs(z) >= cfg["desvio_baseline_z"]:
                desvios.append((nome, med, getattr(f, attr), z))
        if desvios:
            texto = "; ".join(f"{n}: {a:.2f} → {b:.2f} (z = {z:+.1f})" for n, a, b, z in desvios)
            novo("mudanca_vocal", Severidade.MEDIA,
                 f"Voz diferente do padrão das {len(baseline)} consultas anteriores do paciente — {texto}.",
                 "desvio robusto vs. linha de base do paciente",
                 desvios={n: {"antes": round(a, 3), "agora": round(b, 3), "z": round(z, 2)} for n, a, b, z in desvios})
    return anomalias


@dataclass
class ResultadoAudio:
    features: FeaturesVoz
    anomalias: list[Anomalia]
    transcricao: Transcricao | None = None
    analise_texto: AnaliseTexto | None = None
    avisos: list[str] = field(default_factory=list)   # serviços indisponíveis, falhas de nuvem etc.

    def texto_dict(self) -> dict:
        """Resumo serializável da parte textual (para a API e o painel)."""
        d: dict = {"avisos": self.avisos}
        if self.transcricao:
            d["transcricao"] = self.transcricao.texto
            d["transcricao_origem"] = self.transcricao.origem
            d["trechos"] = [{"inicio_s": round(t.inicio_s, 2), "fim_s": round(t.fim_s, 2), "texto": t.texto,
                             "confianca": t.confianca} for t in self.transcricao.trechos]
        if self.analise_texto:
            d["analise_texto"] = self.analise_texto.to_dict()
        return d


def obter_transcricao(caminho: str | Path, avisos: list[str]) -> Transcricao | None:
    """Ordem: arquivo .txt ao lado do áudio (transcrição já existente) -> Azure Speech to Text
    -> faster-whisper local (se instalado) -> None."""
    lateral = Path(caminho).with_suffix(".txt")
    if lateral.exists():
        return Transcricao(lateral.read_text(encoding="utf-8").strip(), origem="arquivo .txt")
    if azure_ia.fala_configurada():
        try:
            from ..series.prescricoes import MEDICAMENTOS

            return azure_ia.transcrever(caminho, termos_dica=list(MEDICAMENTOS))
        except Exception as e:  # rede, cota, credencial: a análise acústica não pode parar por isso
            avisos.append(f"Azure Speech to Text falhou: {e}")
    else:
        avisos.append("Azure Speech to Text não configurado (AZURE_SPEECH_KEY/AZURE_SPEECH_REGION)")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    segmentos, _ = WhisperModel("small", compute_type="int8").transcribe(str(caminho), language="pt")
    trechos = [Trecho(s.start, s.end, s.text.strip()) for s in segmentos]
    return Transcricao(" ".join(t.texto for t in trechos), trechos, origem="faster-whisper (local)")


def analisar_transcricao(texto: str, paciente_id: str, timestamp: datetime | None = None,
                         historico_negativo: list[float] | None = None,
                         avisos: list[str] | None = None) -> tuple[AnaliseTexto | None, list[Anomalia]]:
    """Termos críticos (léxico local) + sentimento/frases-chave/entidades (Azure Text Analytics)."""
    avisos = avisos if avisos is not None else []
    analise = None
    if not texto.strip():
        return None, []
    if azure_ia.linguagem_configurada():
        try:
            analise = azure_ia.analisar_texto(texto)
        except Exception as e:
            avisos.append(f"Azure Text Analytics falhou: {e}")
    else:
        avisos.append("Azure Text Analytics não configurado (AZURE_LANGUAGE_KEY/AZURE_LANGUAGE_ENDPOINT): "
                      "termos críticos apenas pelo léxico local, sem sentimento")
    return analise, anomalias_do_texto(texto, paciente_id, timestamp, analise, historico_negativo)


def analisar_audio(caminho: str | Path, paciente_id: str, timestamp: datetime | None = None,
                   baseline: list[FeaturesVoz] | None = None, transcricao: str | None = None,
                   historico_negativo: list[float] | None = None) -> ResultadoAudio:
    """Análise completa de um áudio de consulta: voz (acústica) + fala (transcrição e texto)."""
    f = extrair_de_arquivo(caminho)
    res = ResultadoAudio(f, analisar_features(f, paciente_id, timestamp, baseline))
    res.transcricao = (Transcricao(transcricao, origem="fornecida") if transcricao is not None
                       else obter_transcricao(caminho, res.avisos))
    if res.transcricao and res.transcricao.texto:
        res.analise_texto, extras = analisar_transcricao(res.transcricao.texto, paciente_id, timestamp,
                                                         historico_negativo, res.avisos)
        res.anomalias += extras
    return res
