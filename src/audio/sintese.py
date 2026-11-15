"""Sintetizador de voz paramétrico para gerar áudios de teste com alterações controladas.

Não é fala inteligível: são trechos vozeados (fonte glotal harmônica + formantes) com
jitter, shimmer, ruído de soprosidade, tremor, pausas e tosses configuráveis — o
suficiente para validar a extração de biomarcadores sem usar gravações de pacientes reais.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def sintetizar_voz(duracao: float = 12.0, fs: int = 16000, f0: float = 125.0, jitter: float = 0.003,
                   shimmer: float = 0.012, ruido: float = 0.01, entonacao: float = 0.12,
                   tremor_hz: float = 0.0, tremor_prof: float = 0.0, pausa_frac: float = 0.18,
                   tosses: int = 0, semente: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semente)
    n_total = int(duracao * fs)
    saida = np.zeros(n_total, dtype=np.float64)
    harmonicos = np.arange(1, 26)

    pos = int(0.3 * fs)
    while pos < n_total - int(0.8 * fs):
        dur_fala = rng.uniform(1.6, 3.0)
        n_seg = min(int(dur_fala * fs), n_total - pos - int(0.2 * fs))
        # frequência e amplitude constantes dentro de cada ciclo glotal
        freq, amp = np.empty(n_seg), np.empty(n_seg)
        fase_ent, vel_ent = rng.uniform(0, 2 * np.pi), rng.uniform(0.35, 0.7)
        i = 0
        while i < n_seg:
            t = (pos + i) / fs
            f = f0 * (1 + entonacao * np.sin(2 * np.pi * vel_ent * t + fase_ent))
            if tremor_hz:
                f *= 1 + tremor_prof * np.sin(2 * np.pi * tremor_hz * t)
            f *= 1 + jitter * rng.standard_normal()
            n_ciclo = max(2, int(round(fs / f)))
            freq[i: i + n_ciclo] = f
            amp[i: i + n_ciclo] = 1 + shimmer * rng.standard_normal()
            i += n_ciclo
        fase = 2 * np.pi * np.cumsum(freq) / fs
        f_h = harmonicos[:, None] * freq[None, :]
        # envelope espectral: decaimento glotal + dois formantes (vogal /a/)
        ganho = (1 / harmonicos[:, None]) * (
            0.25 + np.exp(-((f_h - 700) / 180) ** 2) + 0.6 * np.exp(-((f_h - 1200) / 220) ** 2))
        ganho[f_h > 0.45 * fs] = 0
        seg = (ganho * np.sin(harmonicos[:, None] * fase[None, :])).sum(axis=0) * amp
        seg /= np.abs(seg).max() + 1e-9
        seg += ruido * 3.0 * rng.standard_normal(n_seg)  # soprosidade (reduz o HNR)
        rampa = int(0.03 * fs)
        env = np.ones(n_seg)
        env[:rampa], env[-rampa:] = np.linspace(0, 1, rampa), np.linspace(1, 0, rampa)
        saida[pos: pos + n_seg] += seg * env
        pausa = dur_fala * pausa_frac / max(1 - pausa_frac, 0.05) * rng.uniform(0.8, 1.2)
        pos += n_seg + int(max(pausa, 0.12) * fs)

    for _ in range(tosses):  # explosão de ruído com decaimento rápido
        ini = int(rng.uniform(0.5, duracao - 1.0) * fs)
        n = int(rng.uniform(0.15, 0.3) * fs)
        saida[ini: ini + n] = 1.3 * rng.standard_normal(n) * np.exp(-np.arange(n) / (0.07 * fs))

    saida += 0.0015 * rng.standard_normal(n_total)  # ruído de fundo da sala
    return (0.7 * saida / np.abs(saida).max()).astype(np.float32)


def salvar_wav(caminho: str | Path, x: np.ndarray, fs: int = 16000) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(caminho), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return caminho


PERFIS = {
    "normal": dict(),
    "disfonia": dict(jitter=0.025, shimmer=0.10, ruido=0.035),          # rouquidão/soprosidade
    "tremor": dict(tremor_hz=5.5, tremor_prof=0.06, entonacao=0.03),  # tremor vocal
    "monotona": dict(entonacao=0.008, pausa_frac=0.3),                # fala monótona
    "dispneia": dict(pausa_frac=0.72, ruido=0.04, tosses=3),           # pausas longas + tosse
}
