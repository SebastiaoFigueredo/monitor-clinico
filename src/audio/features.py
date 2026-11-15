"""Extração de biomarcadores vocais (apenas NumPy, sem dependências nativas extras).

Medidas clássicas de análise acústica da voz:
* F0 (frequência fundamental) por autocorrelação normalizada, quadro a quadro;
* jitter local (% de variação ciclo a ciclo do período) e shimmer local (% da amplitude);
* HNR (relação harmônico-ruído, dB) a partir do pico de autocorrelação;
* proporção de pausas, taxa de segmentos de fala, tremor (modulação lenta de F0)
  e eventos explosivos tipo tosse.
"""
from __future__ import annotations

import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import config

QUADRO_S = 0.040
PASSO_S = 0.010


@dataclass
class FeaturesVoz:
    duracao_s: float
    fracao_vozeada: float       # quadros com voz / quadros totais
    fracao_pausa: float         # silêncio entre o início e o fim da fala
    n_pausas_longas: int        # pausas > 0,7 s
    f0_media: float
    f0_dp: float
    f0_cv: float                # dp/média — entonação (baixo = monótona)
    jitter_pct: float
    shimmer_pct: float
    hnr_db: float
    tremor_indice: float        # fração da variância de F0 na banda 3–12 Hz
    tremor_hz: float
    tosses_por_min: float
    energia_db: float

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def ler_wav(caminho: str | Path) -> tuple[np.ndarray, int]:
    """Lê WAV PCM (8/16/24/32 bits) -> (mono float32 em [-1, 1], taxa)."""
    with wave.open(str(caminho), "rb") as w:
        fs, n_canais, largura = w.getframerate(), w.getnchannels(), w.getsampwidth()
        bruto = w.readframes(w.getnframes())
    if largura == 1:
        x = (np.frombuffer(bruto, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif largura == 2:
        x = np.frombuffer(bruto, dtype="<i2").astype(np.float32) / 32768
    elif largura == 3:
        b = np.frombuffer(bruto, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 8388608
    elif largura == 4:
        x = np.frombuffer(bruto, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"largura de amostra não suportada: {largura}")
    if n_canais > 1:
        x = x.reshape(-1, n_canais).mean(axis=1)
    return x, fs


def _quadros(x: np.ndarray, fs: int) -> np.ndarray:
    n, passo = int(QUADRO_S * fs), int(PASSO_S * fs)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x, n)[::passo]


def _f0_por_quadro(q: np.ndarray, fs: int, f0_min: float, f0_max: float) -> tuple[np.ndarray, np.ndarray]:
    """(f0 Hz, força de periodicidade 0–1) por quadro, via autocorrelação normalizada (FFT)."""
    n = q.shape[1]
    q = q - q.mean(axis=1, keepdims=True)
    jan = np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    esp = np.fft.rfft(q * jan, nfft, axis=1)
    ac = np.fft.irfft(np.abs(esp) ** 2, axis=1)[:, :n]
    ac_jan = np.fft.irfft(np.abs(np.fft.rfft(jan, nfft)) ** 2)[:n]
    ac = ac / (ac[:, :1] + 1e-12) / (ac_jan / ac_jan[0])  # correção de Boersma para a janela
    lag_min, lag_max = int(fs / f0_max), min(int(fs / f0_min), n - 2)
    trecho = ac[:, lag_min:lag_max]
    # primeiro pico local com pelo menos 90% do máximo: evita erro de oitava (escolher 2T, 3T...)
    maximo = trecho.max(axis=1, keepdims=True)
    pico_local = np.zeros_like(trecho, dtype=bool)
    pico_local[:, 1:-1] = (trecho[:, 1:-1] >= trecho[:, :-2]) & (trecho[:, 1:-1] >= trecho[:, 2:])
    candidato = pico_local & (trecho >= 0.9 * maximo)
    k = np.where(candidato.any(axis=1), candidato.argmax(axis=1), trecho.argmax(axis=1))
    forca = np.clip(trecho[np.arange(len(k)), k], 0, 1)
    lag = k + lag_min
    # interpolação parabólica para precisão sub-amostra
    a, b, c = ac[np.arange(len(k)), lag - 1], ac[np.arange(len(k)), lag], ac[np.arange(len(k)), lag + 1]
    desloc = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    return fs / (lag + np.clip(desloc, -1, 1)), forca


def _ciclos(x: np.ndarray, fs: int, f0: float) -> tuple[np.ndarray, np.ndarray]:
    """Períodos (s) e amplitudes pico a pico ciclo a ciclo de um trecho vozeado."""
    periodo = int(round(fs / f0))
    # suaviza para o pico de cada ciclo glotal ficar único
    nucleo = np.hanning(max(3, periodo // 4) | 1)
    s = np.convolve(x, nucleo / nucleo.sum(), mode="same")
    picos = []
    i = int(np.argmax(s[: periodo * 2]))
    while i < len(s) - periodo // 2:
        picos.append(i)
        a, b = i + int(0.7 * periodo), min(i + int(1.4 * periodo), len(s))
        if a >= b:
            break
        i = a + int(np.argmax(s[a:b]))
    picos = np.array(picos)
    if len(picos) < 4:
        return np.array([]), np.array([])
    # refino sub-amostra do instante de pico (parábola)
    p = picos[(picos > 0) & (picos < len(s) - 1)]
    y0, y1, y2 = s[p - 1], s[p], s[p + 1]
    t = (p + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)) / fs
    amp = np.array([np.ptp(x[max(0, a - periodo // 2): a + periodo // 2 + 1]) for a in p])
    return np.diff(t), amp[:-1]


def extrair_features(x: np.ndarray, fs: int) -> FeaturesVoz:
    cfg = config.AUDIO
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean()
    pico = float(np.abs(x).max()) or 1.0
    q = _quadros(x / pico, fs)
    energia = np.sqrt((q ** 2).mean(axis=1))
    limiar = max(0.1 * np.percentile(energia, 95), 1e-4)
    ativo = energia > limiar
    f0, forca = _f0_por_quadro(q, fs, cfg["f0_min"], cfg["f0_max"])
    vozeado = ativo & (forca > 0.40)
    cruzamentos = (np.diff(np.signbit(q), axis=1) != 0).mean(axis=1)  # taxa de cruzamentos por zero
    passo = int(PASSO_S * fs)

    # --- pausas (dentro do intervalo de fala) ---------------------------------------
    idx = np.flatnonzero(ativo)
    fracao_pausa, pausas_longas = 0.0, 0
    if len(idx) > 1:
        trecho = ativo[idx[0]: idx[-1] + 1]
        fracao_pausa = float(1 - trecho.mean())
        borda = np.diff(np.concatenate(([1], trecho.astype(np.int8), [1])))
        duracoes = (np.flatnonzero(borda == 1) - np.flatnonzero(borda == -1)) * PASSO_S
        pausas_longas = int((duracoes > 0.7).sum())

    # --- F0, tremor ------------------------------------------------------------------
    f0v = f0[vozeado]
    if len(f0v) >= 5:
        med = np.median(f0v)
        f0v_ok = f0v[(f0v > med / 1.6) & (f0v < med * 1.6)]  # remove saltos de oitava
        f0_media, f0_dp = float(f0v_ok.mean()), float(f0v_ok.std())
    else:
        f0_media = f0_dp = 0.0

    # --- jitter / shimmer / HNR / tremor nos trechos vozeados contínuos -------------
    jit, shim, pesos, tremores = [], [], [], []
    borda = np.diff(np.concatenate(([0], vozeado.astype(np.int8), [0])))
    for ini, fim in zip(np.flatnonzero(borda == 1), np.flatnonzero(borda == -1)):
        if (fim - ini) * PASSO_S < 0.25:
            continue
        seg = x[ini * passo: fim * passo + int(QUADRO_S * fs)]
        T, A = _ciclos(seg, fs, float(np.median(f0[ini:fim])))
        if len(T) >= 8:
            ok = np.abs(T - np.median(T)) < 0.25 * np.median(T)  # descarta erros de marcação
            T, A = T[ok], A[ok]
        if len(T) >= 8:
            # remove a tendência lenta (entonação/tremor) com média móvel de 5 ciclos, para que
            # só a perturbação ciclo a ciclo conte como jitter/shimmer
            nucleo = np.ones(5) / 5
            Td = (T - np.convolve(T, nucleo, mode="same"))[2:-2]
            Ad = (A - np.convolve(A, nucleo, mode="same"))[2:-2]
            jit.append(np.abs(np.diff(Td)).mean() / T.mean() * 100)
            shim.append(np.abs(np.diff(Ad)).mean() / (A.mean() + 1e-12) * 100)
            pesos.append(len(T))
        contorno = f0[ini + 3: fim - 3]  # descarta as bordas (ataque/final instáveis)
        if len(contorno) * PASSO_S >= 1.0:  # tremor exige pelo menos 1 s de vogal sustentada
            contorno = np.clip(contorno, 0.75 * np.median(contorno), 1.33 * np.median(contorno))
            c = contorno - np.polyval(np.polyfit(np.arange(len(contorno)), contorno, 1), np.arange(len(contorno)))
            esp = np.abs(np.fft.rfft(c * np.hanning(len(c)))) ** 2
            fr = np.fft.rfftfreq(len(c), PASSO_S)
            banda = (fr >= 3) & (fr <= 12)
            if esp[1:].sum() > 0 and banda.any():
                profundidade = c.std() / (contorno.mean() + 1e-9)
                tremores.append((esp[banda].sum() / esp[1:].sum() * min(1.0, profundidade / 0.02),
                                 fr[banda][esp[banda].argmax()], len(contorno)))
    jitter = float(np.average(jit, weights=pesos)) if jit else 0.0
    shimmer = float(np.average(shim, weights=pesos)) if shim else 0.0
    r = np.clip(forca[vozeado], 1e-3, 0.999)
    hnr = float(np.median(10 * np.log10(r / (1 - r)))) if len(r) else 0.0
    if tremores:
        w = np.array([t[2] for t in tremores], dtype=float)
        tremor_indice = float(np.average([t[0] for t in tremores], weights=w))
        tremor_hz = float(tremores[int(np.argmax([t[0] * t[2] for t in tremores]))][1])
    else:
        tremor_indice = tremor_hz = 0.0

    # --- eventos tipo tosse: explosões curtas, intensas e não periódicas ------------
    explosivo = ativo & (energia > 0.5 * np.percentile(energia, 99)) & (forca < 0.35) & (cruzamentos > 0.25)
    borda = np.diff(np.concatenate(([0], explosivo.astype(np.int8), [0])))
    dur = (np.flatnonzero(borda == -1) - np.flatnonzero(borda == 1)) * PASSO_S
    n_tosses = int(((dur >= 0.05) & (dur <= 0.6)).sum())
    duracao = len(x) / fs

    return FeaturesVoz(
        duracao_s=duracao,
        fracao_vozeada=float(vozeado.mean()),
        fracao_pausa=fracao_pausa,
        n_pausas_longas=pausas_longas,
        f0_media=f0_media, f0_dp=f0_dp, f0_cv=f0_dp / f0_media if f0_media else 0.0,
        jitter_pct=jitter, shimmer_pct=shimmer, hnr_db=hnr,
        tremor_indice=tremor_indice, tremor_hz=tremor_hz,
        tosses_por_min=n_tosses / max(duracao / 60, 1e-6),
        energia_db=float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)),
    )


def extrair_de_arquivo(caminho: str | Path) -> FeaturesVoz:
    x, fs = ler_wav(caminho)
    return extrair_features(x, fs)
