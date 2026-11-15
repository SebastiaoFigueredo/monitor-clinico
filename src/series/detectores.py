"""Técnicas genéricas de detecção de anomalias em séries temporais.

* z-score robusto móvel (mediana/MAD) — picos e quedas abruptas;
* CUSUM tabular — desvios pequenos porém sustentados (tendência/deriva);
* Isolation Forest — combinações multivariadas atípicas;
* utilitários de episódios (trechos contíguos de uma máscara).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def episodios(mascara: np.ndarray, minimo: int = 1) -> list[tuple[int, int]]:
    """Trechos contíguos True como pares (início, fim_exclusivo) com tamanho >= minimo."""
    m = np.asarray(mascara, dtype=bool)
    if not m.any():
        return []
    borda = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    inicios, fins = np.flatnonzero(borda == 1), np.flatnonzero(borda == -1)
    return [(int(i), int(f)) for i, f in zip(inicios, fins) if f - i >= minimo]


def z_robusto_movel(x: np.ndarray, janela: int, escala_min: float = 1e-6) -> np.ndarray:
    """z de cada ponto em relação à mediana/MAD das `janela` amostras ANTERIORES.

    Usar apenas o passado evita que a própria anomalia contamine a referência.
    Pontos sem histórico suficiente recebem z = 0. NaN é ignorado.
    """
    x = np.asarray(x, dtype=float)
    z = np.zeros_like(x)
    if len(x) <= janela:
        return z
    jan = sliding_window_view(x, janela)[:-1]          # jan[i] = x[i : i+janela] -> referência de x[i+janela]
    med = np.nanmedian(jan, axis=1)
    mad = np.nanmedian(np.abs(jan - med[:, None]), axis=1) * 1.4826
    escala = np.maximum(mad, escala_min)
    z[janela:] = (x[janela:] - med) / escala
    return np.nan_to_num(z)


def residuo_sazonal(x: np.ndarray, periodo: int, meia_janela: int = 6) -> np.ndarray:
    """x menos o valor esperado para aquele horário: mediana das amostras vizinhas (±meia_janela)
    no mesmo ponto do ciclo nos períodos ANTERIORES (ex.: mesmo horário dos dias anteriores).

    Remove o ritmo circadiano, que de outro modo seria confundido com tendência.
    Devolve NaN onde ainda não há um período completo de histórico.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    larg = 2 * meia_janela + 1
    suave = np.full(n, np.nan)  # mediana centrada, tolerante a NaN
    if n >= larg:
        suave[meia_janela: n - meia_janela] = np.nanmedian(sliding_window_view(x, larg), axis=1)
    lags = []
    for k in range(1, n // periodo + 1):
        desloc = np.full(n, np.nan)
        desloc[k * periodo:] = suave[: n - k * periodo]
        lags.append(desloc)
    if not lags:
        return np.full(n, np.nan)
    pilha = np.vstack(lags)
    esperado = np.full(n, np.nan)
    tem = ~np.isnan(pilha).all(axis=0)
    esperado[tem] = np.nanmedian(pilha[:, tem], axis=0)
    return x - esperado


def cusum(x: np.ndarray, n_base: int, k: float = 0.5, h: float = 6.0,
          escala_min: float = 1e-6, mu: float | None = None, sigma: float | None = None,
          ) -> list[tuple[int, int, int]]:
    """CUSUM tabular bilateral sobre a série padronizada pela linha de base inicial
    (ou por `mu`/`sigma` informados, caso em que a análise começa em `n_base`).

    Devolve [(indice_alarme, indice_inicio_estimado, direcao)], direcao = +1 (alta) ou -1 (queda).
    Após cada alarme a soma é zerada e a linha de base passa a ser o nível atual,
    de modo que apenas NOVAS mudanças de nível geram novos alarmes.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < n_base + 5:
        return []
    if mu is None or sigma is None:
        base = x[:n_base]
        mu = float(np.nanmedian(base))
        sigma = max(float(np.nanmedian(np.abs(base - mu)) * 1.4826), escala_min)
    alarmes: list[tuple[int, int, int]] = []
    s_pos = s_neg = 0.0
    ini_pos = ini_neg = n_base
    for i in range(n_base, len(x)):
        if np.isnan(x[i]):
            continue
        zi = (x[i] - mu) / sigma
        if s_pos == 0.0:
            ini_pos = i
        if s_neg == 0.0:
            ini_neg = i
        s_pos = max(0.0, s_pos + zi - k)
        s_neg = max(0.0, s_neg - zi - k)
        if s_pos > h or s_neg > h:
            direcao = 1 if s_pos > h else -1
            alarmes.append((i, ini_pos if direcao == 1 else ini_neg, direcao))
            recente = x[max(n_base, i - 12): i + 1]
            mu = float(np.nanmedian(recente))
            s_pos = s_neg = 0.0
    return alarmes


def isolation_forest(X: np.ndarray, n_ref: int, semente: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """(escore, é_anomalia) por linha; escore maior = mais anômalo.

    O modelo aprende o "normal" do paciente nas primeiras `n_ref` linhas (período de referência)
    e marca como anômalas as linhas mais isoladas do que QUALQUER ponto da referência — em vez de
    marcar uma fração fixa (`contamination`), que geraria alertas mesmo em pacientes estáveis.

    Usa scikit-learn quando disponível; caso contrário cai para distância de Mahalanobis
    robusta (mesma interface), para o sistema não parar por falta da biblioteca.
    """
    X = np.asarray(X, dtype=float)
    ref = X[:n_ref]
    med = np.nanmedian(ref, axis=0)
    X = np.where(np.isnan(X), med, X)
    ref = X[:n_ref]
    try:
        from sklearn.ensemble import IsolationForest

        modelo = IsolationForest(n_estimators=200, random_state=semente).fit(ref)
        escore = -modelo.score_samples(X)
    except ImportError:
        escala = np.maximum(np.nanmedian(np.abs(ref - med), axis=0) * 1.4826, 1e-6)
        Z = (X - med) / escala
        cov = np.cov(Z[:n_ref], rowvar=False) + 1e-6 * np.eye(Z.shape[1])
        escore = np.sqrt(np.einsum("ij,jk,ik->i", Z, np.linalg.inv(cov), Z))
    flag = escore > escore[:n_ref].max()
    flag[:n_ref] = False
    return escore, flag
