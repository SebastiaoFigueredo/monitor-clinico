"""Detecção de anomalias em séries de sinais vitais (FC, PA, SpO2, FR, temperatura).

Entrada: DataFrame com coluna `timestamp` e quaisquer das colunas
`fc, pas, pad, spo2, fr, temp` (amostragem regular, ex.: a cada 5 min).

Camadas, da mais específica para a mais sensível:
1. limites clínicos absolutos (alerta/crítico) com persistência;
2. escore de alerta precoce (NEWS2 simplificado, sem consciência/O2 suplementar);
3. variação abrupta em relação ao próprio paciente (z robusto móvel);
4. tendência sustentada dentro da faixa "normal" (CUSUM);
5. padrão multivariado atípico (Isolation Forest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..schemas import Anomalia, Severidade
from .detectores import cusum, episodios, isolation_forest, residuo_sazonal, z_robusto_movel

VITAIS = list(config.LIMITES_VITAIS)
# Escala mínima do MAD: evita z explosivo em sinais quantizados (ex.: SpO2 inteira e estável)
ESCALA_MIN = {"fc": 3.0, "pas": 5.0, "pad": 4.0, "spo2": 1.0, "fr": 1.5, "temp": 0.2}
# Desvio mínimo (vs. o esperado para o horário) para uma tendência ter relevância clínica
RELEVANCIA = {"fc": 12.0, "pas": 15.0, "pad": 10.0, "spo2": 3.0, "fr": 4.0, "temp": 0.6}


def _fmt(vital: str, valor: float) -> str:
    casas = 1 if vital == "temp" else 0
    return f"{valor:.{casas}f} {config.UNIDADES[vital]}"


def news2(df: pd.DataFrame) -> pd.Series:
    """Escore NEWS2 simplificado por linha (apenas parâmetros fisiológicos disponíveis)."""
    total = pd.Series(0, index=df.index, dtype=int)
    faixas = {
        "fr": [(-np.inf, 8, 3), (8, 11, 1), (11, 20, 0), (20, 24, 2), (24, np.inf, 3)],
        "spo2": [(-np.inf, 91, 3), (91, 93, 2), (93, 95, 1), (95, np.inf, 0)],
        "pas": [(-np.inf, 90, 3), (90, 100, 2), (100, 110, 1), (110, 219, 0), (219, np.inf, 3)],
        "fc": [(-np.inf, 40, 3), (40, 50, 1), (50, 90, 0), (90, 110, 1), (110, 130, 2), (130, np.inf, 3)],
        "temp": [(-np.inf, 35.0, 3), (35.0, 36.0, 1), (36.0, 38.0, 0), (38.0, 39.0, 1), (39.0, np.inf, 2)],
    }
    for vital, regras in faixas.items():
        if vital not in df:
            continue
        v = df[vital]
        for baixo, alto, pontos in regras:  # intervalos (baixo, alto]
            if pontos:
                total += ((v > baixo) & (v <= alto)).astype(int) * pontos
    return total


def detectar_anomalias_vitais(df: pd.DataFrame, paciente_id: str) -> list[Anomalia]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])
    cfg = config.SERIES
    presentes = [v for v in VITAIS if v in df.columns]
    passo_s = float(ts.diff().median().total_seconds()) if len(df) > 1 else 300.0
    por_dia = max(1, int(round(86400 / passo_s)))
    anomalias: list[Anomalia] = []
    fora_limite = np.zeros(len(df), dtype=bool)

    def novo(i: int, tipo: str, sev: Severidade, desc: str, detector: str, **ev) -> None:
        anomalias.append(Anomalia(paciente_id, "sinais_vitais", tipo, sev, desc,
                                  ts.iloc[i].to_pydatetime(), ev, detector))

    # 1) limites clínicos -------------------------------------------------------------
    for vital in presentes:
        x = df[vital].to_numpy(dtype=float)
        crit_b, al_b, al_a, crit_a = config.LIMITES_VITAIS[vital]
        nome = config.NOMES_VITAIS[vital]
        for lado, m_alerta, m_crit, extremo in (
            ("baixa", x < al_b, x < crit_b, np.nanmin),
            ("alta", x > al_a, x > crit_a, np.nanmax),
        ):
            fora_limite |= np.nan_to_num(m_alerta, nan=False)
            for ini, fim in episodios(m_alerta):
                critico = bool(m_crit[ini:fim].any())
                # valor crítico alerta de imediato; faixa de alerta exige persistência (anti-artefato)
                if not critico and fim - ini < cfg["persistencia_min"]:
                    continue
                pior = float(extremo(x[ini:fim]))
                dur = (ts.iloc[fim - 1] - ts.iloc[ini]).total_seconds() / 60
                i_alerta = ini + int(np.argmax(m_crit[ini:fim])) if critico else ini + cfg["persistencia_min"] - 1
                novo(
                    i_alerta, f"{vital}_{lado}", Severidade.CRITICA if critico else Severidade.MEDIA,
                    f"{nome[0].upper() + nome[1:]} {lado}: {_fmt(vital, pior)} "
                    f"(faixa esperada {al_b}–{al_a} {config.UNIDADES[vital]}), episódio de {dur:.0f} min.",
                    "limite clínico", valor_extremo=pior, duracao_min=round(dur),
                    inicio=str(ts.iloc[ini]), limite_alerta=[al_b, al_a], limite_critico=[crit_b, crit_a],
                )

    # 2) escore de alerta precoce ------------------------------------------------------
    escore = news2(df).to_numpy()
    for limiar, sev in ((7, Severidade.CRITICA), (5, Severidade.ALTA)):
        for ini, fim in episodios(escore >= limiar, minimo=cfg["persistencia_min"]):
            if sev is Severidade.ALTA and escore[ini:fim].max() >= 7:
                continue  # já coberto pelo episódio crítico
            pico = int(escore[ini:fim].max())
            novo(ini + cfg["persistencia_min"] - 1, "escore_alerta_precoce", sev,
                 f"Escore de alerta precoce (NEWS2 simplificado) = {pico}: deterioração clínica "
                 f"{'grave' if pico >= 7 else 'provável'}.",
                 "NEWS2 simplificado", escore_max=pico, inicio=str(ts.iloc[ini]))

    # 3) variação abrupta (z robusto) e 4) tendência (CUSUM) --------------------------
    for vital in presentes:
        x = df[vital].to_numpy(dtype=float)
        nome = config.NOMES_VITAIS[vital]
        z = z_robusto_movel(x, cfg["janela_z"], ESCALA_MIN[vital])
        for ini, fim in episodios((np.abs(z) > cfg["z_limite"]) & ~fora_limite):
            i = ini + int(np.argmax(np.abs(z[ini:fim])))
            ref = float(np.nanmedian(x[max(0, ini - cfg["janela_z"]): ini]))
            novo(ini, f"{vital}_variacao_abrupta", Severidade.MEDIA,
                 f"Variação abrupta de {nome}: {_fmt(vital, x[i])} contra mediana recente de "
                 f"{_fmt(vital, ref)} (z robusto = {z[i]:+.1f}), ainda dentro dos limites absolutos.",
                 "z-score robusto móvel (mediana/MAD)", valor=float(x[i]), referencia=ref, z=round(float(z[i]), 2))

        # Tendência: CUSUM sobre o resíduo em relação ao esperado para o horário (sem ritmo
        # circadiano). Com menos de ~1,5 dia de histórico, usa a linha de base inicial.
        if len(x) >= 1.5 * por_dia:
            resid = residuo_sazonal(x, por_dia)
            i0 = int(np.argmax(~np.isnan(resid)))
            sigma = max(float(np.nanmedian(np.abs(resid - np.nanmedian(resid))) * 1.4826), ESCALA_MIN[vital])
            alarmes = cusum(resid, i0, cfg["cusum_k"], cfg["cusum_h"], mu=0.0, sigma=sigma)
        else:
            resid = x - np.nanmedian(x[: cfg["janela_z"]])
            alarmes = cusum(x, cfg["janela_z"], cfg["cusum_k"], cfg["cusum_h"], ESCALA_MIN[vital])
        ja = set()
        for i_alarme, i_inicio, direcao in alarmes:
            desvio = float(np.nanmedian(resid[max(0, i_alarme - 5): i_alarme + 1]))
            if fora_limite[i_alarme] or direcao in ja or abs(desvio) < RELEVANCIA[vital] or np.sign(desvio) != direcao:
                continue  # já coberto por limite, repetido, ou sem relevância clínica
            ja.add(direcao)
            atual = float(np.nanmedian(x[max(0, i_alarme - 5): i_alarme + 1]))
            sentido = "alta" if direcao > 0 else "queda"
            novo(i_alarme, f"{vital}_tendencia_{sentido}", Severidade.MEDIA,
                 f"Tendência sustentada de {sentido} em {nome}: ~{_fmt(vital, atual)}, contra ~{_fmt(vital, atual - desvio)} "
                 f"esperados para o horário (desvio de {desvio:+.1f}), ainda dentro dos limites absolutos.",
                 "CUSUM sobre resíduo sazonal", esperado=round(atual - desvio, 1), atual=atual,
                 desvio=round(desvio, 1), inicio=str(ts.iloc[i_inicio]))

    # 5) padrão multivariado ----------------------------------------------------------
    if len(df) >= 120 and len(presentes) >= 3:
        X = df[presentes].to_numpy(dtype=float)
        dX = np.vstack([np.zeros((1, X.shape[1])), np.diff(X, axis=0)])
        n_ref = max(60, int(len(df) * cfg["iforest_frac_referencia"]))
        escore_if, flag = isolation_forest(np.hstack([X, dX]), n_ref)
        ja_explicado = fora_limite.copy()
        for a in anomalias:  # pontos já cobertos pelas camadas univariadas
            idx = int(np.searchsorted(ts.to_numpy(), np.datetime64(a.timestamp)))
            ja_explicado[max(0, idx - 6): idx + 7] = True
        for ini, fim in episodios(flag & ~ja_explicado, minimo=cfg["persistencia_min"]):
            i = ini + int(np.argmax(escore_if[ini:fim]))
            valores = {v: float(df[v].iloc[i]) for v in presentes}
            novo(i, "padrao_multivariado", Severidade.BAIXA,
                 "Combinação atípica de sinais vitais para este paciente (cada sinal isolado está "
                 "na faixa): " + ", ".join(f"{v.upper()} {_fmt(v, val)}" for v, val in valores.items()) + ".",
                 "Isolation Forest", escore=round(float(escore_if[i]), 3), valores=valores)

    return sorted(anomalias, key=lambda a: a.timestamp)
