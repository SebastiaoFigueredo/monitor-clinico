"""Padrões anômalos de movimentação do paciente durante a internação.

Entrada: DataFrame em intervalos regulares (ex.: 15 min), normalmente agregado a partir
do módulo de vídeo ou de sensores de leito, com as colunas
`timestamp, atividade (0–1), fora_do_leito_min, mudancas_posicao`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import Anomalia, Severidade
from .detectores import episodios

PARAMS = {
    "imobilidade_h": 2.0,          # sem mudança de posição (risco de lesão por pressão)
    "imobilidade_grave_h": 4.0,
    "noite": (0, 5),               # horas [início, fim)
    "noturno_fora_leito_min": 20,  # minutos fora do leito na madrugada
    "queda_atividade_frac": 0.4,   # atividade diurna < 40% da mediana dos dias anteriores
    "agitacao_z": 4.0,
    "agitacao_intervalos": 3,
    "dias_base_min": 3,
}


def detectar_anomalias_movimentacao(df: pd.DataFrame, paciente_id: str) -> list[Anomalia]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]
    passo_h = float(ts.diff().median().total_seconds()) / 3600 if len(df) > 1 else 0.25
    anomalias: list[Anomalia] = []

    def novo(quando, tipo, sev, desc, detector, **ev):
        anomalias.append(Anomalia(paciente_id, "movimentacao", tipo, sev, desc,
                                  pd.Timestamp(quando).to_pydatetime(), ev, detector))

    # 1) imobilidade prolongada ------------------------------------------------------
    parado = (df["mudancas_posicao"].to_numpy() == 0) & (df["fora_do_leito_min"].to_numpy() == 0)
    n_min = int(np.ceil(PARAMS["imobilidade_h"] / passo_h))
    for ini, fim in episodios(parado, minimo=n_min):
        horas = (fim - ini) * passo_h
        grave = horas >= PARAMS["imobilidade_grave_h"]
        # o alerta dispara no momento em que o limite é atingido, não no fim do episódio
        novo(ts.iloc[ini + n_min - 1], "imobilidade_prolongada", Severidade.ALTA if grave else Severidade.MEDIA,
             f"Paciente sem mudança de posição por {horas:.1f} h (desde {ts.iloc[ini]:%d/%m %H:%M}) — "
             "risco de lesão por pressão.", "regra de duração", horas=round(horas, 1), inicio=str(ts.iloc[ini]))

    # 2) deambulação noturna ---------------------------------------------------------
    h0, h1 = PARAMS["noite"]
    noite = df[(ts.dt.hour >= h0) & (ts.dt.hour < h1)]
    por_noite = noite.groupby(noite["timestamp"].dt.normalize())["fora_do_leito_min"].sum()
    for i, (dia, minutos) in enumerate(por_noite.items()):
        base = por_noite.iloc[:i]
        habitual = float(base.median()) if len(base) else 0.0
        if minutos >= PARAMS["noturno_fora_leito_min"] and minutos > 3 * max(habitual, 5):
            trecho = noite[(noite["timestamp"].dt.normalize() == dia) & (noite["fora_do_leito_min"] > 0)]
            novo(trecho["timestamp"].iloc[0], "deambulacao_noturna", Severidade.MEDIA,
                 f"{minutos:.0f} min fora do leito na madrugada de {dia:%d/%m} (habitual: {habitual:.0f} min) — "
                 "avaliar delirium/risco de queda.", "comparação com a linha de base do paciente",
                 minutos=float(minutos), habitual=habitual)

    # 3) queda da atividade diurna ---------------------------------------------------
    dia_claro = df[(ts.dt.hour >= 7) & (ts.dt.hour < 21)]
    diaria = dia_claro.groupby(dia_claro["timestamp"].dt.normalize())["atividade"].agg(["mean", "count"])
    esperado = int(14 / passo_h)
    diaria = diaria[diaria["count"] >= 0.8 * esperado]["mean"]  # só dias completos
    for i in range(PARAMS["dias_base_min"], len(diaria)):
        base = float(diaria.iloc[:i].median())
        hoje = float(diaria.iloc[i])
        if base > 0 and hoje < PARAMS["queda_atividade_frac"] * base:
            novo(diaria.index[i] + pd.Timedelta(hours=21), "queda_de_atividade", Severidade.MEDIA,
                 f"Atividade diurna em {diaria.index[i]:%d/%m} caiu para {100 * hoje / base:.0f}% do padrão dos dias "
                 "anteriores — possível piora clínica, dor ou sedação excessiva.",
                 "comparação com a linha de base do paciente", atividade=round(hoje, 3), base=round(base, 3))

    # 4) agitação: atividade muito acima do padrão do paciente naquela hora do dia ---
    atividade = df["atividade"].to_numpy(dtype=float)
    hora = ts.dt.hour.to_numpy()
    z = np.zeros(len(df))
    for h in range(24):
        idx = np.flatnonzero(hora == h)
        if len(idx) < 8:
            continue
        v = atividade[idx]
        med = np.median(v)
        escala = max(np.median(np.abs(v - med)) * 1.4826, 0.05)
        z[idx] = (v - med) / escala
    for ini, fim in episodios(z > PARAMS["agitacao_z"], minimo=PARAMS["agitacao_intervalos"]):
        novo(ts.iloc[ini + PARAMS["agitacao_intervalos"] - 1], "agitacao", Severidade.MEDIA,
             f"Agitação motora sustentada por {(fim - ini) * passo_h * 60:.0f} min a partir de "
             f"{ts.iloc[ini]:%d/%m %H:%M} (atividade {atividade[ini:fim].mean():.2f}, muito acima do padrão do horário).",
             "z-score robusto por hora do dia", z_max=round(float(z[ini:fim].max()), 1))

    return sorted(anomalias, key=lambda a: a.timestamp)
