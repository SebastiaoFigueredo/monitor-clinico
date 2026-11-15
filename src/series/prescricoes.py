"""Detecção de alterações inesperadas na evolução das prescrições.

Entrada: DataFrame de eventos de prescrição com as colunas
`timestamp, medicamento, acao (iniciar|alterar|suspender), dose_mg, intervalo_h, via, prescritor`.

A base de medicamentos abaixo é ILUSTRATIVA (protótipo acadêmico): em produção deve
vir do sistema de prescrição/farmácia do hospital.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import Anomalia, Severidade
from .detectores import z_robusto_movel

# classe terapêutica, dose máxima diária (mg ou UI), alto risco (lista ISMP), uso que não
# deve ser interrompido abruptamente, e duração mínima de curso (dias) para antimicrobianos.
MEDICAMENTOS: dict[str, dict] = {
    "heparina": dict(classe="anticoagulante", max_dia=40000, alto_risco=True, continuo=True),
    "enoxaparina": dict(classe="anticoagulante", max_dia=200, alto_risco=True, continuo=True),
    "varfarina": dict(classe="anticoagulante", max_dia=15, alto_risco=True, continuo=True),
    "insulina regular": dict(classe="insulina", max_dia=100, alto_risco=True, continuo=True),
    "morfina": dict(classe="opioide", max_dia=120, alto_risco=True),
    "tramadol": dict(classe="opioide", max_dia=400),
    "cloreto de potassio": dict(classe="eletrolito", max_dia=6000, alto_risco=True),
    "digoxina": dict(classe="digitalico", max_dia=0.5, alto_risco=True, continuo=True),
    "amiodarona": dict(classe="antiarritmico", max_dia=1200, alto_risco=True, continuo=True),
    "metoprolol": dict(classe="betabloqueador", max_dia=400, continuo=True),
    "fenitoina": dict(classe="anticonvulsivante", max_dia=600, continuo=True),
    "ceftriaxona": dict(classe="antibiotico", max_dia=4000, curso_min_dias=5),
    "meropenem": dict(classe="antibiotico", max_dia=6000, curso_min_dias=5),
    "vancomicina": dict(classe="antibiotico", max_dia=4000, alto_risco=True, curso_min_dias=5),
    "ciprofloxacino": dict(classe="antibiotico", max_dia=1500, curso_min_dias=5),
    "dipirona": dict(classe="analgesico", max_dia=4000),
    "paracetamol": dict(classe="analgesico", max_dia=4000),
    "cetoprofeno": dict(classe="aine", max_dia=300),
    "ibuprofeno": dict(classe="aine", max_dia=3200),
    "omeprazol": dict(classe="ibp", max_dia=80),
    "furosemida": dict(classe="diuretico", max_dia=600),
    "espironolactona": dict(classe="diuretico poupador de potassio", max_dia=400),
    "enalapril": dict(classe="ieca", max_dia=40, continuo=True),
    "haloperidol": dict(classe="antipsicotico", max_dia=20),
    "ondansetrona": dict(classe="antiemetico", max_dia=24),
}

# pares (classe ou nome) com interação relevante -> risco
INTERACOES: list[tuple[str, str, str]] = [
    ("anticoagulante", "aine", "aumento do risco de sangramento"),
    ("anticoagulante", "anticoagulante", "anticoagulação duplicada — risco de sangramento grave"),
    ("opioide", "opioide", "depressão respiratória por opioides associados"),
    ("amiodarona", "digoxina", "elevação do nível sérico de digoxina (toxicidade)"),
    ("amiodarona", "varfarina", "potencialização da varfarina (INR elevado)"),
    ("amiodarona", "haloperidol", "prolongamento do intervalo QT"),
    ("ciprofloxacino", "amiodarona", "prolongamento do intervalo QT"),
    ("ondansetrona", "haloperidol", "prolongamento do intervalo QT"),
    ("ieca", "diuretico poupador de potassio", "hipercalemia"),
    ("cloreto de potassio", "diuretico poupador de potassio", "hipercalemia"),
    ("ieca", "cloreto de potassio", "hipercalemia"),
]
# duplicidade na mesma classe é esperada/aceitável para estas
CLASSES_SEM_DUPLICIDADE = {"antibiotico", "analgesico", "eletrolito", "diuretico"}


def _norm(nome: str) -> str:
    tabela = str.maketrans("áàâãéêíóôõúç", "aaaaeeiooouc")
    return str(nome).strip().lower().translate(tabela)


def _casa(alvo: str, nome: str, classe: str) -> bool:
    return alvo in (nome, classe)


def detectar_anomalias_prescricao(df: pd.DataFrame, paciente_id: str) -> list[Anomalia]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["medicamento"] = df["medicamento"].map(_norm)
    df = df.sort_values("timestamp").reset_index(drop=True)
    anomalias: list[Anomalia] = []
    ativos: dict[str, dict] = {}  # medicamento -> {dose_dia, inicio, ...}

    def novo(linha, tipo: str, sev: Severidade, desc: str, detector: str = "regra de segurança", **ev) -> None:
        ev.setdefault("prescritor", linha.get("prescritor", ""))
        anomalias.append(Anomalia(paciente_id, "prescricao", tipo, sev, desc,
                                  linha["timestamp"].to_pydatetime(), ev, detector))

    for _, linha in df.iterrows():
        med, acao = linha["medicamento"], str(linha["acao"]).lower()
        info = MEDICAMENTOS.get(med, {})
        classe = info.get("classe", "desconhecida")
        alto_risco = bool(info.get("alto_risco"))

        if acao == "suspender":
            anterior = ativos.pop(med, None)
            if anterior is None:
                continue
            dias = (linha["timestamp"] - anterior["inicio"]).total_seconds() / 86400
            curso = info.get("curso_min_dias")
            if curso and dias < curso:
                novo(linha, "suspensao_inesperada", Severidade.MEDIA,
                     f"{med.capitalize()} suspenso após {dias:.1f} dias (curso mínimo usual: {curso} dias) — "
                     "risco de falha terapêutica/resistência.", dias_de_uso=round(dias, 1), curso_min_dias=curso)
            elif info.get("continuo"):
                novo(linha, "suspensao_inesperada", Severidade.ALTA if alto_risco else Severidade.MEDIA,
                     f"Suspensão de {med} ({classe}), medicamento de uso contínuo cuja interrupção abrupta "
                     "exige confirmação.", dias_de_uso=round(dias, 1))
            continue

        dose = float(linha["dose_mg"])
        dose_dia = dose * 24.0 / float(linha["intervalo_h"])
        anterior = ativos.get(med)

        # dose acima do máximo diário de referência
        maximo = info.get("max_dia")
        if maximo and dose_dia > maximo * 1.001:
            novo(linha, "dose_acima_maximo", Severidade.CRITICA if alto_risco else Severidade.ALTA,
                 f"{med.capitalize()}: dose diária prescrita de {dose_dia:g} excede o máximo de referência "
                 f"({maximo:g}/dia){' — medicamento de alto risco' if alto_risco else ''}.",
                 dose_dia=dose_dia, maximo_dia=maximo)

        if anterior is not None:  # alteração de um medicamento já em uso
            razao = dose_dia / anterior["dose_dia"] if anterior["dose_dia"] else np.inf
            if razao >= 2 or razao <= 0.5:
                if razao >= 5 or razao <= 0.2:
                    sev = Severidade.CRITICA if alto_risco else Severidade.ALTA
                    obs = " Variação ≥5x é compatível com erro de digitação/unidade."
                else:
                    sev, obs = (Severidade.ALTA if alto_risco else Severidade.MEDIA), ""
                novo(linha, "alteracao_dose", sev,
                     f"{med.capitalize()}: dose diária alterada de {anterior['dose_dia']:g} para {dose_dia:g} "
                     f"({razao:.1f}x) sem etapa intermediária.{obs}",
                     dose_anterior_dia=anterior["dose_dia"], dose_nova_dia=dose_dia, razao=round(razao, 2))
            if str(linha.get("via", "")) != anterior["via"]:
                novo(linha, "mudanca_via", Severidade.BAIXA,
                     f"{med.capitalize()}: via de administração alterada de {anterior['via']} para {linha.get('via')}.",
                     via_anterior=anterior["via"], via_nova=linha.get("via"))
            anterior.update(dose_dia=dose_dia, via=str(linha.get("via", "")))
            anterior["alteracoes"].append(linha["timestamp"])
            recentes = [t for t in anterior["alteracoes"] if (linha["timestamp"] - t).total_seconds() <= 86400]
            if len(recentes) == 3:
                novo(linha, "prescricao_instavel", Severidade.MEDIA,
                     f"{med.capitalize()} teve 3 alterações em menos de 24 h — revisar plano terapêutico.",
                     alteracoes_24h=len(recentes))
            continue

        # novo medicamento: interações e duplicidade contra os ativos
        for outro, dados in ativos.items():
            c_outro = dados["classe"]
            for a, b, risco in INTERACOES:
                if (_casa(a, med, classe) and _casa(b, outro, c_outro)) or (_casa(b, med, classe) and _casa(a, outro, c_outro)):
                    novo(linha, "interacao_medicamentosa", Severidade.ALTA,
                         f"Associação {med} + {outro}: {risco}.", medicamentos=[med, outro], risco=risco)
                    break
            else:
                if classe == c_outro and classe not in CLASSES_SEM_DUPLICIDADE and classe != "desconhecida":
                    novo(linha, "duplicidade_terapeutica", Severidade.MEDIA,
                         f"Duplicidade terapêutica: {med} e {outro} pertencem à classe {classe}.",
                         medicamentos=[med, outro], classe=classe)
        if not info:
            novo(linha, "medicamento_desconhecido", Severidade.BAIXA,
                 f"{med.capitalize()} não consta na base de referência — sem verificação automática de dose/interação.")
        ativos[med] = dict(dose_dia=dose_dia, inicio=linha["timestamp"], classe=classe,
                           via=str(linha.get("via", "")), alteracoes=[])

    # volume diário de mudanças como série temporal: dia muito acima do padrão do paciente
    por_dia = df.set_index("timestamp").resample("D").size()
    if len(por_dia) >= 6:
        z = z_robusto_movel(por_dia.to_numpy(dtype=float), janela=min(7, len(por_dia) - 1), escala_min=1.0)
        for i in np.flatnonzero((z > 3.5) & (por_dia.to_numpy() >= 5)):
            dia = por_dia.index[i]
            ultima = df[df["timestamp"].dt.normalize() == dia].iloc[-1]
            novo(ultima, "volume_alteracoes_atipico", Severidade.MEDIA,
                 f"{int(por_dia.iloc[i])} mudanças de prescrição em {dia:%d/%m}, muito acima do padrão do paciente "
                 f"(z robusto = {z[i]:.1f}) — possível mudança de quadro ou erro em lote.",
                 "z-score robusto sobre contagem diária", eventos_no_dia=int(por_dia.iloc[i]), z=round(float(z[i]), 2))

    return sorted(anomalias, key=lambda a: a.timestamp)
