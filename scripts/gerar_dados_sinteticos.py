"""Gera 4 pacientes FICTÍCIOS em data/pacientes/ com anomalias plantadas e conhecidas,
para demonstrar e validar o sistema sem usar dados reais de pacientes.

P001 — deterioração tipo sepse (vitais), antibiótico suspenso cedo, queda de atividade, voz dispneica
P002 — controle: tudo normal (não deve gerar alertas relevantes)
P003 — erro de dose 10x de heparina + AINE, dessaturação transitória, deambulação noturna, queda (vídeo)
P004 — tendência de bradicardia, TV no ECG, tremor vocal novo, imobilidade prolongada, agitação
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.audio.sintese import PERFIS, salvar_wav, sintetizar_voz  # noqa: E402

FIM = datetime(2026, 11, 14, 8, 0)
DIAS_VITAIS, DIAS_MOV = 4, 5
RAIZ = config.DADOS / "pacientes"


FALA_ROTINA = "Estou bem, sem queixas hoje. Dormi bem e me alimentei normalmente."
FALAS = {  # transcrições fictícias da consulta mais recente
    "P001": "O senhor sente dor no peito? Não, dor no peito eu não tenho. Mas desde ontem estou com muita falta de ar, "
            "piora quando eu deito. Tive febre e calafrios a noite inteira.",
    "P002": "Estou me sentindo bem, a dor diminuiu bastante. Consegui caminhar no corredor hoje. "
            "Não tive febre nem falta de ar.",
    "P003": "Minha voz está rouca desde ontem. Hoje cedo notei sangue na urina e fiquei muito assustada. "
            "Também escorreguei no banheiro de madrugada.",
    "P004": "Eu não aguento mais ficar aqui, estou muito triste e sem esperança. Às vezes fico confuso e não lembro "
            "onde estou. Em casa eu parei de tomar o remédio da pressão.",
}


# ----------------------------------------------------------------------------- sinais vitais
def vitais(semente: int) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    ts = pd.date_range(FIM - timedelta(days=DIAS_VITAIS), FIM, freq="5min", inclusive="left")
    n = len(ts)
    hora = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    circ = np.sin(2 * np.pi * (hora - 9) / 24)  # pico à tarde, vale de madrugada

    def ar1(dp, phi=0.9):
        e = rng.normal(0, dp * np.sqrt(1 - phi ** 2), n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x

    return pd.DataFrame({
        "timestamp": ts,
        "fc": 74 + rng.normal(0, 2) + 5 * circ + ar1(2.5),
        "pas": 122 + rng.normal(0, 3) + 6 * circ + ar1(4),
        "pad": 77 + rng.normal(0, 2) + 3 * circ + ar1(3),
        "spo2": 97 + ar1(0.6),
        "fr": 16 + ar1(1.0),
        "temp": 36.6 + 0.25 * circ + ar1(0.08),
    })


def finalizar_vitais(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("fc", "pas", "pad", "fr"):
        df[c] = df[c].round()
    df["spo2"] = df["spo2"].round().clip(upper=100)
    df["temp"] = df["temp"].round(1)
    return df


def rampa(n_total: int, n_final: int) -> np.ndarray:
    r = np.zeros(n_total)
    r[-n_final:] = np.linspace(0, 1, n_final) ** 1.3
    return r


# ----------------------------------------------------------------------------- movimentação
def movimentacao(semente: int) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    ts = pd.date_range(FIM - timedelta(days=DIAS_MOV), FIM, freq="15min", inclusive="left")
    dia = (ts.hour >= 7) & (ts.hour < 21)
    n = len(ts)
    return pd.DataFrame({
        "timestamp": ts,
        "atividade": np.clip(np.where(dia, rng.normal(0.35, 0.08, n), rng.normal(0.03, 0.015, n)), 0, 1).round(3),
        "fora_do_leito_min": np.where(dia & (rng.random(n) < 0.25), rng.integers(3, 11, n), 0),
        "mudancas_posicao": np.where(dia, rng.poisson(2.2, n), rng.poisson(1.3, n)),
    })


def entre(df: pd.DataFrame, inicio: datetime, fim: datetime) -> pd.Series:
    return (df["timestamp"] >= inicio) & (df["timestamp"] < fim)


# ----------------------------------------------------------------------------- prescrições
def prescricoes(linhas: list[tuple]) -> pd.DataFrame:
    """linhas: (dias antes do fim, hora, medicamento, ação, dose_mg, intervalo_h, via, prescritor)"""
    reg = []
    for d, h, med, acao, dose, intervalo, via, quem in linhas:
        quando = (FIM - timedelta(days=d)).replace(hour=h, minute=0)
        reg.append(dict(timestamp=quando, medicamento=med, acao=acao, dose_mg=dose, intervalo_h=intervalo,
                        via=via, prescritor=quem))
    return pd.DataFrame(reg).sort_values("timestamp")


# ----------------------------------------------------------------------------- ECG
def tira_ecg(blocos: list[tuple[str, str, int]]) -> np.ndarray | None:
    """Concatena fragmentos reais da base PhysioNet: [(pasta, padrão do nome, quantidade), ...]."""
    from src.series.ecg import ler_fragmento

    if not config.ECG_DATASET.exists():
        print(f"  (base de ECG não encontrada em {config.ECG_DATASET}; ECG omitido)")
        return None
    partes = []
    for pasta, padrao, qtd in blocos:
        arquivos = sorted((config.ECG_DATASET / pasta / "frag").glob(f"*{padrao}*.hea"))[:qtd]
        for hea in arquivos:
            sinal, _ = ler_fragmento(hea.with_suffix(""))
            partes.append(sinal[:720] - np.median(sinal[:720]))
    return np.concatenate(partes).astype(np.float32)


# ----------------------------------------------------------------------------- escrita
def salvar(pid: str, meta: dict, vit=None, presc=None, mov=None, ecg=None, vozes=()):
    pasta = RAIZ / pid
    (pasta / "audio").mkdir(parents=True, exist_ok=True)
    if vit is not None:
        finalizar_vitais(vit).to_csv(pasta / "sinais_vitais.csv", index=False)
    if presc is not None:
        presc.to_csv(pasta / "prescricoes.csv", index=False)
    if mov is not None:
        mov.to_csv(pasta / "movimentacao.csv", index=False)
    if ecg is not None:
        np.save(pasta / "ecg.npy", ecg)
        meta.update(ecg_fs=360, ecg_inicio=(FIM - timedelta(hours=1)).isoformat())
    for dias_atras, perfil, f0, semente in vozes:
        data = (FIM - timedelta(days=dias_atras)).date().isoformat()
        salvar_wav(pasta / "audio" / f"{data}_consulta.wav", sintetizar_voz(f0=f0, semente=semente, **PERFIS[perfil]))
        # A voz sintética não é fala inteligível: a transcrição fictícia vai num .txt ao lado do áudio
        # (o sistema usa o .txt quando existe; com áudio real, quem transcreve é o Azure Speech to Text).
        fala = FALAS[pid] if dias_atras == 0 else FALA_ROTINA
        (pasta / "audio" / f"{data}_consulta.txt").write_text(fala, encoding="utf-8")
    (pasta / "paciente.json").write_text(json.dumps({"id": pid, **meta}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {pid}: {meta.get('motivo', '')}")


def main():
    print("Gerando pacientes fictícios em", RAIZ)
    d = lambda dias, h=0: (FIM - timedelta(days=dias)).replace(hour=h, minute=0)  # noqa: E731

    # ---- P001: deterioração tipo sepse nas últimas 10 h --------------------------------
    v = vitais(1)
    r = rampa(len(v), 120)
    v["fc"] += 52 * r; v["pas"] -= 40 * r; v["pad"] -= 24 * r  # noqa: E702
    v["temp"] += 2.4 * r; v["fr"] += 11 * r; v["spo2"] -= 7 * r  # noqa: E702
    m = movimentacao(1)
    m.loc[entre(m, d(1, 7), d(1, 21)), "atividade"] *= 0.25
    salvar("P001", dict(leito="301-A", idade=71, motivo="pneumonia; evolui com sinais de sepse"),
           v, prescricoes([
               (4, 9, "ceftriaxona", "iniciar", 2000, 24, "IV", "Dr. A"),
               (4, 9, "dipirona", "iniciar", 1000, 6, "IV", "Dr. A"),
               (4, 9, "enoxaparina", "iniciar", 40, 24, "SC", "Dr. A"),
               (2, 10, "ceftriaxona", "suspender", 0, 24, "IV", "Dr. B"),
               (1, 15, "omeprazol", "iniciar", 40, 24, "VO", "Dr. B"),
           ]), m, vozes=[(0, "dispneia", 118, 11)])

    # ---- P002: controle -------------------------------------------------------------------
    salvar("P002", dict(leito="302-B", idade=54, motivo="pós-operatório sem intercorrências (controle)",
                        video=dict(tipo="cenario", valor="normal", inicio=d(0, 6).isoformat())),
           vitais(2), prescricoes([
               (4, 8, "omeprazol", "iniciar", 40, 24, "VO", "Dra. C"),
               (4, 8, "dipirona", "iniciar", 1000, 6, "VO", "Dra. C"),
               (4, 8, "enalapril", "iniciar", 10, 12, "VO", "Dra. C"),
               (2, 8, "dipirona", "alterar", 1000, 8, "VO", "Dra. C"),
           ]), movimentacao(2),
           tira_ecg([("6_Sinus_rhythm", "_N_", 10)]),
           vozes=[(9, "normal", 205, 21), (6, "normal", 205, 22), (3, "normal", 205, 23), (0, "normal", 205, 24)])

    # ---- P003: erro de prescrição, dessaturação, deambulação noturna, queda --------------
    v = vitais(3)
    ep = entre(v, d(2, 14), d(2, 14) + timedelta(minutes=30))
    v.loc[ep, "spo2"] -= np.array([4, 8, 11, 11, 9, 5])
    pico = entre(v, d(1, 16), d(1, 16) + timedelta(minutes=15))
    v.loc[pico, "fc"] += 24
    m = movimentacao(3)
    noite = entre(m, d(1, 1), d(1, 1) + timedelta(minutes=90))
    m.loc[noite, "fora_do_leito_min"] = 12
    m.loc[noite, "atividade"] = 0.5
    salvar("P003", dict(leito="303-A", idade=79, motivo="TVP em anticoagulação; risco de queda",
                        video=dict(tipo="arquivo", valor="../../video/validacao_queda.mp4",
                                   inicio=d(0, 5).isoformat())),
           v, prescricoes([
               (4, 9, "heparina", "iniciar", 5000, 12, "SC", "Dr. D"),
               (4, 9, "metoprolol", "iniciar", 50, 12, "VO", "Dr. D"),
               (3, 9, "heparina", "alterar", 50000, 12, "SC", "Dr. E"),     # 10x: erro de digitação
               (2, 11, "cetoprofeno", "iniciar", 100, 12, "IV", "Dr. E"),   # AINE + anticoagulante
               (1, 9, "metoprolol", "suspender", 0, 12, "VO", "Dr. E"),
           ]), m,
           vozes=[(9, "normal", 110, 31), (6, "normal", 110, 32), (3, "normal", 110, 33), (0, "disfonia", 110, 34)])

    # ---- P004: bradicardia progressiva, TV, tremor vocal, imobilidade, agitação -----------
    v = vitais(4)
    v["fc"] -= 21 * rampa(len(v), 260)
    m = movimentacao(4)
    imovel = entre(m, d(1, 13), d(1, 18) + timedelta(minutes=30))
    m.loc[imovel, ["mudancas_posicao", "fora_do_leito_min"]] = 0
    m.loc[imovel, "atividade"] = 0.01
    m.loc[entre(m, d(2, 3), d(2, 3) + timedelta(minutes=75)), "atividade"] = 0.8
    salvar("P004", dict(leito="304-B", idade=68, motivo="FA em amiodarona; investigação neurológica",
                        video=dict(tipo="cenario", valor="agitacao", inicio=d(0, 3).isoformat())),
           v, prescricoes([
               (4, 9, "amiodarona", "iniciar", 200, 8, "VO", "Dra. F"),
               (4, 9, "enoxaparina", "iniciar", 60, 12, "SC", "Dra. F"),
               (3, 20, "haloperidol", "iniciar", 2.5, 8, "IV", "Dr. G"),    # QT com amiodarona
               (2, 9, "morfina", "iniciar", 5, 4, "IV", "Dr. G"),
               (2, 21, "tramadol", "iniciar", 100, 8, "IV", "Dr. H"),       # dois opioides
           ]), m,
           tira_ecg([("6_Sinus_rhythm", "_N_", 8), ("3_Threatening_VT", "VTHR", 6), ("6_Sinus_rhythm", "_N_", 4)]),
           vozes=[(9, "normal", 128, 41), (6, "normal", 128, 42), (3, "normal", 128, 43), (0, "tremor", 128, 44)])


if __name__ == "__main__":
    if not (config.DADOS / "video" / "validacao_queda.mp4").exists():
        from gerar_video_validacao import gerar

        gerar()
    main()
