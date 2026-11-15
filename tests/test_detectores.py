"""Testes dos detectores (não exigem GPU nem os modelos YOLO)."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.alertas import GerenciadorAlertas
from src.schemas import Anomalia, Severidade

T0 = datetime(2026, 11, 10, 8, 0)


def tipos(anomalias):
    return {a.tipo for a in anomalias}


# ----------------------------------------------------------------------------- sinais vitais
@pytest.fixture(scope="module")
def vitais_normais():
    from gerar_dados_sinteticos import finalizar_vitais, vitais

    return finalizar_vitais(vitais(123))


def test_vitais_paciente_estavel_nao_gera_alerta(vitais_normais):
    from src.series.sinais_vitais import detectar_anomalias_vitais

    assert detectar_anomalias_vitais(vitais_normais, "P") == []


def test_vitais_dessaturacao_critica_e_persistencia(vitais_normais):
    from src.series.sinais_vitais import detectar_anomalias_vitais

    df = vitais_normais.copy()
    df.loc[500:505, "spo2"] = 84          # 30 min abaixo do limite crítico
    df.loc[700, "fc"] = 118               # uma única leitura na faixa de alerta = artefato
    an = detectar_anomalias_vitais(df, "P")
    spo2 = [a for a in an if a.tipo == "spo2_baixa"]
    assert len(spo2) == 1 and spo2[0].severidade is Severidade.CRITICA
    assert "fc_alta" not in tipos(an)


def test_vitais_tendencia_dentro_dos_limites(vitais_normais):
    from src.series.sinais_vitais import detectar_anomalias_vitais

    df = vitais_normais.copy()
    df.loc[len(df) - 150:, "fc"] += np.linspace(0, 22, 150).round()   # sobe ~22 bpm sem passar de 110
    assert "fc_tendencia_alta" in tipos(detectar_anomalias_vitais(df, "P"))


def test_news2_deterioracao():
    from src.series.sinais_vitais import news2

    df = pd.DataFrame({"fc": [75, 125], "pas": [120, 88], "spo2": [98, 90], "fr": [16, 26], "temp": [36.6, 39.2]})
    assert news2(df).tolist() == [0, 13]


# ----------------------------------------------------------------------------- prescrições
def presc(*linhas):
    return pd.DataFrame([dict(timestamp=T0 + timedelta(hours=h), medicamento=m, acao=a, dose_mg=d, intervalo_h=i,
                              via="IV", prescritor="Dr. X") for h, m, a, d, i in linhas])


def test_prescricao_erro_de_dose_10x():
    from src.series.prescricoes import detectar_anomalias_prescricao

    an = detectar_anomalias_prescricao(presc((0, "Heparina", "iniciar", 5000, 12), (24, "heparina", "alterar", 50000, 12)), "P")
    assert {"alteracao_dose", "dose_acima_maximo"} <= tipos(an)
    assert all(a.severidade is Severidade.CRITICA for a in an)


def test_prescricao_interacao_suspensao_e_ajuste_normal():
    from src.series.prescricoes import detectar_anomalias_prescricao

    an = detectar_anomalias_prescricao(presc(
        (0, "varfarina", "iniciar", 5, 24), (2, "ibuprofeno", "iniciar", 400, 8),
        (4, "ceftriaxona", "iniciar", 2000, 24), (30, "ceftriaxona", "suspender", 0, 24),
        (6, "dipirona", "iniciar", 1000, 6), (40, "dipirona", "alterar", 1000, 8)), "P")
    assert tipos(an) == {"interacao_medicamentosa", "suspensao_inesperada"}


# ----------------------------------------------------------------------------- movimentação
def test_movimentacao_imobilidade_e_controle():
    from gerar_dados_sinteticos import movimentacao
    from src.series.movimentacao import detectar_anomalias_movimentacao

    m = movimentacao(7)
    assert detectar_anomalias_movimentacao(m, "P") == []
    m.loc[200:219, ["mudancas_posicao", "fora_do_leito_min"]] = 0   # 5 h sem mudar de posição
    an = detectar_anomalias_movimentacao(m, "P")
    assert tipos(an) == {"imobilidade_prolongada"} and an[0].severidade is Severidade.ALTA


# ----------------------------------------------------------------------------- áudio
@pytest.mark.parametrize("perfil,esperado", [
    ("normal", set()), ("disfonia", {"disfonia"}), ("tremor", {"tremor_vocal"}), ("monotona", {"fala_monotona"}),
    ("dispneia", {"disfonia", "pausas_excessivas", "tosse_frequente"}),
])
def test_audio_perfis_vocais(perfil, esperado):
    from src.audio.analise import analisar_features
    from src.audio.features import extrair_features
    from src.audio.sintese import PERFIS, sintetizar_voz

    f = extrair_features(sintetizar_voz(semente=5, **PERFIS[perfil]), 16000)
    assert tipos(analisar_features(f, "P")) == esperado


def test_audio_f0_e_mudanca_vs_linha_de_base(tmp_path):
    from src.audio.analise import analisar_audio
    from src.audio.features import extrair_features
    from src.audio.sintese import PERFIS, salvar_wav, sintetizar_voz

    base = [extrair_features(sintetizar_voz(f0=200, semente=s), 16000) for s in (1, 2, 3)]
    assert all(abs(b.f0_media - 200) < 10 for b in base)
    wav = salvar_wav(tmp_path / "c.wav", sintetizar_voz(f0=200, semente=4, **PERFIS["disfonia"]))
    r = analisar_audio(wav, "P", baseline=base, transcricao="Doutor, estou com falta de ar e a voz falhando.")
    assert {"disfonia", "mudanca_vocal", "relato_dispneia", "relato_rouquidao"} <= tipos(r.anomalias)
    assert r.transcricao.origem == "fornecida"


# ----------------------------------------------------------------------------- vídeo (pontos-chave simulados)
@pytest.mark.parametrize("cenario,esperado", [
    ("normal", set()), ("queda", {"saida_do_leito", "queda"}), ("agitacao", {"agitacao"}),
    ("zona_restrita", {"saida_do_leito", "zona_restrita"}), ("ausente", {"saida_do_leito", "paciente_ausente"}),
])
def test_video_cenarios(cenario, esperado):
    from src.video.pipeline import processar_cenario

    r = processar_cenario(cenario, "P")
    assert tipos(r.anomalias) == esperado
    if cenario == "queda":
        queda = next(a for a in r.anomalias if a.tipo == "queda")
        assert queda.severidade is Severidade.CRITICA and 12.5 < queda.evidencias["t_video_s"] < 14.5


def test_zona_ponto_em_poligono():
    from src.video.objetos import Zona

    z = Zona("porta", "restrita", [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0)])   # triângulo
    assert z.contem(0.9, 0.3) and not z.contem(0.6, 0.9)


# ----------------------------------------------------------------------------- ECG
def test_ecg_tv_e_sinusal():
    from src import config
    from src.series.ecg import analisar_ecg, ler_fragmento

    if not (config.ECG_MODELO.exists() and config.ECG_DATASET.exists()):
        pytest.skip("modelo/base de ECG ausente")
    def tira(pasta, padrao):
        arqs = sorted((config.ECG_DATASET / pasta / "frag").glob(f"*{padrao}*.hea"))[:6]
        return np.concatenate([ler_fragmento(a.with_suffix(""))[0][:720] for a in arqs])
    assert analisar_ecg(tira("6_Sinus_rhythm", "_N_"), 360, "P") == []
    an = analisar_ecg(tira("1_Dangerous_VFL_VF", "_VF_"), 360, "P")
    assert an and an[0].severidade is Severidade.CRITICA


# ----------------------------------------------------------------------------- alertas
def test_alertas_dedup_escalonamento_e_reconhecimento():
    enviados = []
    g = GerenciadorAlertas(canais=[lambda a, ev: enviados.append((a.anomalia.tipo, ev))], cooldown_s=600)
    def an(minuto, sev, tipo="spo2_baixa"):
        return Anomalia("P1", "sinais_vitais", tipo, sev, "x", T0 + timedelta(minutes=minuto))
    g.processar([an(0, Severidade.MEDIA), an(3, Severidade.MEDIA), an(5, Severidade.CRITICA),
                 an(30, Severidade.MEDIA), an(31, Severidade.INFO, "outro")])
    assert enviados == [("spo2_baixa", "novo"), ("spo2_baixa", "escalonado"), ("spo2_baixa", "novo")]
    primeiro = g.alertas[0]
    assert primeiro.ocorrencias == 3 and "time de resposta rápida" in primeiro.destinatarios
    assert g.reconhecer(primeiro.id, "Enf. Ana").status == "reconhecido"
    assert [a.id for a in g.listar(status="aberto")] == [g.alertas[1].id]
