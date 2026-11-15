"""Transcrição/Text Analytics: léxico local e integração com a Azure usando clientes simulados
(nenhuma chamada de rede; a Azure real é exercitada por scripts/validar_azure.py)."""
from types import SimpleNamespace as NS

import pytest

from src.audio import analise, azure_ia
from src.audio.texto import anomalias_do_texto, termos_criticos
from src.schemas import Severidade


def tipos(anomalias):
    return {a.tipo for a in anomalias}


def test_lexico_ignora_perguntas_e_negacoes():
    texto = ("O senhor sente dor no peito? Não, dor no peito eu não tenho. Nega febre. "
             "Mas estou com muita falta de ar e ontem eu caí no banheiro.")
    assert set(termos_criticos(texto)) == {"dispneia", "queda"}
    assert set(termos_criticos("Estou com dor no peito desde cedo.")) == {"dor torácica"}
    assert termos_criticos("Dormi bem, sem falta de ar, não tive sangramento.") == {}


def test_risco_de_autolesao_e_critico():
    an = anomalias_do_texto("Às vezes penso que não quero mais viver.", "P")
    assert [(a.tipo, a.severidade) for a in an] == [("relato_risco_de_autolesao", Severidade.CRITICA)]


class ClienteFalso:
    """Imita o TextAnalyticsClient: frases com 'triste'/'aguento' são negativas."""

    def __init__(self):
        self.chamadas = []

    def analyze_sentiment(self, docs, language=None, **kw):
        self.chamadas.append(("sentimento", language, len(docs)))
        saida = []
        for d in docs:
            frases = [f.strip() for f in d.split(".") if f.strip()]
            sent = [NS(text=f, sentiment="negative" if ("triste" in f or "aguento" in f) else "neutral",
                       confidence_scores=NS(positive=0.02, neutral=0.08, negative=0.9) if ("triste" in f or "aguento" in f)
                       else NS(positive=0.1, neutral=0.8, negative=0.1)) for f in frases]
            neg = sum(s.sentiment == "negative" for s in sent) / len(sent)
            saida.append(NS(is_error=False, sentiment="negative" if neg >= 0.5 else "neutral", sentences=sent,
                            confidence_scores=NS(positive=0.05, neutral=0.95 - neg * 0.9, negative=neg * 0.9)))
        return saida

    def extract_key_phrases(self, docs, language=None, **kw):
        return [NS(is_error=False, key_phrases=["falta de ar", "remédio da pressão"]) for _ in docs]

    def begin_analyze_healthcare_entities(self, docs, language=None, **kw):
        self.chamadas.append(("saude", language, len(docs)))
        ents = [NS(text="dor no peito", category="SymptomOrSign", confidence_score=0.97, normalized_text=None,
                   assertion=NS(certainty="negative")),
                NS(text="falta de ar", category="SymptomOrSign", confidence_score=0.95, normalized_text="Dyspnea",
                   assertion=None)]
        return NS(result=lambda: [NS(is_error=False, entities=ents) for _ in docs])


def test_analise_azure_sentimento_entidades_e_negacao():
    cliente = ClienteFalso()
    texto = "Estou muito triste. Não aguento mais. Paciente refere ausência de dor no peito. Tenho falta de ar."
    a = azure_ia.analisar_texto(texto, cliente=cliente, entidades_saude=True)
    assert a.sentimento == "negative" and a.negativo == pytest.approx(0.45)
    assert [f.texto for f in a.frases_negativas] == ["Estou muito triste", "Não aguento mais"]
    assert ("saude", "pt", 1) in cliente.chamadas and ("sentimento", "pt-BR", 1) in cliente.chamadas
    assert [(e.texto, e.negada) for e in a.entidades_saude] == [("dor no peito", True), ("falta de ar", False)]

    # "ausência de dor no peito" engana o léxico local; a negação da Azure corrige
    assert "dor torácica" in termos_criticos(texto) and "dor torácica" not in termos_criticos(texto, a)
    an = anomalias_do_texto(texto, "P", analise=a, historico_negativo=[0.05, 0.08])
    assert tipos(an) == {"relato_dispneia", "relato_sofrimento_psiquico", "sentimento_negativo", "piora_do_sentimento"}
    assert "Azure" in next(x for x in an if x.tipo == "relato_dispneia").detector


def test_texto_longo_e_dividido_em_blocos():
    cliente = ClienteFalso()
    azure_ia.analisar_texto("Estou bem hoje. " * 800, cliente=cliente, entidades_saude=False)
    assert cliente.chamadas[0][2] == 3
    assert all(len(b) <= azure_ia.LIMITE_DOC for b in azure_ia.dividir_texto("palavra " * 3000))


def test_fluxo_de_audio_sem_e_com_azure(tmp_path, monkeypatch):
    from src.audio.sintese import salvar_wav, sintetizar_voz

    wav = salvar_wav(tmp_path / "c.wav", sintetizar_voz(duracao=6))
    monkeypatch.setattr(azure_ia, "fala_configurada", lambda: False)
    monkeypatch.setattr(azure_ia, "linguagem_configurada", lambda: False)
    r = analise.analisar_audio(wav, "P")
    assert r.transcricao is None and r.anomalias == [] and "não configurado" in r.avisos[0]

    # Azure "configurada": Speech devolve a transcrição e o Text Analytics o sentimento
    monkeypatch.setattr(azure_ia, "fala_configurada", lambda: True)
    monkeypatch.setattr(azure_ia, "linguagem_configurada", lambda: True)
    monkeypatch.setattr(azure_ia, "transcrever", lambda caminho, **kw: azure_ia.Transcricao(
        "Estou muito triste. Tenho falta de ar.", [azure_ia.Trecho(0.5, 3.0, "Estou muito triste.", 0.93)]))
    monkeypatch.setattr(azure_ia, "_cliente_linguagem", ClienteFalso)
    monkeypatch.setattr(analise.config, "AZURE_ENTIDADES_SAUDE", False)
    r = analise.analisar_audio(wav, "P")
    assert tipos(r.anomalias) == {"relato_dispneia", "relato_sofrimento_psiquico", "sentimento_negativo"}
    d = r.texto_dict()
    assert d["transcricao_origem"] == "azure-speech" and d["trechos"][0]["confianca"] == 0.93
    assert d["analise_texto"]["sentimento"] in {"negative", "mixed"}

    # falha de nuvem não derruba a análise acústica
    def falha(*a, **k):
        raise RuntimeError("401 chave inválida")
    monkeypatch.setattr(azure_ia, "transcrever", falha)
    r = analise.analisar_audio(wav, "P")
    assert r.transcricao is None and "Azure Speech to Text falhou" in r.avisos[0]

    # um .txt ao lado do áudio dispensa a transcrição
    wav.with_suffix(".txt").write_text("Hoje cedo vomitei sangue.", encoding="utf-8")
    r = analise.analisar_audio(wav, "P")
    assert r.transcricao.origem == "arquivo .txt" and "relato_sangramento" in tipos(r.anomalias)


def test_sentimento_so_alerta_com_conteudo_emocional():
    """Regressão da Azure real: frases com termos clínicos vêm como 'negativas' mesmo sendo boas notícias."""
    F = azure_ia.FraseSentimento
    boa_noticia = azure_ia.AnaliseTexto("negative", 0.05, 0.09, 0.86, [
        F("Não tive febre nem falta de ar.", "negative", 0.97), F("Estou me sentindo bem, a dor diminuiu bastante.", "negative", 0.90)])
    assert anomalias_do_texto("Estou me sentindo bem, a dor diminuiu bastante. Não tive febre nem falta de ar.", "P",
                              analise=boa_noticia, historico_negativo=[0.0, 0.0, 0.0]) == []
    assustada = azure_ia.AnaliseTexto("negative", 0.0, 0.05, 0.95, [F("Fiquei muito assustada com o exame.", "negative", 0.96)])
    an = anomalias_do_texto("Fiquei muito assustada com o exame.", "P", analise=assustada)
    assert [(a.tipo, a.severidade) for a in an] == [("sentimento_negativo", Severidade.BAIXA)]
