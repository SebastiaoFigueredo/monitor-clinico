"""Integração com os serviços de IA da Azure para os áudios de consulta.

* **Azure Speech to Text** — transcrição contínua do WAV, com tempo e confiança por trecho;
* **Azure Text Analytics (AI Language)** — sentimento (documento e frases), frases-chave e,
  opcionalmente, entidades de saúde (*Text Analytics for health*: sintomas, diagnósticos,
  medicamentos, com detecção de negação).

Credenciais no `.env`: AZURE_SPEECH_KEY + AZURE_SPEECH_REGION e AZURE_LANGUAGE_KEY + AZURE_LANGUAGE_ENDPOINT.
Sem elas as funções `*_configurado()` devolvem False e o restante do sistema segue funcionando
(biomarcadores vocais e termos críticos por léxico local).

ATENÇÃO: usar estes serviços envia áudio e texto da consulta para a nuvem da Azure. Com pacientes
reais isso exige contrato/BAA, região adequada e consentimento — ver README.
"""
from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import config

LIMITE_DOC = 5000  # o Text Analytics aceita até 5.120 caracteres por documento


# ----------------------------------------------------------------------------- estruturas
@dataclass
class Trecho:
    inicio_s: float
    fim_s: float
    texto: str
    confianca: float | None = None


@dataclass
class Transcricao:
    texto: str
    trechos: list[Trecho] = field(default_factory=list)
    origem: str = "azure-speech"
    idioma: str = "pt-BR"


@dataclass
class FraseSentimento:
    texto: str
    sentimento: str
    negativo: float


@dataclass
class EntidadeSaude:
    texto: str
    categoria: str            # SymptomOrSign, Diagnosis, MedicationName, ...
    confianca: float
    negada: bool = False      # "nega dor no peito" -> True
    normalizado: str | None = None


@dataclass
class AnaliseTexto:
    sentimento: str                                   # positive | neutral | negative | mixed
    positivo: float
    neutro: float
    negativo: float
    frases_negativas: list[FraseSentimento] = field(default_factory=list)
    frases_chave: list[str] = field(default_factory=list)
    entidades_saude: list[EntidadeSaude] = field(default_factory=list)
    origem: str = "azure-text-analytics"

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("positivo", "neutro", "negativo"):
            d[k] = round(d[k], 3)
        return d


# ----------------------------------------------------------------------------- Speech to Text
def fala_configurada() -> bool:
    return bool(config.AZURE_SPEECH_KEY and config.AZURE_SPEECH_REGION)


def transcrever(caminho: str | Path, idioma: str = config.AZURE_SPEECH_IDIOMA,
                termos_dica: list[str] | None = None, tempo_max_s: float = 600.0) -> Transcricao:
    """Transcreve um WAV PCM (mono, 16 kHz recomendado) com reconhecimento contínuo.

    `termos_dica` entra como *phrase list* e melhora o reconhecimento de nomes de medicamentos
    e termos clínicos pouco frequentes.
    """
    import azure.cognitiveservices.speech as speechsdk

    if not fala_configurada():
        raise RuntimeError("Azure Speech não configurado (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION)")
    cfg = speechsdk.SpeechConfig(subscription=config.AZURE_SPEECH_KEY, region=config.AZURE_SPEECH_REGION)
    cfg.speech_recognition_language = idioma
    cfg.output_format = speechsdk.OutputFormat.Detailed   # inclui confiança por trecho
    # o SDK é nativo e não lida bem com caminhos acentuados no Windows: envia os bytes por stream
    import wave

    with wave.open(str(caminho), "rb") as w:
        formato = speechsdk.audio.AudioStreamFormat(samples_per_second=w.getframerate(),
                                                    bits_per_sample=8 * w.getsampwidth(), channels=w.getnchannels())
        dados = w.readframes(w.getnframes())
    fluxo = speechsdk.audio.PushAudioInputStream(stream_format=formato)
    reconhecedor = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=speechsdk.audio.AudioConfig(stream=fluxo))
    if termos_dica:
        lista = speechsdk.PhraseListGrammar.from_recognizer(reconhecedor)
        for termo in termos_dica:
            lista.addPhrase(termo)

    trechos: list[Trecho] = []
    erro: list[str] = []
    fim = threading.Event()

    def ao_reconhecer(evt):
        r = evt.result
        if r.reason == speechsdk.ResultReason.RecognizedSpeech and r.text.strip():
            conf = None
            try:
                import json

                conf = json.loads(r.json)["NBest"][0]["Confidence"]
            except (KeyError, IndexError, ValueError):
                pass
            trechos.append(Trecho(r.offset / 1e7, (r.offset + r.duration) / 1e7, r.text.strip(), conf))

    def ao_cancelar(evt):
        d = evt.cancellation_details if hasattr(evt, "cancellation_details") else evt.result.cancellation_details
        if d.reason == speechsdk.CancellationReason.Error:
            erro.append(f"{d.error_code}: {d.error_details}")
        fim.set()

    reconhecedor.recognized.connect(ao_reconhecer)
    reconhecedor.canceled.connect(ao_cancelar)
    reconhecedor.session_stopped.connect(lambda _: fim.set())
    reconhecedor.start_continuous_recognition()
    fluxo.write(dados)
    fluxo.close()                      # fim do áudio -> o serviço encerra a sessão
    concluiu = fim.wait(tempo_max_s)
    reconhecedor.stop_continuous_recognition()
    if erro:
        raise RuntimeError(f"Azure Speech falhou — {erro[0]}")
    if not concluiu:
        raise TimeoutError(f"Azure Speech não concluiu em {tempo_max_s:.0f} s")
    return Transcricao(" ".join(t.texto for t in trechos), trechos, idioma=idioma)


# ----------------------------------------------------------------------------- Text Analytics
def linguagem_configurada() -> bool:
    return bool(config.AZURE_LANGUAGE_KEY and config.AZURE_LANGUAGE_ENDPOINT)


def _cliente_linguagem():
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    return TextAnalyticsClient(config.AZURE_LANGUAGE_ENDPOINT, AzureKeyCredential(config.AZURE_LANGUAGE_KEY))


def dividir_texto(texto: str, limite: int = LIMITE_DOC) -> list[str]:
    """Quebra em blocos <= limite respeitando o fim das frases."""
    blocos, atual = [], ""
    for frase in re.split(r"(?<=[.!?…])\s+", texto.strip()):
        while len(frase) > limite:  # frase gigante (transcrição sem pontuação): corta no espaço
            if atual:
                blocos.append(atual)
                atual = ""
            corte = frase.rfind(" ", 0, limite)
            corte = corte if corte > 0 else limite
            blocos.append(frase[:corte])
            frase = frase[corte:].strip()
        if atual and len(atual) + len(frase) + 1 > limite:
            blocos.append(atual)
            atual = frase
        else:
            atual = f"{atual} {frase}".strip()
    if atual:
        blocos.append(atual)
    return [b for b in blocos if b]


def analisar_texto(texto: str, idioma: str = "pt-BR", entidades_saude: bool | None = None,
                   cliente=None) -> AnaliseTexto:
    """Sentimento + frases-chave (+ entidades de saúde) da transcrição de uma consulta."""
    from azure.core.exceptions import HttpResponseError

    cliente = cliente or _cliente_linguagem()
    docs = dividir_texto(texto)
    if not docs:
        raise ValueError("texto vazio")
    if entidades_saude is None:
        entidades_saude = config.AZURE_ENTIDADES_SAUDE

    try:
        r_sent = [r for r in cliente.analyze_sentiment(docs, language=idioma, disable_service_logs=True) if not r.is_error]
        r_chave = [r for r in cliente.extract_key_phrases(docs, language=idioma, disable_service_logs=True) if not r.is_error]
    except HttpResponseError as e:
        raise RuntimeError(f"Azure Text Analytics falhou — {e.message}") from e
    if not r_sent:
        raise RuntimeError("Azure Text Analytics não devolveu análise de sentimento")

    # agrega os blocos ponderando pelo tamanho
    pesos = [len(d) for d in docs[: len(r_sent)]]
    total = sum(pesos)
    media = lambda attr: sum(getattr(r.confidence_scores, attr) * p for r, p in zip(r_sent, pesos)) / total  # noqa: E731
    pos, neu, neg = media("positive"), media("neutral"), media("negative")
    rotulos = {r.sentiment for r in r_sent}
    sentimento = r_sent[0].sentiment if len(rotulos) == 1 else (
        "mixed" if {"positive", "negative"} <= rotulos or "mixed" in rotulos
        else max((("positive", pos), ("neutral", neu), ("negative", neg)), key=lambda x: x[1])[0])

    negativas = [FraseSentimento(s.text, s.sentiment, s.confidence_scores.negative)
                 for r in r_sent for s in r.sentences if s.sentiment == "negative"]
    negativas.sort(key=lambda f: -f.negativo)
    chaves = list(dict.fromkeys(k for r in r_chave for k in r.key_phrases))

    entidades: list[EntidadeSaude] = []
    if entidades_saude:
        try:
            # o Text Analytics for health usa o código de idioma curto ("pt")
            for r in cliente.begin_analyze_healthcare_entities(
                    docs, language=idioma.split("-")[0], disable_service_logs=True).result():
                if r.is_error:
                    continue
                for e in r.entities:
                    negada = bool(e.assertion and e.assertion.certainty in ("negative", "negativePossible"))
                    entidades.append(EntidadeSaude(e.text, e.category, e.confidence_score, negada, e.normalized_text))
        except HttpResponseError as e:  # recurso sem o plano/idioma: segue sem entidades
            print(f"[azure] entidades de saúde indisponíveis: {e.message}")

    return AnaliseTexto(sentimento, pos, neu, neg, negativas[:8], chaves[:25], entidades)


# ----------------------------------------------------------------------------- diagnóstico
def diagnostico() -> list[dict]:
    """Testa as credenciais de verdade e explica, em português, o que corrigir.

    Devolve [{"servico", "ok", "mensagem"}]. Custa 1 registro de texto do Text Analytics e
    nada do Speech (só pede um token).
    """
    import urllib.error
    import urllib.request

    itens: list[dict] = []

    def add(servico: str, ok: bool, mensagem: str) -> None:
        itens.append({"servico": servico, "ok": ok, "mensagem": mensagem})

    arquivo_env = config.RAIZ / ".env"
    if not arquivo_env.exists():
        add("arquivo .env", False, f"Não existe {arquivo_env}. Copie .env.example para .env e preencha as chaves "
                                   "(passo a passo em GUIA_AZURE.md).")

    # --- Speech: pede um token de acesso (valida chave + região sem gastar cota) -----------
    if not fala_configurada():
        faltam = [n for n, v in (("AZURE_SPEECH_KEY", config.AZURE_SPEECH_KEY),
                                 ("AZURE_SPEECH_REGION", config.AZURE_SPEECH_REGION)) if not v]
        add("Azure Speech to Text", False, f"Falta definir no .env: {', '.join(faltam)}.")
    else:
        url = f"https://{config.AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        req = urllib.request.Request(url, data=b"", method="POST",
                                     headers={"Ocp-Apim-Subscription-Key": config.AZURE_SPEECH_KEY})
        try:
            urllib.request.urlopen(req, timeout=15).close()
            add("Azure Speech to Text", True, f"Chave válida na região {config.AZURE_SPEECH_REGION}.")
        except urllib.error.HTTPError as e:
            dica = {401: "chave inválida OU a chave é de outra região — confira AZURE_SPEECH_KEY e se "
                         "AZURE_SPEECH_REGION é a região do recurso (ex.: brazilsouth, eastus)",
                    403: "recurso bloqueado ou cota do plano gratuito esgotada",
                    429: "muitas requisições — cota do plano gratuito atingida, tente mais tarde"}.get(e.code, e.reason)
            add("Azure Speech to Text", False, f"HTTP {e.code}: {dica}.")
        except OSError as e:
            add("Azure Speech to Text", False,
                f"Não conectou em {url} ({e}). Região escrita errada (use o nome curto, ex.: brazilsouth) ou sem internet.")

    # --- Language / Text Analytics: uma análise mínima -------------------------------------
    if not linguagem_configurada():
        faltam = [n for n, v in (("AZURE_LANGUAGE_KEY", config.AZURE_LANGUAGE_KEY),
                                 ("AZURE_LANGUAGE_ENDPOINT", config.AZURE_LANGUAGE_ENDPOINT)) if not v]
        add("Azure Text Analytics", False, f"Falta definir no .env: {', '.join(faltam)}.")
    else:
        from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ServiceRequestError

        try:
            r = _cliente_linguagem().analyze_sentiment(["Estou bem hoje."], language="pt-BR")[0]
            if r.is_error:
                add("Azure Text Analytics", False, f"O serviço recusou o texto: {r.error.message}")
            else:
                add("Azure Text Analytics", True, f"Chave e endpoint válidos (sentimento de teste: {r.sentiment}).")
        except ClientAuthenticationError:
            add("Azure Text Analytics", False, "Chave recusada (401): confira AZURE_LANGUAGE_KEY e se ela pertence "
                                               "ao recurso do AZURE_LANGUAGE_ENDPOINT.")
        except ServiceRequestError as e:
            add("Azure Text Analytics", False, f"Não conectou ao endpoint ({e.message}). Ele deve ter o formato "
                                               "https://<nome-do-recurso>.cognitiveservices.azure.com/")
        except (HttpResponseError, ValueError) as e:
            add("Azure Text Analytics", False, f"Erro do serviço: {getattr(e, 'message', e)}")
    return itens
