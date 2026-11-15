"""Termos críticos e sentimento na transcrição da consulta -> anomalias.

Duas camadas:
* léxico local de termos críticos (funciona sem nuvem), ignorando perguntas e negações simples;
* resultado do Azure Text Analytics, quando disponível: sentimento, frases-chave e entidades
  de saúde — cuja detecção de negação tem prioridade sobre a do léxico.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime

from .. import config
from ..schemas import Anomalia, Severidade
from .azure_ia import AnaliseTexto

# categoria -> (severidade, padrão sobre texto SEM acentos e em minúsculas, orientação)
TERMOS_CRITICOS: dict[str, tuple[Severidade, str, str]] = {
    "risco de autolesão": (Severidade.CRITICA,
                           r"vontade de morrer|me matar|tirar (a )?minha (propria )?vida|nao quero mais viver|suicid|"
                           r"melhor (se eu )?morr",
                           "não deixar o paciente sozinho e acionar avaliação de saúde mental imediata"),
    "dor torácica": (Severidade.ALTA, r"dor no peito|aperto no peito|pressao no peito|dor (que )?(irradia|vai) para o braco",
                     "avaliar síndrome coronariana: ECG e sinais vitais"),
    "dispneia": (Severidade.ALTA, r"falta de ar|nao consigo respirar|sufoc|dificuldade (para|de) respirar|cansaco para respirar",
                 "aferir SpO2 e frequência respiratória"),
    "sinal neurológico agudo": (Severidade.ALTA,
                                r"fala (enrolada|arrastada)|boca torta|fraqueza (em|no|na|de) um (lado|braco|perna)|"
                                r"perd\w+ (a )?forca|visao (dupla|embacada de repente)|dormencia (em|de) um lado",
                                "aplicar escala de AVC e registrar o horário de início"),
    "síncope": (Severidade.ALTA, r"desmai|perdi a consciencia|apaguei", "avaliar causa cardíaca/neurológica"),
    "sangramento": (Severidade.ALTA, r"sangramento|sangrando|vomit\w+ sangue|sangue n[ao]s? (fezes|urina|escarro)|fezes (escuras|pretas)",
                    "avaliar sinais de instabilidade e anticoagulação em uso"),
    "reação alérgica": (Severidade.ALTA, r"inchaco n[ao] (rosto|lingua|garganta|boca)|coceira no corpo (todo|inteiro)|garganta fechando",
                        "avaliar anafilaxia e medicamento iniciado recentemente"),
    "problema com medicação": (Severidade.MEDIA,
                               r"parei de tomar|esqueci de tomar|tomei (o )?(dobro|a mais)|nao (estou|to) tomando|acabou o remedio",
                               "revisar adesão e conciliação medicamentosa"),
    "queda": (Severidade.MEDIA, r"\bcai\b|levei um tombo|sofri uma queda|escorreguei", "avaliar lesões e risco de queda"),
    "dor intensa": (Severidade.MEDIA, r"dor (muito )?(forte|intensa|insuportavel)|pior dor", "aplicar escala de dor e reavaliar analgesia"),
    "confusão mental": (Severidade.MEDIA, r"confus[ao]|nao (sei|lembro) onde estou|esquecendo (de )?tudo", "rastrear delirium"),
    "sofrimento psíquico": (Severidade.MEDIA, r"nao aguento mais|sem esperanca|muito triste|chor\w+ (muito|o dia todo)|desanimad",
                            "acolher e considerar avaliação psicológica"),
    "febre": (Severidade.BAIXA, r"\bfebre\b|calafrio", "aferir temperatura"),
    "rouquidão": (Severidade.BAIXA, r"rouc|rouquid|perdi a voz|voz (falh|fraca)", "avaliar via aérea/laringe se persistente"),
}
_NEGACAO = re.compile(r"\b(nao|nunca|sem|nega|negou|nenhum|nenhuma|jamais)\b[^,;]{0,25}$")
_NEGACAO_POS = re.compile(r"^[^,.;]{0,15}\b(eu )?nao (tenho|tive|sinto|senti|estou|apresent\w+)\b")  # "dor no peito eu não tenho"
_CATEGORIAS_SAUDE = {"SymptomOrSign", "Diagnosis"}


def sem_acento(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", sem_acento(s)).strip("_")


def termos_criticos(texto: str, analise: AnaliseTexto | None = None) -> dict[str, dict]:
    """categoria -> {trecho, frase, fonte}. Perguntas (“sente dor no peito?”) e negações são ignoradas."""
    negadas = [sem_acento(e.texto) for e in (analise.entidades_saude if analise else []) if e.negada]
    achados: dict[str, dict] = {}
    for frase in re.split(r"(?<=[.!?…])\s+", texto):
        if frase.rstrip().endswith("?"):
            continue  # pergunta do profissional, não relato do paciente
        plano = sem_acento(frase)
        for categoria, (_, padrao, _) in TERMOS_CRITICOS.items():
            if categoria in achados:
                continue
            for m in re.finditer(padrao, plano):
                negado_local = bool(_NEGACAO.search(plano[: m.start()]) or _NEGACAO_POS.search(plano[m.end():]))
                negado_azure = any(m.group(0) in n or n in m.group(0) for n in negadas)
                if not (negado_local or negado_azure):
                    achados[categoria] = {"trecho": m.group(0), "frase": frase.strip(), "fonte": "léxico local"}
                    break
    # sintomas/diagnósticos afirmados que o Text Analytics for health encontrou e o léxico confirma
    for e in (analise.entidades_saude if analise else []):
        if e.negada or e.categoria not in _CATEGORIAS_SAUDE:
            continue
        for categoria, (_, padrao, _) in TERMOS_CRITICOS.items():
            if re.search(padrao, sem_acento(e.texto)):
                achados.setdefault(categoria, {"trecho": e.texto, "frase": e.texto, "fonte": "léxico local"})
                achados[categoria]["fonte"] = "léxico local + Azure Text Analytics for health"
    return achados


# Palavras de conteúdo emocional. O sentimento da Azure, sozinho, não serve de alerta em consulta
# médica: verificado na Azure real, ele classifica como "negativa" qualquer frase com termos clínicos,
# inclusive "Não tive febre nem falta de ar" e "a dor diminuiu bastante".
_EMOCAO = re.compile(r"trist|chor|medo|assustad|apavorad|angusti|desesper|sozinh|ansios|nervos|preocupad|deprimid|"
                     r"desanim|sem esperanca|nao aguento|nao quero mais viver|vontade de morrer|sofr")


def negatividade_emocional(analise: AnaliseTexto) -> tuple[float, list]:
    """(maior escore negativo entre as frases com conteúdo emocional, essas frases em ordem de força).

    A Azure mede a intensidade; o filtro garante que a negatividade vem de emoção expressa
    pelo paciente, e não da simples menção a sintomas.
    """
    frases = [f for f in analise.frases_negativas
              if _EMOCAO.search(sem_acento(f.texto)) and not f.texto.rstrip().endswith("?")]
    return (max((f.negativo for f in frases), default=0.0), frases)



def anomalias_do_texto(texto: str, paciente_id: str, timestamp: datetime | None = None,
                       analise: AnaliseTexto | None = None,
                       historico_negativo: list[float] | None = None) -> list[Anomalia]:
    ts = timestamp or datetime.now()
    cfg = config.TEXTO
    anomalias: list[Anomalia] = []

    for categoria, achado in termos_criticos(texto, analise).items():
        sev, _, orientacao = TERMOS_CRITICOS[categoria]
        anomalias.append(Anomalia(
            paciente_id, "audio", f"relato_{slug(categoria)}", sev,
            f"Termo crítico na consulta — {categoria}: “{achado['frase'][:160]}”. Sugestão: {orientacao}.",
            ts, {"categoria": categoria, **achado}, f"termos críticos ({achado['fonte']})"))

    if analise is not None:
        emocional, frases = negatividade_emocional(analise)
        if emocional >= cfg["frase_negativa_min"]:
            exemplos = "; ".join(f"“{f.texto.strip()[:90]}”" for f in frases[:2])
            anomalias.append(Anomalia(
                paciente_id, "audio", "sentimento_negativo",
                Severidade.MEDIA if len(frases) >= 2 else Severidade.BAIXA,
                f"Sofrimento emocional expresso na consulta (sentimento negativo {emocional:.0%}) — {exemplos}. "
                "Considerar acolhimento/avaliação do estado emocional.",
                ts, {"sentimento_documento": analise.sentimento, "negativo_documento": round(analise.negativo, 3),
                     "negativo_emocional": round(emocional, 3), "frases": [f.texto.strip() for f in frases[:5]]},
                "Azure Text Analytics (sentimento) + filtro de conteúdo emocional"))
        if historico_negativo and len(historico_negativo) >= 2:
            base = sorted(historico_negativo)[len(historico_negativo) // 2]
            if emocional - base >= cfg["piora_sentimento"]:
                anomalias.append(Anomalia(
                    paciente_id, "audio", "piora_do_sentimento", Severidade.MEDIA,
                    f"Tom emocional bem mais negativo que nas {len(historico_negativo)} consultas anteriores "
                    f"(negativo {base:.0%} → {emocional:.0%}).",
                    ts, {"antes": round(base, 3), "agora": round(emocional, 3)},
                    "Azure Text Analytics vs. linha de base do paciente"))
    return anomalias
