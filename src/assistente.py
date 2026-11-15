"""Assistente Virtual de Monitoramento Clínico — orquestra todos os detectores,
transforma anomalias em alertas e responde à equipe sobre a situação dos pacientes."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .alertas import GerenciadorAlertas, canal_console, canal_jsonl, canal_webhook
from .audio.texto import negatividade_emocional
from .schemas import Alerta, Anomalia, Severidade

AVISO = ("Protótipo acadêmico de apoio à decisão: os alertas são achados de triagem e não substituem "
         "a avaliação da equipe assistencial.")

SISTEMA_LLM = """Você é o assistente virtual de monitoramento de uma unidade de internação hospitalar.
Você recebe, em JSON, os alertas gerados automaticamente por detectores de anomalias (vídeo do leito,
áudio de consultas, sinais vitais, ECG, prescrições e movimentação) e responde a perguntas da equipe
de enfermagem e médica.

Baseie-se apenas nos alertas fornecidos: se a informação pedida não estiver neles, diga que não está
disponível. Os alertas são achados de triagem, não diagnósticos — você pode relacionar achados e
sugerir o que reavaliar primeiro, mas não prescreve nem fecha diagnóstico; a decisão é da equipe.
Responda em português, de forma direta, começando pelo que é mais urgente."""


class AssistenteClinico:
    def __init__(self, gerenciador: GerenciadorAlertas | None = None, silencioso: bool = False):
        if gerenciador is None:
            canais = [canal_jsonl()] if silencioso else [canal_console, canal_jsonl()]
            if config.WEBHOOK_URL:
                canais.append(canal_webhook(config.WEBHOOK_URL))
            gerenciador = GerenciadorAlertas(canais)
        self.alertas = gerenciador
        self.pacientes: dict[str, dict] = {}       # metadados e resultados por paciente
        self._vozes: dict[str, list] = {}          # linha de base vocal por paciente
        self._sentimentos: dict[str, list[float]] = {}  # escore negativo das consultas anteriores

    def _registrar(self, paciente_id: str, origem: str, anomalias: list[Anomalia], extra: dict | None = None) -> list[Alerta]:
        p = self.pacientes.setdefault(paciente_id, {"id": paciente_id, "analises": {}})
        p["analises"][origem] = {"anomalias": len(anomalias), "em": datetime.now().isoformat(timespec="seconds"),
                                 **(extra or {})}
        return self.alertas.processar(anomalias)

    # ------------------------------------------------------------------ entradas
    def analisar_sinais_vitais(self, paciente_id: str, df: pd.DataFrame) -> list[Alerta]:
        from .series.sinais_vitais import detectar_anomalias_vitais, news2

        anomalias = detectar_anomalias_vitais(df, paciente_id)
        ultimo = df.sort_values("timestamp").iloc[-1]
        extra = {"ultima_leitura": {k: (str(v) if k == "timestamp" else float(v)) for k, v in ultimo.items()},
                 "news2_atual": int(news2(df.tail(1)).iloc[0]), "amostras": len(df)}
        return self._registrar(paciente_id, "sinais_vitais", anomalias, extra)

    def analisar_prescricoes(self, paciente_id: str, df: pd.DataFrame) -> list[Alerta]:
        from .series.prescricoes import detectar_anomalias_prescricao

        return self._registrar(paciente_id, "prescricao", detectar_anomalias_prescricao(df, paciente_id),
                               {"eventos": len(df)})

    def analisar_movimentacao(self, paciente_id: str, df: pd.DataFrame) -> list[Alerta]:
        from .series.movimentacao import detectar_anomalias_movimentacao

        return self._registrar(paciente_id, "movimentacao", detectar_anomalias_movimentacao(df, paciente_id),
                               {"intervalos": len(df)})

    def analisar_ecg(self, paciente_id: str, sinal_mv: np.ndarray, fs: float, inicio: datetime | None = None) -> list[Alerta]:
        from .series.ecg import analisar_ecg

        return self._registrar(paciente_id, "ecg", analisar_ecg(sinal_mv, fs, paciente_id, inicio),
                               {"duracao_s": round(len(sinal_mv) / fs, 1)})

    def analisar_audio(self, paciente_id: str, caminho: str | Path, quando: datetime | None = None,
                       transcricao: str | None = None) -> list[Alerta]:
        from .audio.analise import analisar_audio

        historico = self._vozes.setdefault(paciente_id, [])
        negativos = self._sentimentos.setdefault(paciente_id, [])
        r = analisar_audio(caminho, paciente_id, quando, baseline=list(historico), transcricao=transcricao,
                           historico_negativo=list(negativos))
        if r.features.fracao_vozeada >= 0.05:
            historico.append(r.features)
        if r.analise_texto:
            negativos.append(negatividade_emocional(r.analise_texto)[0])
        return self._registrar(paciente_id, "audio", r.anomalias,
                               {"arquivo": Path(caminho).name, "features": r.features.to_dict(),
                                "consultas": len(historico), **r.texto_dict()})

    def analisar_transcricao(self, paciente_id: str, texto: str, quando: datetime | None = None) -> list[Alerta]:
        """Analisa só o texto de uma consulta (já transcrita por outro sistema)."""
        from .audio.analise import analisar_transcricao

        negativos = self._sentimentos.setdefault(paciente_id, [])
        avisos: list[str] = []
        analise, anomalias = analisar_transcricao(texto, paciente_id, quando, list(negativos), avisos)
        if analise:
            negativos.append(negatividade_emocional(analise)[0])
        return self._registrar(paciente_id, "transcricao", anomalias,
                               {"transcricao": texto, "avisos": avisos,
                                **({"analise_texto": analise.to_dict()} if analise else {})})

    def analisar_video(self, paciente_id: str, caminho: str | Path | None = None, cenario: str | None = None,
                       inicio: datetime | None = None, saida_anotada: str | Path | None = None, **kw) -> list[Alerta]:
        from .video.pipeline import processar_cenario, processar_video

        if cenario:
            r = processar_cenario(cenario, paciente_id, inicio, saida_anotada, kw.get("params"))
        else:
            r = processar_video(caminho, paciente_id, inicio=inicio, saida_anotada=saida_anotada, **kw)
        extra = {"fonte": cenario and f"cenário simulado: {cenario}" or Path(str(caminho)).name, **r.resumo,
                 "video_anotado": str(r.video_anotado) if r.video_anotado else None}
        return self._registrar(paciente_id, "video", r.anomalias, extra)

    def analisar_pasta(self, pasta: str | Path, anotar_video: bool = True) -> list[Alerta]:
        """Processa tudo o que houver na pasta de um paciente (ver scripts/gerar_dados_sinteticos.py)."""
        pasta = Path(pasta)
        meta = json.loads((pasta / "paciente.json").read_text(encoding="utf-8"))
        pid = meta["id"]
        self.pacientes.setdefault(pid, {"id": pid, "analises": {}}).update(
            {k: meta[k] for k in ("leito", "idade", "motivo") if k in meta})
        emitidos: list[Alerta] = []
        if (f := pasta / "sinais_vitais.csv").exists():
            emitidos += self.analisar_sinais_vitais(pid, pd.read_csv(f, parse_dates=["timestamp"]))
        if (f := pasta / "prescricoes.csv").exists():
            emitidos += self.analisar_prescricoes(pid, pd.read_csv(f, parse_dates=["timestamp"]))
        if (f := pasta / "movimentacao.csv").exists():
            emitidos += self.analisar_movimentacao(pid, pd.read_csv(f, parse_dates=["timestamp"]))
        if (f := pasta / "ecg.npy").exists():
            emitidos += self.analisar_ecg(pid, np.load(f), meta.get("ecg_fs", 360),
                                          datetime.fromisoformat(meta["ecg_inicio"]) if "ecg_inicio" in meta else None)
        for wav in sorted((pasta / "audio").glob("*.wav")):  # nome: AAAA-MM-DD_consulta.wav
            emitidos += self.analisar_audio(pid, wav, datetime.fromisoformat(wav.stem[:10] + "T10:00:00"))
        video = meta.get("video")
        if video:
            saida = config.SAIDAS / f"{pid}_video_anotado.mp4" if anotar_video else None
            inicio = datetime.fromisoformat(video["inicio"]) if "inicio" in video else None
            if video["tipo"] == "cenario":
                emitidos += self.analisar_video(pid, cenario=video["valor"], inicio=inicio, saida_anotada=saida)
            else:
                emitidos += self.analisar_video(pid, caminho=(pasta / video["valor"]), inicio=inicio, saida_anotada=saida)
        return emitidos

    # ------------------------------------------------------------------ saídas
    def situacao(self) -> list[dict]:
        """Uma linha por paciente, do mais grave para o menos grave."""
        linhas = []
        for pid, p in self.pacientes.items():
            abertos = self.alertas.listar(pid, status="aberto")
            pior = max((a.anomalia.severidade for a in abertos), default=Severidade.INFO)
            contagem = {s.name: sum(a.anomalia.severidade == s for a in abertos) for s in reversed(Severidade) if s > 0}
            linhas.append({**{k: v for k, v in p.items() if k != "analises"}, "pior_severidade": pior.name,
                           "pior_nivel": int(pior), "alertas_abertos": len(abertos), "por_severidade": contagem,
                           "fontes": sorted(p["analises"]), "analises": p["analises"]})
        return sorted(linhas, key=lambda x: (-x["pior_nivel"], -x["alertas_abertos"]))

    def resumo_paciente(self, paciente_id: str) -> str:
        """Resumo determinístico (sem LLM) para passagem de plantão."""
        p = self.pacientes.get(paciente_id)
        if p is None:
            return f"Paciente {paciente_id} não encontrado."
        alertas = self.alertas.listar(paciente_id)
        cab = f"Paciente {paciente_id}" + (f" — leito {p['leito']}" if p.get("leito") else "")
        if not alertas:
            return f"{cab}: nenhuma anomalia detectada nas fontes analisadas ({', '.join(sorted(p['analises']))})."
        linhas = [f"{cab}: {len(alertas)} alerta(s), maior severidade {alertas[0].anomalia.severidade.rotulo}."]
        for a in alertas:
            an = a.anomalia
            rep = f" (x{a.ocorrencias})" if a.ocorrencias > 1 else ""
            estado = "" if a.status == "aberto" else f" [reconhecido por {a.reconhecido_por}]"
            linhas.append(f"- [{an.severidade.name}] {an.timestamp:%d/%m %H:%M} {an.origem}: {an.descricao}{rep}{estado}\n"
                          f"    Ação sugerida: {a.acao_sugerida} → {', '.join(a.destinatarios)}")
        linhas.append(AVISO)
        return "\n".join(linhas)

    def perguntar(self, pergunta: str, paciente_id: str | None = None) -> str:
        """Pergunta em linguagem natural sobre os alertas (Claude). Sem credencial, devolve o resumo determinístico."""
        try:
            import anthropic
        except ImportError:
            return self._sem_llm(paciente_id, "biblioteca `anthropic` não instalada")
        alertas = [a.to_dict() for a in self.alertas.listar(paciente_id)]
        contexto = json.dumps({"pacientes": [{k: v for k, v in s.items() if k != "analises"} for s in self.situacao()
                                             if paciente_id in (None, s["id"])],
                               "alertas": alertas}, ensure_ascii=False, default=str)
        try:
            resposta = anthropic.Anthropic().beta.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # se um classificador de segurança recusar, outro modelo responde
                system=SISTEMA_LLM,
                messages=[{"role": "user", "content": f"<alertas>\n{contexto}\n</alertas>\n\n{pergunta}"}],
            )
        except anthropic.AuthenticationError:
            return self._sem_llm(paciente_id, "credencial da Anthropic ausente ou inválida")
        except anthropic.RateLimitError:
            return self._sem_llm(paciente_id, "limite de requisições atingido, tente novamente em instantes")
        except anthropic.APIStatusError as e:
            return self._sem_llm(paciente_id, f"erro {e.status_code} da API")
        except anthropic.APIConnectionError:
            return self._sem_llm(paciente_id, "sem conexão com a API da Anthropic")
        except TypeError:  # o SDK levanta TypeError quando não encontra nenhuma credencial
            return self._sem_llm(paciente_id, "nenhuma credencial da Anthropic configurada (defina ANTHROPIC_API_KEY)")
        if resposta.stop_reason == "refusal":
            return self._sem_llm(paciente_id, "o modelo recusou a pergunta")
        return "".join(b.text for b in resposta.content if b.type == "text").strip() + f"\n\n{AVISO}"

    def _sem_llm(self, paciente_id: str | None, motivo: str) -> str:
        ids = [paciente_id] if paciente_id else [s["id"] for s in self.situacao()]
        corpo = "\n\n".join(self.resumo_paciente(i) for i in ids)
        return f"(Resposta em linguagem natural indisponível: {motivo}. Segue o resumo automático.)\n\n{corpo}"
