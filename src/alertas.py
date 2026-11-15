"""Geração automática de alertas para a equipe a partir das anomalias detectadas.

Responsabilidades:
* roteamento por severidade/origem (quem precisa ser avisado);
* deduplicação com janela de resfriamento (evita fadiga de alarmes);
* escalonamento quando a mesma anomalia se repete ou piora;
* entrega em canais plugáveis (console, arquivo JSONL, webhook).
"""
from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from pathlib import Path

from . import config
from .schemas import Alerta, Anomalia, Severidade

Canal = Callable[[Alerta, str], None]  # (alerta, evento) -> None; evento: novo | escalonado

ROTEAMENTO = {
    Severidade.CRITICA: ["time de resposta rápida", "médico plantonista", "enfermagem do setor"],
    Severidade.ALTA: ["médico plantonista", "enfermagem do setor"],
    Severidade.MEDIA: ["enfermagem do setor"],
    Severidade.BAIXA: ["enfermagem do setor (painel)"],
}
EXTRA_POR_ORIGEM = {"prescricao": ["farmácia clínica"], "audio": ["médico assistente"]}

ACOES = {
    "queda": "Ir ao leito imediatamente, avaliar consciência e lesões, acionar protocolo de queda.",
    "imobilidade_prolongada": "Checar responsividade e realizar mudança de decúbito.",
    "agitacao": "Avaliar dor, delirium ou desconforto respiratório; verificar dispositivos.",
    "saida_do_leito": "Verificar o paciente; risco de queda se houver restrição de deambulação.",
    "zona_restrita": "Verificar presença na área crítica e orientar/retirar.",
    "paciente_ausente": "Localizar o paciente.",
    "postura_inclinada": "Avaliar posicionamento e risco de queda da cadeira/leito.",
    "relato_risco_de_autolesao": "Não deixar o paciente sozinho; acionar avaliação de saúde mental imediatamente.",
    "relato_dor_toracica": "Avaliar à beira-leito, ECG e sinais vitais imediatos.",
    "relato_dispneia": "Aferir SpO2/FR e avaliar esforço respiratório.",
    "arritmia": "Avaliar o paciente e o traçado; se instável, acionar código azul.",
    "escore_alerta_precoce": "Reavaliar sinais vitais e comunicar o médico conforme o escore.",
    "interacao_medicamentosa": "Farmácia clínica deve revisar a associação com o prescritor.",
    "dose_acima_maximo": "Confirmar a dose com o prescritor antes da administração.",
    "alteracao_dose": "Confirmar com o prescritor se a mudança de dose é intencional.",
    "suspensao_inesperada": "Confirmar com o prescritor a suspensão do medicamento.",
}
ACAO_PADRAO = {
    "sinais_vitais": "Repetir a aferição, avaliar o paciente à beira-leito e comunicar o médico.",
    "ecg": "Avaliar o paciente e o traçado de ECG.",
    "prescricao": "Revisar a prescrição com o prescritor/farmácia clínica.",
    "movimentacao": "Avaliar o paciente e revisar o plano de mobilidade.",
    "audio": "Considerar avaliação clínica dirigida (achado de triagem, não diagnóstico).",
    "video": "Verificar o paciente no leito.",
}


# ----------------------------------------------------------------------------- canais
def canal_console(alerta: Alerta, evento: str) -> None:
    a = alerta.anomalia
    marca = {4: "!!!", 3: "!! ", 2: "!  ", 1: ".  "}.get(int(a.severidade), "   ")
    print(
        f"[ALERTA {marca}] {a.severidade.name:<7} {a.timestamp:%d/%m %H:%M} "
        f"paciente={a.paciente_id} {a.origem}/{a.tipo} ({evento}, x{alerta.ocorrencias}) — {a.descricao}"
    )


def canal_jsonl(caminho: Path = config.ALERTAS_ARQUIVO) -> Canal:
    trava = threading.Lock()

    def _enviar(alerta: Alerta, evento: str) -> None:
        with trava, open(caminho, "a", encoding="utf-8") as f:
            f.write(json.dumps({"evento": evento, **alerta.to_dict()}, ensure_ascii=False, default=str) + "\n")

    return _enviar


def canal_webhook(url: str, timeout: float = 5.0) -> Canal:
    """POST JSON para um webhook (Teams/Slack/sistema de chamada de enfermagem)."""

    def _enviar(alerta: Alerta, evento: str) -> None:
        corpo = json.dumps({"evento": evento, **alerta.to_dict()}, default=str).encode()
        req = urllib.request.Request(url, data=corpo, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=timeout).close()
        except OSError as e:  # falha de entrega não pode derrubar o monitoramento
            print(f"[alertas] webhook falhou: {e}")

    return _enviar


# ----------------------------------------------------------------------------- gerenciador
class GerenciadorAlertas:
    def __init__(
        self,
        canais: Iterable[Canal] | None = None,
        cooldown_s: int = config.ALERTA_COOLDOWN_S,
        severidade_minima: Severidade = Severidade.BAIXA,
    ):
        self.canais: list[Canal] = list(canais) if canais is not None else [canal_console]
        self.cooldown = timedelta(seconds=cooldown_s)
        self.severidade_minima = severidade_minima
        self.alertas: list[Alerta] = []
        self._ultimo: dict[tuple[str, str, str], Alerta] = {}
        self._notificado_em: dict[tuple[str, str, str], datetime] = {}
        self._trava = threading.Lock()

    def processar(self, anomalias: Iterable[Anomalia]) -> list[Alerta]:
        """Converte anomalias em alertas; devolve os alertas novos ou escalonados."""
        emitidos: list[Alerta] = []
        for an in sorted(anomalias, key=lambda x: x.timestamp):
            if an.severidade < self.severidade_minima:
                continue
            with self._trava:
                alerta, evento = self._registrar(an)
            if evento:
                emitidos.append(alerta)
                for canal in self.canais:
                    canal(alerta, evento)
        return emitidos

    def _registrar(self, an: Anomalia) -> tuple[Alerta, str | None]:
        chave = (an.paciente_id, an.origem, an.tipo)
        anterior = self._ultimo.get(chave)
        # Janela fixa a partir da última notificação: condição persistente volta a alertar
        # a cada `cooldown`, mas não a cada amostra.
        if anterior and abs(an.timestamp - self._notificado_em[chave]) <= self.cooldown:
            anterior.ocorrencias += 1
            if an.severidade > anterior.anomalia.severidade:  # piora -> escalona e renotifica
                anterior.anomalia = an
                anterior.destinatarios = self._destinatarios(an)
                anterior.status = "aberto"
                self._notificado_em[chave] = an.timestamp
                return anterior, "escalonado"
            return anterior, None
        alerta = Alerta(
            anomalia=an,
            destinatarios=self._destinatarios(an),
            acao_sugerida=ACOES.get(an.tipo, ACAO_PADRAO.get(an.origem, "Avaliar o paciente.")),
        )
        self._ultimo[chave] = alerta
        self._notificado_em[chave] = an.timestamp
        self.alertas.append(alerta)
        return alerta, "novo"

    @staticmethod
    def _destinatarios(an: Anomalia) -> list[str]:
        dest = list(ROTEAMENTO.get(an.severidade, ROTEAMENTO[Severidade.BAIXA]))
        for extra in EXTRA_POR_ORIGEM.get(an.origem, []):
            if extra not in dest:
                dest.append(extra)
        return dest

    # ------------------------------------------------------------------ consulta
    def listar(self, paciente_id: str | None = None, status: str | None = None,
               severidade_minima: Severidade = Severidade.INFO) -> list[Alerta]:
        itens = [
            a for a in self.alertas
            if (paciente_id is None or a.anomalia.paciente_id == paciente_id)
            and (status is None or a.status == status)
            and a.anomalia.severidade >= severidade_minima
        ]
        return sorted(itens, key=lambda a: (-int(a.anomalia.severidade), a.anomalia.timestamp))

    def reconhecer(self, alerta_id: str, profissional: str) -> Alerta | None:
        for a in self.alertas:
            if a.id == alerta_id:
                a.status, a.reconhecido_por = "reconhecido", profissional
                return a
        return None

    def limpar(self) -> None:
        self.alertas.clear()
        self._ultimo.clear()
        self._notificado_em.clear()
