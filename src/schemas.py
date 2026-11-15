"""Estruturas de dados compartilhadas entre os módulos."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any


class Severidade(IntEnum):
    INFO = 0
    BAIXA = 1
    MEDIA = 2
    ALTA = 3
    CRITICA = 4

    @property
    def rotulo(self) -> str:
        return {0: "informativo", 1: "baixa", 2: "média", 3: "alta", 4: "crítica"}[int(self)]


@dataclass
class Anomalia:
    """Achado bruto produzido por um detector (vídeo, áudio ou séries)."""

    paciente_id: str
    origem: str            # "video" | "audio" | "sinais_vitais" | "ecg" | "prescricao" | "movimentacao"
    tipo: str              # identificador curto, ex.: "queda", "spo2_baixa"
    severidade: Severidade
    descricao: str
    timestamp: datetime = field(default_factory=datetime.now)
    evidencias: dict[str, Any] = field(default_factory=dict)
    detector: str = ""     # técnica que gerou o achado (regra, z-robusto, IsolationForest...)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severidade"] = self.severidade.name
        d["timestamp"] = self.timestamp.isoformat(timespec="seconds")
        return d


@dataclass
class Alerta:
    """Anomalia promovida a alerta, com destinatários e estado de atendimento."""

    anomalia: Anomalia
    destinatarios: list[str]
    acao_sugerida: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    criado_em: datetime = field(default_factory=datetime.now)
    ocorrencias: int = 1
    status: str = "aberto"          # aberto | reconhecido
    reconhecido_por: str | None = None

    def to_dict(self) -> dict[str, Any]:
        a = self.anomalia
        return {
            "id": self.id,
            "paciente_id": a.paciente_id,
            "origem": a.origem,
            "tipo": a.tipo,
            "severidade": a.severidade.name,
            "severidade_nivel": int(a.severidade),
            "descricao": a.descricao,
            "detector": a.detector,
            "evidencias": a.evidencias,
            "ocorrido_em": a.timestamp.isoformat(timespec="seconds"),
            "criado_em": self.criado_em.isoformat(timespec="seconds"),
            "destinatarios": self.destinatarios,
            "acao_sugerida": self.acao_sugerida,
            "ocorrencias": self.ocorrencias,
            "status": self.status,
            "reconhecido_por": self.reconhecido_por,
        }
