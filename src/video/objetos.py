"""Detecção de objetos (YOLOv8) e definição das áreas críticas do quarto.

Zonas são polígonos em coordenadas normalizadas (0–1) com um tipo:
* `leito`    — onde o paciente deveria estar; sair dela gera evento, deitar FORA dela pode ser queda;
* `restrita` — área crítica (porta de saída, janela, bancada de medicação, equipamentos).

A zona do leito pode ser inferida automaticamente da classe "bed" do YOLOv8 (COCO).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config

# classes COCO relevantes num quarto hospitalar
CLASSES_INTERESSE = {0: "pessoa", 56: "cadeira", 57: "sofá", 59: "leito", 60: "mesa", 62: "monitor/tv",
                     39: "frasco", 41: "copo", 43: "faca", 76: "tesoura"}
CLASSES_PERIGOSAS = {43, 76}


@dataclass
class Deteccao:
    classe: int
    nome: str
    conf: float
    bbox: tuple[float, float, float, float]  # pixels


@dataclass
class Zona:
    nome: str
    tipo: str                                 # "leito" | "restrita"
    poligono: list[tuple[float, float]]       # normalizado 0–1
    _arr: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._arr = np.asarray(self.poligono, dtype=float)

    def contem(self, x: float, y: float) -> bool:
        """Ponto em polígono (ray casting), coordenadas normalizadas."""
        p = self._arr
        x1, y1 = p[:, 0], p[:, 1]
        x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
        cruza = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        return bool(cruza.sum() % 2)

    @classmethod
    def retangulo(cls, nome: str, tipo: str, x1: float, y1: float, x2: float, y2: float) -> "Zona":
        return cls(nome, tipo, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def carregar_zonas(caminho: str | Path) -> list[Zona]:
    """JSON: [{"nome": "leito 12", "tipo": "leito", "poligono": [[0.1, 0.4], ...]}, ...]"""
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return [Zona(z["nome"], z["tipo"], [tuple(p) for p in z["poligono"]]) for z in dados]


def zona_leito_de_deteccoes(deteccoes: list[Deteccao], largura: int, altura: int) -> Zona | None:
    camas = [d for d in deteccoes if d.classe == 59]
    if not camas:
        return None
    x1, y1, x2, y2 = max(camas, key=lambda d: d.conf).bbox
    return Zona.retangulo("leito (detectado pelo YOLOv8)", "leito", x1 / largura, y1 / altura, x2 / largura, y2 / altura)


class DetectorObjetos:
    def __init__(self, pesos: str = config.YOLO_OBJETOS, conf: float = 0.35, dispositivo: str | None = None):
        from ultralytics import YOLO

        self.modelo = YOLO(pesos)
        self.conf = conf
        self.dispositivo = dispositivo

    def detectar(self, quadro: np.ndarray) -> list[Deteccao]:
        r = self.modelo.predict(quadro, conf=self.conf, verbose=False, device=self.dispositivo,
                                classes=list(CLASSES_INTERESSE))[0]
        if r.boxes is None:
            return []
        return [Deteccao(int(c), CLASSES_INTERESSE.get(int(c), str(int(c))), float(p), tuple(map(float, b)))
                for c, p, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
