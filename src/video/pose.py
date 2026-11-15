"""Estimativa de pose humana com dois backends intercambiáveis.

* `yolo`     — YOLOv8-pose (Ultralytics): multi-pessoa, rápido em GPU; padrão.
* `openpose` — modelo COCO original do OpenPose (CMU) executado pelo módulo DNN do OpenCV,
               sem precisar compilar o OpenPose. Decodificação de pessoa única (o paciente).

Ambos devolvem pontos-chave no formato COCO-17, em pixels: array [17, 3] = (x, y, confiança).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config

NOMES_COCO = ["nariz", "olho_e", "olho_d", "orelha_e", "orelha_d", "ombro_e", "ombro_d", "cotovelo_e",
              "cotovelo_d", "punho_e", "punho_d", "quadril_e", "quadril_d", "joelho_e", "joelho_d",
              "tornozelo_e", "tornozelo_d"]
ESQUELETO = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
             (12, 14), (14, 16), (0, 5), (0, 6)]
# índice no OpenPose-COCO (18 pontos) correspondente a cada ponto COCO-17
_OPENPOSE_PARA_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]


@dataclass
class Pessoa:
    kpts: np.ndarray                      # [17, 3] em pixels
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    conf: float = 1.0


def bbox_dos_kpts(kpts: np.ndarray, conf_min: float = 0.2) -> tuple[float, float, float, float]:
    ok = kpts[kpts[:, 2] >= conf_min]
    if len(ok) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(ok[:, 0].min()), float(ok[:, 1].min()), float(ok[:, 0].max()), float(ok[:, 1].max()))


class EstimadorPoseYOLO:
    nome = "yolov8-pose"

    def __init__(self, pesos: str = config.YOLO_POSE, conf: float = 0.35, dispositivo: str | None = None):
        from ultralytics import YOLO

        self.modelo = YOLO(pesos)
        self.conf = conf
        self.dispositivo = dispositivo

    def estimar(self, quadro: np.ndarray) -> list[Pessoa]:
        r = self.modelo.predict(quadro, conf=self.conf, verbose=False, device=self.dispositivo)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []
        kpts = r.keypoints.data.cpu().numpy()        # [n, 17, 3]
        caixas = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        return [Pessoa(k.astype(np.float32), tuple(map(float, b)), float(c))
                for k, b, c in zip(kpts, caixas, confs)]


class EstimadorOpenPose:
    """OpenPose (modelo COCO, 18 pontos) via cv2.dnn. Baixe os pesos com scripts/baixar_openpose.py."""

    nome = "openpose"

    def __init__(self, proto=config.OPENPOSE_PROTO, pesos=config.OPENPOSE_PESOS,
                 altura_entrada: int = 368, limiar: float = 0.1):
        import cv2

        if not (proto.exists() and pesos.exists()):
            raise FileNotFoundError(
                f"Pesos do OpenPose não encontrados em {pesos.parent}. Rode: python scripts/baixar_openpose.py")
        self.cv2 = cv2
        # Lê os arquivos em Python e passa os bytes: o OpenCV não abre caminhos com acentos no Windows
        self.rede = cv2.dnn.readNetFromCaffe(np.frombuffer(proto.read_bytes(), np.uint8),
                                             np.frombuffer(pesos.read_bytes(), np.uint8))
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:  # só em builds do OpenCV com CUDA
            self.rede.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.rede.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.altura_entrada = altura_entrada
        self.limiar = limiar

    def estimar(self, quadro: np.ndarray) -> list[Pessoa]:
        cv2 = self.cv2
        h, w = quadro.shape[:2]
        largura_entrada = int(round(self.altura_entrada * w / h / 8)) * 8
        blob = cv2.dnn.blobFromImage(quadro, 1 / 255.0, (largura_entrada, self.altura_entrada),
                                     (0, 0, 0), swapRB=False, crop=False)
        self.rede.setInput(blob)
        mapas = self.rede.forward()[0, :18]            # mapas de calor dos 18 pontos
        kp18 = np.zeros((18, 3), dtype=np.float32)
        for i, mapa in enumerate(mapas):
            _, conf, _, (px, py) = cv2.minMaxLoc(mapa)
            if conf >= self.limiar:
                kp18[i] = (px * w / mapa.shape[1], py * h / mapa.shape[0], conf)
        kpts = kp18[_OPENPOSE_PARA_COCO17]
        if (kpts[:, 2] > 0).sum() < 5:
            return []
        return [Pessoa(kpts, bbox_dos_kpts(kpts), float(kpts[kpts[:, 2] > 0, 2].mean()))]


def criar_estimador(backend: str = config.POSE_BACKEND):
    if backend == "openpose":
        return EstimadorOpenPose()
    if backend == "yolo":
        return EstimadorPoseYOLO()
    raise ValueError(f"backend de pose desconhecido: {backend}")
