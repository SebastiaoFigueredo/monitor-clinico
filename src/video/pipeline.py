"""Processamento de vídeos clínicos: pose + objetos + zonas -> anomalias e vídeo anotado."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config
from ..schemas import Anomalia
from .analise import AnalisadorPostural
from .objetos import Deteccao, Zona, zona_leito_de_deteccoes
from .pose import ESQUELETO, Pessoa

CORES_SEV = {4: (0, 0, 255), 3: (0, 100, 255), 2: (0, 200, 255), 1: (0, 220, 180), 0: (200, 200, 200)}


@dataclass
class ResultadoVideo:
    anomalias: list[Anomalia]
    resumo: dict
    serie_movimentacao: list[dict] = field(default_factory=list)
    video_anotado: Path | None = None
    zonas: list[Zona] = field(default_factory=list)


def desenhar(quadro: np.ndarray, pessoas: list[Pessoa], zonas: list[Zona], objetos: list[Deteccao],
             avisos: list[tuple[str, int]], rodape: str = "") -> np.ndarray:
    import cv2

    h, w = quadro.shape[:2]
    for z in zonas:
        pts = (np.array(z.poligono) * (w, h)).astype(np.int32)
        cor = (80, 200, 80) if z.tipo == "leito" else (60, 60, 230)
        cv2.polylines(quadro, [pts], True, cor, 2)
        cv2.putText(quadro, _sem_acento(z.nome), tuple(pts[0] + (6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)
    for o in objetos:
        if o.classe == 0:
            continue
        x1, y1, x2, y2 = map(int, o.bbox)
        cv2.rectangle(quadro, (x1, y1), (x2, y2), (200, 160, 60), 1)
        cv2.putText(quadro, _sem_acento(f"{o.nome} {o.conf:.2f}"), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 160, 60), 1)
    for p in pessoas:
        for a, b in ESQUELETO:
            if p.kpts[a, 2] > 0.3 and p.kpts[b, 2] > 0.3:
                cv2.line(quadro, tuple(map(int, p.kpts[a, :2])), tuple(map(int, p.kpts[b, :2])), (255, 200, 0), 2)
        for x, y, c in p.kpts:
            if c > 0.3:
                cv2.circle(quadro, (int(x), int(y)), 3, (0, 255, 255), -1)
    for i, (texto, sev) in enumerate(avisos[-4:]):
        y = 34 + 34 * i
        (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(quadro, (8, y - 24), (20 + tw, y + 8), CORES_SEV[sev], -1)
        cv2.putText(quadro, texto, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if rodape:
        cv2.putText(quadro, rodape, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return quadro


def _sem_acento(s: str) -> str:  # as fontes Hershey do OpenCV não têm acentos
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def processar_video(caminho: str | Path, paciente_id: str, zonas: list[Zona] | None = None,
                    backend: str = config.POSE_BACKEND, saida_anotada: str | Path | None = None,
                    inicio: datetime | None = None, fps_analise: float = config.VIDEO_FPS_ANALISE,
                    detectar_objetos: bool = True, params: dict | None = None,
                    estimador=None, detector=None) -> ResultadoVideo:
    """Analisa um arquivo de vídeo (ou índice de câmera) e devolve as anomalias encontradas."""
    import cv2

    from .objetos import DetectorObjetos
    from .pose import criar_estimador

    cap = cv2.VideoCapture(caminho if isinstance(caminho, int) else str(caminho))
    if not cap.isOpened():
        raise FileNotFoundError(f"não foi possível abrir o vídeo: {caminho}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    passo = max(1, int(round(fps / fps_analise)))

    estimador = estimador or criar_estimador(backend)
    detector = detector or (DetectorObjetos() if detectar_objetos else None)
    zonas = list(zonas or [])
    analisador = AnalisadorPostural(paciente_id, w, h, zonas, inicio, params)

    gravador = None
    if saida_anotada:
        Path(saida_anotada).parent.mkdir(parents=True, exist_ok=True)
        gravador = cv2.VideoWriter(str(saida_anotada), cv2.VideoWriter_fourcc(*"mp4v"), fps / passo, (w, h))

    anomalias: list[Anomalia] = []
    avisos: list[tuple[str, int]] = []
    objetos: list[Deteccao] = []
    i = -1
    while True:
        ok, quadro = cap.read()
        if not ok:
            break
        i += 1
        if i % passo:
            continue
        t = i / fps
        pessoas = estimador.estimar(quadro)
        if detector is not None and (i // passo) % 5 == 0:  # objetos mudam devagar: 1 a cada 5 quadros analisados
            objetos = detector.detectar(quadro)
            if not analisador._zona("leito"):  # sem zona configurada: usa o leito visto pelo YOLOv8
                z = zona_leito_de_deteccoes(objetos, w, h)
                if z is not None:
                    analisador.zonas.append(z)
        novas = analisador.atualizar(t, pessoas, objetos)
        anomalias += novas
        avisos += [(_sem_acento(f"[{t:5.1f}s] {a.tipo.upper()} ({a.severidade.name})"), int(a.severidade)) for a in novas]
        if gravador is not None:
            gravador.write(desenhar(quadro, pessoas, analisador.zonas, objetos, avisos,
                                    f"paciente {paciente_id} | t={t:6.1f}s | pose: {estimador.nome}"))
    cap.release()
    if gravador is not None:
        gravador.release()
    return ResultadoVideo(anomalias, analisador.resumo(), analisador.serie_movimentacao(),
                          Path(saida_anotada) if saida_anotada else None, analisador.zonas)


def processar_cenario(nome: str, paciente_id: str, inicio: datetime | None = None,
                      saida_anotada: str | Path | None = None, params: dict | None = None) -> ResultadoVideo:
    """Roda a mesma análise sobre um cenário sintético de pontos-chave (sem modelo de visão)."""
    from . import simulador as sim

    quadros, zonas = sim.cenario(nome)
    analisador = AnalisadorPostural(paciente_id, sim.LARGURA, sim.ALTURA, zonas, inicio, params)
    gravador = None
    if saida_anotada:
        import cv2

        Path(saida_anotada).parent.mkdir(parents=True, exist_ok=True)
        gravador = cv2.VideoWriter(str(saida_anotada), cv2.VideoWriter_fourcc(*"mp4v"), sim.FPS, (sim.LARGURA, sim.ALTURA))
    anomalias, avisos = [], []
    for i, pessoas in enumerate(quadros):
        t = i / sim.FPS
        novas = analisador.atualizar(t, pessoas)
        anomalias += novas
        avisos += [(_sem_acento(f"[{t:5.1f}s] {a.tipo.upper()} ({a.severidade.name})"), int(a.severidade)) for a in novas]
        if gravador is not None:
            tela = np.full((sim.ALTURA, sim.LARGURA, 3), 40, dtype=np.uint8)
            gravador.write(desenhar(tela, pessoas, zonas, [], avisos,
                                    f"paciente {paciente_id} | t={t:6.1f}s | cenario simulado: {nome}"))
    if gravador is not None:
        gravador.release()
    return ResultadoVideo(anomalias, analisador.resumo(), analisador.serie_movimentacao(),
                          Path(saida_anotada) if saida_anotada else None, zonas)
