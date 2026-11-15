"""Gera um vídeo curto com PIXELS REAIS de uma pessoa (foto de exemplo do Ultralytics) que fica
em pé, "cai" (rotação de 90° em 0,8 s) e permanece no chão — para validar o pipeline completo
(YOLOv8-pose/OpenPose + YOLOv8 objetos + análise postural) sem filmar pacientes."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402


def gerar(saida: Path = config.DADOS / "video" / "validacao_queda.mp4", fps: int = 15) -> Path:
    import ultralytics

    foto = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    pessoa = foto[395:905, 40:250]                      # homem à esquerda da foto
    ph, pw = 400, int(400 * pessoa.shape[1] / pessoa.shape[0])
    pessoa = cv2.resize(pessoa, (pw, ph))
    W, H = 960, 720
    fundo = np.full((H, W, 3), (185, 190, 195), np.uint8)
    cv2.rectangle(fundo, (0, 600), (W, H), (120, 130, 140), -1)   # piso
    pe = (420, 640)                                              # ponto de apoio (pés)
    saida.parent.mkdir(parents=True, exist_ok=True)
    gravador = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    rng = np.random.default_rng(0)

    def quadro(angulo: float, dx: float = 0.0) -> np.ndarray:
        tela = fundo.copy()
        camada = np.zeros_like(tela)
        mascara = np.zeros((H, W), np.uint8)
        x0, y0 = int(pe[0] - pw / 2 + dx), pe[1] - ph
        camada[y0:y0 + ph, x0:x0 + pw] = pessoa
        mascara[y0:y0 + ph, x0:x0 + pw] = 255
        M = cv2.getRotationMatrix2D((pe[0] + dx, pe[1]), -angulo, 1.0)
        camada = cv2.warpAffine(camada, M, (W, H))
        mascara = cv2.warpAffine(mascara, M, (W, H))
        tela[mascara > 127] = camada[mascara > 127]
        return tela

    for i in range(4 * fps):                                     # em pé, com leve balanço
        gravador.write(quadro(2 * np.sin(i / 6), rng.normal(0, 1)))
    n = int(0.8 * fps)
    for i in range(n):                                           # queda
        gravador.write(quadro(90 * ((i + 1) / n) ** 1.5))
    for i in range(6 * fps):                                     # no chão
        gravador.write(quadro(90 + rng.normal(0, 0.3)))
    gravador.release()
    return saida


if __name__ == "__main__":
    print(gerar())
