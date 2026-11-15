"""Cenários sintéticos de pontos-chave (COCO-17) para testar e demonstrar a análise postural
sem filmar pacientes reais. Cada cenário devolve, quadro a quadro, a lista de pessoas que um
estimador de pose devolveria, mais as zonas do quarto."""
from __future__ import annotations

import numpy as np

from .objetos import Zona
from .pose import Pessoa, bbox_dos_kpts

LARGURA, ALTURA, FPS = 1280, 720, 5.0

# esqueleto em pé, altura unitária, origem entre os pés, y para cima
_MOLDE = np.array([
    (0, .93), (-.02, .95), (.02, .95), (-.04, .94), (.04, .94),      # cabeça
    (-.10, .80), (.10, .80), (-.13, .62), (.13, .62), (-.14, .46), (.14, .46),  # braços
    (-.07, .50), (.07, .50), (-.075, .27), (.075, .27), (-.08, .02), (.08, .02),  # pernas
])
_BRACOS, _PERNAS = [7, 8, 9, 10], [13, 14, 15, 16]

ZONAS_PADRAO = [
    Zona.retangulo("leito", "leito", 0.05, 0.45, 0.47, 0.82),
    Zona.retangulo("porta de saída", "restrita", 0.86, 0.25, 1.0, 1.0),
]


def esqueleto(cx: float, chao: float, altura: float = 0.5, rot: float = 0.0, agito: float = 0.0,
              fase: float = 0.0, rng: np.random.Generator | None = None) -> Pessoa:
    """cx/chao normalizados (0–1); altura como fração da altura do quadro; rot em graus (90 = deitado)."""
    m = _MOLDE.copy()
    if agito:  # membros oscilando
        m[_BRACOS, 0] += agito * np.sin(fase + np.array([0, 1.7, 0.4, 2.3]))
        m[_BRACOS, 1] += agito * np.cos(1.3 * fase + np.array([0.5, 0, 2.0, 1.0]))
        m[_PERNAS, 0] += 0.6 * agito * np.sin(0.9 * fase + np.array([1.0, 0, 2.5, 0.7]))
    a = np.radians(rot)
    x = m[:, 0] * np.cos(a) + m[:, 1] * np.sin(a)
    y = -m[:, 0] * np.sin(a) + m[:, 1] * np.cos(a)
    h_px = altura * ALTURA
    px = cx * LARGURA + x * h_px
    py = chao * ALTURA - y * h_px
    k = np.column_stack([px, py, np.full(17, 0.9)]).astype(np.float32)
    if rng is not None:
        k[:, :2] += rng.normal(0, 0.8, (17, 2))  # tremor típico do estimador
    return Pessoa(k, bbox_dos_kpts(k))


def cenario(nome: str, semente: int = 0) -> tuple[list[list[Pessoa]], list[Zona]]:
    rng = np.random.default_rng(semente)
    quadros: list[list[Pessoa]] = []

    def add(segundos: float, fn):
        n = int(segundos * FPS)
        for i in range(n):
            p = fn(i / max(n - 1, 1), len(quadros) / FPS)
            quadros.append([p] if p is not None else [])

    def no_leito(_, t):  # deitado no leito, respirando/ajustando levemente
        return esqueleto(0.10, 0.63, rot=90, agito=0.004, fase=1.5 * t, rng=rng)

    def levantar(u, t):  # do leito para em pé ao lado
        return esqueleto(0.10 + 0.48 * u, 0.63 + 0.27 * u, rot=90 * (1 - u), rng=rng)

    def andar(x0, x1):
        return lambda u, t: esqueleto(x0 + (x1 - x0) * u, 0.90, agito=0.012, fase=6 * t, rng=rng)

    if nome == "normal":
        add(40, no_leito)
    elif nome == "queda":
        add(6, no_leito)
        add(3, levantar)
        add(4, andar(0.58, 0.70))
        add(0.8, lambda u, t: esqueleto(0.70, 0.90, rot=90 * u ** 1.5, rng=rng))   # queda em 0,8 s
        add(15, lambda u, t: esqueleto(0.70, 0.90, rot=90, rng=rng))               # imóvel no chão
    elif nome == "agitacao":
        add(6, no_leito)
        add(16, lambda u, t: esqueleto(0.10, 0.63, rot=90, agito=0.10, fase=9 * t, rng=rng))
        add(6, no_leito)
    elif nome == "zona_restrita":
        add(5, no_leito)
        add(3, levantar)
        add(5, andar(0.58, 0.93))
        add(6, lambda u, t: esqueleto(0.93, 0.90, rng=rng))
    elif nome == "ausente":
        add(5, no_leito)
        add(3, levantar)
        add(4, andar(0.58, 1.05))
        add(26, lambda u, t: None)
    else:
        raise ValueError(f"cenário desconhecido: {nome}")
    return quadros, ZONAS_PADRAO


CENARIOS = ["normal", "queda", "agitacao", "zona_restrita", "ausente"]
