"""Análise postural e de eventos a partir de sequências de pontos-chave.

Independe do modelo de pose: recebe, quadro a quadro, as pessoas detectadas (COCO-17) e os
objetos, e mantém uma pequena máquina de estados por paciente para detectar:

queda · saída do leito · permanência em zona restrita · agitação motora ·
imobilidade prolongada · postura inclinada sustentada · paciente ausente · objeto perigoso próximo
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .. import config
from ..schemas import Anomalia, Severidade
from .objetos import CLASSES_PERIGOSAS, Deteccao, Zona
from .pose import Pessoa

CONF_MIN = 0.3
OMBROS, QUADRIS, TORNOZELOS = [5, 6], [11, 12], [15, 16]
EXTREMIDADES = [7, 8, 9, 10, 13, 14, 15, 16]  # cotovelos, punhos, joelhos, tornozelos


@dataclass
class EstadoQuadro:
    t: float
    visivel: bool
    kpts: np.ndarray | None = None        # pixels
    centro: tuple[float, float] | None = None   # quadril, normalizado
    pes: tuple[float, float] | None = None      # ponto de apoio, normalizado
    angulo: float = 0.0                   # tronco vs. vertical (0 = ereto, 90 = horizontal)
    deitado: bool = False
    ereto: bool = False
    no_leito: bool | None = None          # None = sem zona de leito definida
    mov: float = 0.0                      # deslocamento global (diagonais/s, defasagem de 1 s)
    mov_rel: float = 0.0                  # movimento dos membros relativo ao quadril (diagonais/s)
    vel_centro: float = 0.0


def _media_pontos(kpts: np.ndarray, indices: list[int]) -> np.ndarray | None:
    p = kpts[indices]
    p = p[p[:, 2] >= CONF_MIN]
    return p[:, :2].mean(axis=0) if len(p) else None


class AnalisadorPostural:
    def __init__(self, paciente_id: str, largura: int, altura: int, zonas: list[Zona] | None = None,
                 inicio: datetime | None = None, params: dict | None = None):
        self.paciente_id = paciente_id
        self.w, self.h = largura, altura
        self.diag = float(np.hypot(largura, altura))
        self.zonas = list(zonas or [])
        self.inicio = inicio or datetime.now()
        self.p = {**config.VIDEO, **(params or {})}
        self.hist: deque[EstadoQuadro] = deque()
        self.registro: list[EstadoQuadro] = []
        self._queda_candidata: float | None = None
        self._caido = False
        self._desde: dict[str, float | None] = {}   # início da condição corrente por evento
        self._emitido: set[str] = set()             # eventos já alertados no episódio corrente
        self._ultimo_visto: float | None = None
        self._viu_paciente = False
        self._esteve_no_leito = False
        self._evid_queda: dict = {}

    # ------------------------------------------------------------------ utilidades
    def _zona(self, tipo: str) -> list[Zona]:
        return [z for z in self.zonas if z.tipo == tipo]

    def _anomalia(self, t: float, tipo: str, sev: Severidade, desc: str, **ev) -> Anomalia:
        ev["t_video_s"] = round(t, 1)
        return Anomalia(self.paciente_id, "video", tipo, sev, desc,
                        self.inicio + timedelta(seconds=t), ev, "análise postural (pontos-chave) + zonas")

    def _escolher_paciente(self, pessoas: list[Pessoa]) -> Pessoa | None:
        if not pessoas:
            return None
        anterior = next((e for e in reversed(self.hist) if e.visivel), None)

        def centro(p: Pessoa) -> np.ndarray:
            c = _media_pontos(p.kpts, QUADRIS)
            if c is None:
                c = np.array([(p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2])
            return c / (self.w, self.h)

        if anterior is not None and anterior.centro is not None:  # rastreamento por proximidade
            return min(pessoas, key=lambda p: float(np.hypot(*(centro(p) - anterior.centro))))
        leitos = self._zona("leito")
        no_leito = [p for p in pessoas if any(z.contem(*centro(p)) for z in leitos)]
        candidatos = no_leito or pessoas
        return max(candidatos, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))

    def _condicao(self, nome: str, ativa: bool, t: float, duracao: float) -> bool:
        """True uma única vez quando `ativa` se mantém por `duracao` segundos; rearma ao cessar."""
        if not ativa:
            self._desde[nome] = None
            self._emitido.discard(nome)
            return False
        if self._desde.get(nome) is None:
            self._desde[nome] = t
        if nome not in self._emitido and t - self._desde[nome] >= duracao:
            self._emitido.add(nome)
            return True
        return False

    # ------------------------------------------------------------------ por quadro
    def _medir(self, t: float, pessoa: Pessoa) -> EstadoQuadro:
        k = pessoa.kpts
        x1, y1, x2, y2 = pessoa.bbox
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        ombro, quadril = _media_pontos(k, OMBROS), _media_pontos(k, QUADRIS)
        if ombro is not None and quadril is not None:
            dx, dy = abs(ombro[0] - quadril[0]), abs(ombro[1] - quadril[1])
            angulo = float(np.degrees(np.arctan2(dx, dy)))
        else:  # sem tronco visível: usa a proporção da caixa
            angulo = 80.0 if bw / bh > 1.4 else 10.0
        centro_px = quadril if quadril is not None else np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        pes_px = _media_pontos(k, TORNOZELOS)
        if pes_px is None:
            pes_px = np.array([(x1 + x2) / 2, y2])
        e = EstadoQuadro(t, True, k, tuple(centro_px / (self.w, self.h)), tuple(pes_px / (self.w, self.h)), angulo)
        e.deitado = angulo > self.p["queda_angulo_tronco"] or bw / bh > 1.6
        e.ereto = angulo < 35.0 and not e.deitado
        leitos = self._zona("leito")
        e.no_leito = any(z.contem(*e.centro) for z in leitos) if leitos else None

        visiveis = [a for a in self.hist if a.visivel]
        if visiveis:
            ant = visiveis[-1]
            dt = max(t - ant.t, 1e-3)
            ambos = (k[:, 2] >= CONF_MIN) & (ant.kpts[:, 2] >= CONF_MIN)
            c_ant = np.array(ant.centro) * (self.w, self.h)
            e.vel_centro = float(np.hypot(*(centro_px - c_ant))) / self.diag / dt
            ext = np.zeros(17, dtype=bool)
            ext[EXTREMIDADES] = True
            if (ambos & ext).any():
                ambos = ambos & ext
                rel = (k[ambos, :2] - centro_px) - (ant.kpts[ambos, :2] - c_ant)
                e.mov_rel = float(np.hypot(rel[:, 0], rel[:, 1]).mean()) / self.diag / dt
            ref = next((a for a in reversed(visiveis) if t - a.t >= 1.0), None)  # ~1 s atrás: filtra tremor do modelo
            if ref is not None:
                ambos = (k[:, 2] >= CONF_MIN) & (ref.kpts[:, 2] >= CONF_MIN)
                if ambos.any():
                    d = k[ambos, :2] - ref.kpts[ambos, :2]
                    e.mov = float(np.hypot(d[:, 0], d[:, 1]).mean()) / self.diag / (t - ref.t)
            else:
                e.mov = np.nan  # ainda sem referência
        else:
            e.mov = np.nan
        return e

    def atualizar(self, t: float, pessoas: list[Pessoa], objetos: list[Deteccao] | None = None) -> list[Anomalia]:
        p = self.p
        saida: list[Anomalia] = []
        pessoa = self._escolher_paciente(pessoas)

        # ---------------- paciente ausente ----------------
        if pessoa is None:
            e = EstadoQuadro(t, False)
            self.hist.append(e)
            self.registro.append(e)
            if self._viu_paciente and self._condicao("ausente", True, t, p["ausencia_s"]):
                saida.append(self._anomalia(t, "paciente_ausente", Severidade.MEDIA,
                                            f"Paciente fora do campo de visão há {p['ausencia_s']:.0f} s.",
                                            segundos=p["ausencia_s"]))
            return saida
        self._condicao("ausente", False, t, 0)
        self._viu_paciente = True

        e = self._medir(t, pessoa)
        self.hist.append(e)
        self.registro.append(e)
        while self.hist and t - self.hist[0].t > max(p["imobilidade_s"], p["agitacao_s"], 5.0) + 2:
            self.hist.popleft()
        recentes = [a for a in self.hist if a.visivel]

        # ---------------- queda ----------------
        fora_do_leito = e.no_leito is not True
        if e.deitado and fora_do_leito and not self._caido and self._queda_candidata is None:
            for a in recentes:
                if t - a.t <= p["queda_janela_s"] and a.ereto and e.centro[1] - a.centro[1] >= p["queda_descida_min"]:
                    self._queda_candidata = t
                    self._evid_queda = dict(angulo_antes=round(a.angulo), angulo_depois=round(e.angulo),
                                            descida_quadril=round(e.centro[1] - a.centro[1], 2),
                                            transicao_s=round(t - a.t, 2))
                    break
        if self._queda_candidata is not None:
            if not e.deitado:
                self._queda_candidata = None            # levantou em seguida: descarta
            elif t - self._queda_candidata >= 1.0:      # permaneceu no chão: confirma
                tq = self._queda_candidata
                saida.append(self._anomalia(
                    tq, "queda", Severidade.CRITICA,
                    f"Queda detectada: tronco passou de {self._evid_queda['angulo_antes']}° para "
                    f"{self._evid_queda['angulo_depois']}° com descida brusca do quadril em "
                    f"{self._evid_queda['transicao_s']} s, e o paciente permanece no chão.", **self._evid_queda))
                self._queda_candidata, self._caido = None, True
        if self._caido and not e.deitado:
            self._caido = False

        # ---------------- saída do leito ----------------
        if e.no_leito is not None:
            if self._condicao("saida_leito", self._esteve_no_leito and not e.no_leito, t, 1.0):
                saida.append(self._anomalia(t, "saida_do_leito", Severidade.MEDIA,
                                            "Paciente saiu da área do leito.", posicao=[round(c, 2) for c in e.centro]))
            self._esteve_no_leito |= bool(e.no_leito)

        # ---------------- zona restrita ----------------
        for z in self._zona("restrita"):
            dentro = z.contem(*e.pes) or z.contem(*e.centro)
            if self._condicao(f"zona:{z.nome}", dentro, t, p["zona_permanencia_s"]):
                saida.append(self._anomalia(t, "zona_restrita", Severidade.ALTA,
                                            f"Paciente na área crítica “{z.nome}” há {p['zona_permanencia_s']:.0f} s.",
                                            zona=z.nome))

        # ---------------- agitação (membros se movem muito, corpo não se desloca) ----------------
        jan = [a for a in recentes if t - a.t <= p["agitacao_s"]]
        coberta = bool(jan) and t - jan[0].t >= 0.9 * p["agitacao_s"]
        agitado = [a.mov_rel > p["agitacao_mov_min"] and a.vel_centro < 0.08 for a in jan]
        if self._condicao("agitacao", coberta and np.mean(agitado) >= 0.75, t, 0.0):
            saida.append(self._anomalia(t, "agitacao", Severidade.MEDIA,
                                        f"Agitação motora sustentada por {p['agitacao_s']:.0f} s "
                                        f"(movimento de membros {np.mean([a.mov_rel for a in jan]):.2f} diag/s).",
                                        mov_rel=round(float(np.mean([a.mov_rel for a in jan])), 3)))

        # ---------------- imobilidade ----------------
        jan = [a for a in recentes if t - a.t <= p["imobilidade_s"]]
        coberta = bool(jan) and t - jan[0].t >= 0.95 * p["imobilidade_s"]
        movs = np.array([a.mov for a in jan], dtype=float)
        parado = coberta and np.nanmax(movs) < p["imobilidade_mov_max"] if np.isfinite(movs).any() else False
        if self._condicao("imobilidade", bool(parado), t, 0.0):
            no_chao = e.deitado and e.no_leito is False
            saida.append(self._anomalia(
                t, "imobilidade_prolongada", Severidade.ALTA if no_chao else Severidade.BAIXA,
                f"Paciente imóvel há {p['imobilidade_s']:.0f} s" + (" deitado fora do leito." if no_chao else "."),
                segundos=p["imobilidade_s"], deitado=e.deitado, no_leito=e.no_leito))

        # ---------------- postura inclinada sustentada ----------------
        inclinado = p["inclinacao_angulo"] <= e.angulo <= p["queda_angulo_tronco"] and e.no_leito is not True
        if self._condicao("inclinado", inclinado, t, p["inclinacao_s"]):
            saida.append(self._anomalia(t, "postura_inclinada", Severidade.BAIXA,
                                        f"Tronco inclinado (~{e.angulo:.0f}°) há {p['inclinacao_s']:.0f} s — "
                                        "risco de escorregar/cair.", angulo=round(e.angulo)))

        # ---------------- objeto perigoso ao alcance ----------------
        x1, y1, x2, y2 = pessoa.bbox
        margem = 0.15 * max(x2 - x1, y2 - y1)
        perto = [o for o in (objetos or []) if o.classe in CLASSES_PERIGOSAS
                 and x1 - margem <= (o.bbox[0] + o.bbox[2]) / 2 <= x2 + margem
                 and y1 - margem <= (o.bbox[1] + o.bbox[3]) / 2 <= y2 + margem]
        if self._condicao("objeto_perigoso", bool(perto), t, 1.0):
            saida.append(self._anomalia(t, "objeto_perigoso", Severidade.ALTA,
                                        f"Objeto perfurocortante ({perto[0].nome}) ao alcance do paciente.",
                                        objeto=perto[0].nome, confianca=round(perto[0].conf, 2)))
        return saida

    # ------------------------------------------------------------------ agregação
    def serie_movimentacao(self, intervalo_s: float = 900.0) -> list[dict]:
        """Agrega o vídeo em intervalos para alimentar `series.movimentacao` (internação)."""
        linhas = []
        if not self.registro:
            return linhas
        t0, t_fim = self.registro[0].t, self.registro[-1].t
        while t0 <= t_fim:
            bloco = [e for e in self.registro if t0 <= e.t < t0 + intervalo_s]
            vis = [e for e in bloco if e.visivel]
            if bloco:
                dt = (bloco[-1].t - bloco[0].t) / max(len(bloco) - 1, 1)
                movs = np.array([e.mov for e in vis], dtype=float)
                trocas = sum(1 for a, b in zip(vis, vis[1:]) if a.deitado != b.deitado or abs(a.angulo - b.angulo) > 30)
                linhas.append({
                    "timestamp": self.inicio + timedelta(seconds=t0),
                    "atividade": float(np.nanmean(movs > 0.02)) if np.isfinite(movs).any() else 0.0,
                    "fora_do_leito_min": sum(dt for e in bloco if not e.visivel or e.no_leito is False) / 60,
                    "mudancas_posicao": trocas,
                })
            t0 += intervalo_s
        return linhas

    def resumo(self) -> dict:
        vis = [e for e in self.registro if e.visivel]
        movs = np.array([e.mov for e in vis], dtype=float)
        return {
            "quadros_analisados": len(self.registro),
            "paciente_visivel_frac": round(len(vis) / max(len(self.registro), 1), 3),
            "deitado_frac": round(float(np.mean([e.deitado for e in vis])), 3) if vis else 0.0,
            "no_leito_frac": (round(float(np.mean([bool(e.no_leito) for e in vis])), 3)
                              if vis and vis[0].no_leito is not None else None),
            "movimento_medio": round(float(np.nanmean(movs)), 4) if np.isfinite(movs).any() else 0.0,
        }
