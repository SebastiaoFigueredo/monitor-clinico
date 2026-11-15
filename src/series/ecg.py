"""Detecção de arritmias perigosas no ECG (batimentos) com uma CNN 1D.

Treinada no "ECG Fragment Database for the Exploration of Dangerous Arrhythmia"
(PhysioNet): fragmentos de 2 s (720 amostras), agrupados em 6 classes de risco.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .. import config
from ..schemas import Anomalia, Severidade
from .detectores import episodios

# Os cabeçalhos .hea declaram 250 Hz, mas cada fragmento de "2 segundos" tem 721 amostras e o
# espectro que acompanha a base tem passo de 0,5 Hz (= janela de 2 s): a taxa efetiva é 360 Hz.
FS = 360
N = 2 * FS  # 720 amostras por janela de 2 s

CLASSES = [
    ("1_Dangerous_VFL_VF", "fibrilação/flutter ventricular", Severidade.CRITICA),
    ("2_Special_Form_VTTdP", "taquicardia ventricular polimórfica (torsades de pointes)", Severidade.CRITICA),
    ("3_Threatening_VT", "taquicardia ventricular de alto risco", Severidade.CRITICA),
    ("4_Potential_Dangerous", "arritmia ventricular potencialmente perigosa", Severidade.ALTA),
    ("5_Supraventricular", "arritmia supraventricular", Severidade.MEDIA),
    ("6_Sinus_rhythm", "ritmo sinusal", Severidade.INFO),
]
IDX_SINUSAL = 5


# ----------------------------------------------------------------------------- leitura
def ler_fragmento(base: Path) -> tuple[np.ndarray, int]:
    """Lê um registro WFDB de 1 canal em formato 16 (base = caminho sem extensão) -> (mV, fs)."""
    linhas = Path(f"{base}.hea").read_text().strip().splitlines()
    fs = int(float(linhas[0].split()[2]))
    campos = linhas[1].split()
    if campos[1] != "16":
        raise ValueError(f"formato WFDB não suportado: {campos[1]}")
    ganho = float(re.match(r"[\d.]+", campos[2]).group()) or 200.0
    adc_zero = int(campos[4]) if len(campos) > 4 else 0
    bruto = np.fromfile(f"{base}.dat", dtype="<i2").astype(np.float32)
    return (bruto - adc_zero) / ganho, fs


def preparar(sinal: np.ndarray) -> np.ndarray:
    """Recorta/completa para N amostras e normaliza (média 0, desvio 1)."""
    x = np.asarray(sinal, dtype=np.float32)[:N]
    if len(x) < N:
        x = np.pad(x, (0, N - len(x)), mode="edge")
    x = x - x.mean()
    return x / (x.std() + 1e-6)


def carregar_dataset(raiz: Path = config.ECG_DATASET) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X [n, N], y [n], registro [n]) — `registro` identifica o paciente/gravação de origem."""
    X, y, reg = [], [], []
    for idx, (pasta, _, _) in enumerate(CLASSES):
        for hea in sorted((raiz / pasta / "frag").glob("*.hea")):
            sinal, _ = ler_fragmento(hea.with_suffix(""))
            X.append(preparar(sinal))
            y.append(idx)
            reg.append("_".join(hea.stem.split("_")[:2]))
    return np.stack(X), np.array(y), np.array(reg)


# ----------------------------------------------------------------------------- modelo
def _criar_modelo():
    import torch.nn as nn

    def bloco(c_in, c_out, k):
        return nn.Sequential(nn.Conv1d(c_in, c_out, k, stride=2, padding=k // 2),
                             nn.BatchNorm1d(c_out), nn.ReLU())

    return nn.Sequential(
        bloco(1, 16, 9), bloco(16, 32, 7), bloco(32, 64, 5), bloco(64, 96, 3),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, len(CLASSES)),
    )


_cache: dict = {}


def carregar_modelo(caminho: Path = config.ECG_MODELO):
    import torch

    if "modelo" not in _cache:
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo de ECG não encontrado em {caminho}. Rode: python scripts/treinar_ecg.py")
        modelo = _criar_modelo()
        modelo.load_state_dict(torch.load(caminho, map_location="cpu", weights_only=True))
        _cache["modelo"] = modelo.eval()
    return _cache["modelo"]


def prever(janelas: np.ndarray) -> np.ndarray:
    """Probabilidades [n, 6] para janelas já preparadas [n, N]."""
    import torch

    with torch.no_grad():
        logits = carregar_modelo()(torch.from_numpy(np.asarray(janelas, dtype=np.float32)).unsqueeze(1))
        return torch.softmax(logits, dim=1).numpy()


# ----------------------------------------------------------------------------- análise
def analisar_ecg(sinal_mv: np.ndarray, fs: float, paciente_id: str, inicio: datetime | None = None,
                 prob_min: float = 0.6) -> list[Anomalia]:
    """Varre o traçado em janelas de 2 s (passo de 1 s) e devolve episódios de arritmia."""
    inicio = inicio or datetime.now()
    x = np.asarray(sinal_mv, dtype=np.float32)
    if fs != FS:  # reamostragem linear para 360 Hz
        t_novo = np.arange(0, len(x) / fs, 1 / FS)
        x = np.interp(t_novo, np.arange(len(x)) / fs, x).astype(np.float32)
    if len(x) < N:
        x = np.pad(x, (0, N - len(x)), mode="edge")
    passos = list(range(0, len(x) - N + 1, FS)) or [0]
    probs = prever(np.stack([preparar(x[p: p + N]) for p in passos]))
    # Um episódio é um trecho contínuo "não sinusal". As classes ventriculares se confundem entre si
    # janela a janela, então a gravidade vem do GRUPO de risco dominante no episódio, não da classe.
    p_anormal = 1 - probs[:, IDX_SINUSAL]
    minimo = 1 if len(passos) < 3 else 2  # exige 2 janelas seguidas em traçados longos (anti-artefato)
    grupos = [("ritmo ventricular grave", [0, 1, 2], Severidade.CRITICA),
              ("arritmia ventricular potencialmente perigosa", [3], Severidade.ALTA),
              ("arritmia supraventricular", [4], Severidade.MEDIA)]

    anomalias = []
    for ini, fim in episodios(p_anormal >= prob_min, minimo=minimo):
        media = probs[ini:fim].mean(axis=0)
        nome_grupo, _, sev = max(grupos, key=lambda g: media[g[1]].sum())
        classe = CLASSES[int(media[:IDX_SINUSAL].argmax())][1]
        dur = (passos[fim - 1] - passos[ini]) / FS + 2
        anomalias.append(Anomalia(
            paciente_id, "ecg", "arritmia", sev,
            f"ECG anormal por ~{dur:.0f} s: {nome_grupo} (padrão mais provável: {classe}; "
            f"probabilidade de ritmo não sinusal {p_anormal[ini:fim].mean():.0%}).",
            inicio + timedelta(seconds=passos[ini] / FS),
            {"grupo": nome_grupo, "classe_mais_provavel": classe, "duracao_s": dur,
             "p_nao_sinusal": round(float(p_anormal[ini:fim].mean()), 3),
             "prob_por_classe": {CLASSES[i][1]: round(float(m), 3) for i, m in enumerate(media)}},
            "CNN 1D (ECG Fragment Database, PhysioNet)",
        ))
    return anomalias


# ----------------------------------------------------------------------------- treino
def _ajustar(X: np.ndarray, y: np.ndarray, epocas: int, semente: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.from_numpy(X).unsqueeze(1).to(dev)
    ytr = torch.from_numpy(y).to(dev)
    modelo = _criar_modelo().to(dev)
    contagem = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    pesos = torch.tensor(contagem.sum() / (len(CLASSES) * np.maximum(contagem, 1)), device=dev)
    perda_fn = nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.05)
    opt = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=epocas * (len(Xtr) // 64 + 1))
    for _ in range(epocas):
        modelo.train()
        for lote in torch.randperm(len(Xtr), device=dev).split(64):
            xb = Xtr[lote]
            # aumento de dados: deslocamento circular, inversão de polaridade, ganho e ruído
            xb = torch.roll(xb, int(rng.integers(0, N)), dims=2)
            sinal = torch.where(torch.rand(len(xb), 1, 1, device=dev) < 0.3, -1.0, 1.0)
            xb = xb * sinal * (0.8 + 0.4 * torch.rand(len(xb), 1, 1, device=dev)) + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            perda_fn(modelo(xb), ytr[lote]).backward()
            opt.step()
            sched.step()
    return modelo.eval()


# nível de risco usado nos alertas: 0 = ventricular grave, 1 = potencialmente perigosa,
# 2 = supraventricular, 3 = sinusal
GRUPO_RISCO = np.array([0, 0, 0, 1, 2, 3])


def treinar(epocas: int = 80, semente: int = 0, dobras: int = 5, salvar: Path = config.ECG_MODELO) -> dict:
    """Avalia por validação cruzada agrupada por registro (nenhum paciente aparece em treino
    e teste ao mesmo tempo) e depois treina o modelo final com todos os fragmentos."""
    import torch
    from sklearn.model_selection import StratifiedGroupKFold

    X, y, reg = carregar_dataset()
    k = len(CLASSES)
    matriz = np.zeros((k, k), dtype=int)
    cv = StratifiedGroupKFold(n_splits=dobras, shuffle=True, random_state=semente)
    for i, (tr, te) in enumerate(cv.split(X, y, groups=reg)):
        modelo = _ajustar(X[tr], y[tr], epocas, semente + i)
        with torch.no_grad():
            dev = next(modelo.parameters()).device
            pred = modelo(torch.from_numpy(X[te]).unsqueeze(1).to(dev)).argmax(1).cpu().numpy()
        np.add.at(matriz, (y[te], pred), 1)

    risco = np.zeros((4, 4), dtype=int)
    for a in range(k):
        for b in range(k):
            risco[GRUPO_RISCO[a], GRUPO_RISCO[b]] += matriz[a, b]
    vent_real, vent_pred = matriz[:4].sum(), matriz[:, :4].sum()
    metricas = {
        "avaliacao": f"validação cruzada {dobras}x agrupada por registro",
        "n_fragmentos": int(len(y)), "n_registros": int(len(set(reg))),
        "acuracia_6_classes": float(np.trace(matriz) / matriz.sum()),
        "acuracia_nivel_de_risco": float(np.trace(risco) / risco.sum()),
        "recall_por_classe": {CLASSES[c][1]: float(matriz[c, c] / max(matriz[c].sum(), 1)) for c in range(k)},
        "sensibilidade_arritmia_ventricular": float(matriz[:4, :4].sum() / max(vent_real, 1)),
        "especificidade_arritmia_ventricular": float(matriz[4:, 4:].sum() / max(matriz[4:].sum(), 1)),
        "vpp_arritmia_ventricular": float(matriz[:4, :4].sum() / max(vent_pred, 1)),
        "sensibilidade_ritmo_critico": float(matriz[:3, :3].sum() / max(matriz[:3].sum(), 1)),
        "matriz_confusao": matriz.tolist(),
    }
    final = _ajustar(X, y, epocas, semente)
    salvar.parent.mkdir(exist_ok=True)
    torch.save(final.cpu().state_dict(), salvar)
    _cache.clear()
    return metricas
