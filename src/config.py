"""Configuração central do Monitor Clínico (caminhos, limiares e parâmetros)."""
from __future__ import annotations

import os
from pathlib import Path

try:  # .env é opcional
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data"
MODELOS = RAIZ / "models"
SAIDAS = RAIZ / "outputs"
for _d in (DADOS, MODELOS, SAIDAS):
    _d.mkdir(exist_ok=True)

# Dataset de ECG (PhysioNet "ECG Fragment Database for the Exploration of Dangerous Arrhythmia")
ECG_DATASET = Path(
    os.getenv(
        "ECG_DATASET",
        RAIZ.parent / "ecg-fragment-database-for-the-exploration-of-dangerous-arrhythmia-1.0.0",
    )
)
ECG_MODELO = MODELOS / "ecg_cnn.pt"

# ----------------------------------------------------------------------------- vídeo
YOLO_OBJETOS = os.getenv("YOLO_OBJETOS", str(MODELOS / "yolov8n.pt"))  # baixado automaticamente
YOLO_POSE = os.getenv("YOLO_POSE", str(MODELOS / "yolov8n-pose.pt"))
POSE_BACKEND = os.getenv("POSE_BACKEND", "yolo")  # "yolo" | "openpose"
OPENPOSE_PROTO = MODELOS / "openpose" / "pose_deploy_linevec.prototxt"
OPENPOSE_PESOS = MODELOS / "openpose" / "pose_iter_440000.caffemodel"
VIDEO_FPS_ANALISE = float(os.getenv("VIDEO_FPS_ANALISE", "5"))  # quadros/s efetivamente analisados

VIDEO = {
    "queda_angulo_tronco": 60.0,     # graus em relação à vertical para considerar "deitado"
    "queda_janela_s": 1.5,           # transição em pé -> deitado dentro desta janela = queda
    "queda_descida_min": 0.18,       # descida mínima do quadril (fração da altura do quadro)
    "imobilidade_s": 300.0,          # sem movimento por este tempo -> alerta
    "imobilidade_mov_max": 0.006,    # deslocamento médio por segundo (fração da diagonal)
    "agitacao_mov_min": 0.05,        # movimento dos membros relativo ao quadril (diagonais/s)
    "agitacao_s": 8.0,               # sustentada por este tempo
    "ausencia_s": 20.0,              # nenhum paciente visível por este tempo
    "zona_permanencia_s": 2.0,       # tempo dentro de zona restrita para alertar
    "inclinacao_angulo": 35.0,       # tronco inclinado (sentado/em pé) sustentado
    "inclinacao_s": 10.0,
}

# ----------------------------------------------------------------------------- áudio
AUDIO = {
    "f0_min": 60.0,
    "f0_max": 400.0,
    "jitter_max_pct": 1.04,      # limiar clássico (MDVP) de jitter local
    "shimmer_max_pct": 3.81,     # limiar clássico (MDVP) de shimmer local
    "hnr_min_db": 12.0,          # abaixo disto: voz soprosa/rouca
    "pausa_frac_max": 0.45,      # fração de silêncio dentro da fala
    "f0_cv_min": 0.035,          # coeficiente de variação de F0 abaixo disto: fala monótona
    "tremor_indice_max": 0.25,   # energia relativa de modulação 3-12 Hz em F0
    "tosse_por_min_max": 4.0,
    "desvio_baseline_z": 3.0,
}

# ----------------------------------------------------------------------------- sinais vitais
# (critico_baixo, alerta_baixo, alerta_alto, critico_alto) — adulto em enfermaria
LIMITES_VITAIS = {
    "fc": (40, 50, 110, 130),        # bpm
    "pas": (80, 90, 160, 200),       # mmHg
    "pad": (40, 50, 100, 120),       # mmHg
    "spo2": (88, 92, 101, 101),      # %
    "fr": (8, 11, 21, 25),           # irpm
    "temp": (35.0, 36.0, 38.0, 39.1),  # °C
}
NOMES_VITAIS = {
    "fc": "frequência cardíaca",
    "pas": "pressão arterial sistólica",
    "pad": "pressão arterial diastólica",
    "spo2": "saturação de O2",
    "fr": "frequência respiratória",
    "temp": "temperatura",
}
UNIDADES = {"fc": "bpm", "pas": "mmHg", "pad": "mmHg", "spo2": "%", "fr": "irpm", "temp": "°C"}

SERIES = {
    "janela_z": 36,            # amostras para mediana/MAD móveis
    "z_limite": 4.0,
    "cusum_k": 1.0,
    "cusum_h": 10.0,
    "iforest_frac_referencia": 0.5,  # fração inicial da série usada como "normal" do paciente
    "persistencia_min": 2,     # amostras consecutivas para confirmar anomalia por limite
}

# ----------------------------------------------------------------------------- alertas
ALERTA_COOLDOWN_S = int(os.getenv("ALERTA_COOLDOWN_S", "600"))
ALERTAS_ARQUIVO = SAIDAS / "alertas.jsonl"
WEBHOOK_URL = os.getenv("ALERTA_WEBHOOK_URL", "")

# ----------------------------------------------------------------------------- Azure (opcional)
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")          # ex.: brazilsouth
AZURE_SPEECH_IDIOMA = os.getenv("AZURE_SPEECH_IDIOMA", "pt-BR")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY", "")
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "")  # https://<recurso>.cognitiveservices.azure.com/
AZURE_ENTIDADES_SAUDE = os.getenv("AZURE_ENTIDADES_SAUDE", "1") == "1"  # Text Analytics for health

TEXTO = {
    "negativo_min": 0.70,          # confiança de sentimento negativo no documento para sinalizar
    "frase_negativa_min": 0.85,    # frase isolada fortemente negativa
    "piora_sentimento": 0.35,      # aumento do escore negativo vs. consultas anteriores
}

# ----------------------------------------------------------------------------- LLM (opcional)
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
