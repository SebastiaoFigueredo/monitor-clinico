"""Treina o classificador de arritmias (CNN 1D) e salva em models/ecg_cnn.pt."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.series.ecg import treinar  # noqa: E402

if __name__ == "__main__":
    metricas = treinar()
    (config.MODELOS / "ecg_metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))
