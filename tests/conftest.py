import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest


@pytest.fixture(autouse=True)
def _sem_azure_real(monkeypatch):
    """Os testes nunca chamam a Azure de verdade, mesmo que exista um .env com chaves."""
    from src import config

    for nome in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"):
        monkeypatch.setattr(config, nome, "")
