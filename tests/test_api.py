"""Testes da API (pacientes de demonstração desligados; sem modelos de visão)."""
import io
import os
import wave
from datetime import datetime, timedelta

import numpy as np
import pytest

os.environ["CARREGAR_DEMO"] = "0"
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402


@pytest.fixture()
def cliente():
    api.assistente.alertas.limpar()
    api.assistente.pacientes.clear()
    api.assistente.alertas.canais = []          # não grava em outputs/ durante os testes
    with TestClient(api.app) as c:
        yield c


def test_fluxo_vitais_alerta_reconhecimento(cliente):
    t0 = datetime(2026, 11, 14, 8)
    leituras = [dict(timestamp=(t0 + timedelta(minutes=5 * i)).isoformat(), fc=76, pas=120, pad=78,
                     spo2=97 if i < 20 else 85, fr=16, temp=36.6) for i in range(26)]
    r = cliente.post("/pacientes/PX/sinais-vitais", json=leituras)
    assert r.status_code == 200 and any(a["tipo"] == "spo2_baixa" for a in r.json()["alertas_emitidos"])
    alertas = cliente.get("/alertas", params={"paciente_id": "PX", "severidade_minima": "critica"}).json()
    assert alertas and alertas[0]["severidade"] == "CRITICA"
    assert cliente.post(f"/alertas/{alertas[0]['id']}/reconhecer", json={"profissional": "Enf. Ana"}).json()["status"] == "reconhecido"
    assert cliente.get("/pacientes").json()[0]["id"] == "PX"
    assert "PX" in cliente.get("/pacientes/PX/resumo").json()["resumo"]


def test_prescricao_e_validacao(cliente):
    eventos = [dict(timestamp="2026-11-10T09:00:00", medicamento="morfina", acao="iniciar", dose_mg=5, intervalo_h=4),
               dict(timestamp="2026-11-10T21:00:00", medicamento="tramadol", acao="iniciar", dose_mg=100, intervalo_h=8)]
    r = cliente.post("/pacientes/PY/prescricoes", json=eventos).json()
    assert [a["tipo"] for a in r["alertas_emitidos"]] == ["interacao_medicamentosa"]
    assert "farmácia clínica" in r["alertas_emitidos"][0]["destinatarios"]
    assert cliente.post("/pacientes/PY/prescricoes", json=[{**eventos[0], "acao": "apagar"}]).status_code == 422
    assert cliente.get("/alertas", params={"severidade_minima": "xyz"}).status_code == 422
    assert cliente.post("/alertas/naoexiste/reconhecer", json={"profissional": "Ana"}).status_code == 404


def test_upload_de_audio(cliente):
    from src.audio.sintese import PERFIS, sintetizar_voz

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)  # noqa: E702
        w.writeframes((sintetizar_voz(**PERFIS["tremor"]) * 32767).astype("<i2").tobytes())
    r = cliente.post("/pacientes/PZ/audio", files={"arquivo": ("consulta.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200 and [a["tipo"] for a in r.json()["alertas_emitidos"]] == ["tremor_vocal"]
    assert 4.5 < r.json()["features"]["tremor_hz"] < 6.5
    assert cliente.post("/pacientes/PZ/audio", files={"arquivo": ("x.mp3", b"123", "audio/mpeg")}).status_code == 415


def test_perguntar_sem_credencial_devolve_resumo(cliente, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(api.assistente, "perguntar", lambda p, pid=None: api.assistente._sem_llm(pid, "teste"))
    cliente.post("/pacientes/PW/movimentacao", json=[dict(timestamp="2026-11-14T08:00:00", atividade=0.3, mudancas_posicao=2)])
    r = cliente.post("/perguntar", json={"pergunta": "Como está a unidade?"}).json()
    assert "PW" in r["resposta"]


def test_transcricao_termos_criticos(cliente):
    r = cliente.post("/pacientes/PT/transcricao", json={"texto": "O senhor tem falta de ar? Não. Mas estou com dor no peito."})
    corpo = r.json()
    assert r.status_code == 200 and [a["tipo"] for a in corpo["alertas_emitidos"]] == ["relato_dor_toracica"]
    assert "Text Analytics não configurado" in corpo["avisos"][0]
    assert set(cliente.get("/health").json()) >= {"azure_speech", "azure_text_analytics"}
