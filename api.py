"""API HTTP do Monitor Clínico (FastAPI) + painel web em /.

    uvicorn api:app --port 8000
"""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import config
from src.assistente import AVISO, AssistenteClinico
from src.schemas import Alerta, Severidade

assistente = AssistenteClinico(silencioso=True)
ESTATICO = Path(__file__).parent / "static"


@asynccontextmanager
async def _ciclo(_: FastAPI):
    """Pré-carrega os pacientes fictícios (desligue com CARREGAR_DEMO=0)."""
    if os.getenv("CARREGAR_DEMO", "1") == "1":
        for pasta in sorted((config.DADOS / "pacientes").glob("*")):
            if (pasta / "paciente.json").exists():
                await run_in_threadpool(assistente.analisar_pasta, pasta, False)
    yield


app = FastAPI(title="Monitor Clínico — Assistente Virtual", version="1.0", description=AVISO, lifespan=_ciclo)


def _saida(alertas: list[Alerta]) -> dict:
    return {"alertas_emitidos": [a.to_dict() for a in alertas], "total": len(alertas)}


# ----------------------------------------------------------------------------- modelos
class LeituraVital(BaseModel):
    timestamp: datetime
    fc: float | None = None
    pas: float | None = None
    pad: float | None = None
    spo2: float | None = None
    fr: float | None = None
    temp: float | None = None


class EventoPrescricao(BaseModel):
    timestamp: datetime
    medicamento: str
    acao: str = Field(pattern="^(iniciar|alterar|suspender)$")
    dose_mg: float = 0
    intervalo_h: float = Field(24, gt=0)
    via: str = ""
    prescritor: str = ""


class IntervaloMovimento(BaseModel):
    timestamp: datetime
    atividade: float = Field(ge=0, le=1)
    fora_do_leito_min: float = 0
    mudancas_posicao: int = 0


class TracadoECG(BaseModel):
    fs: float = Field(gt=0)
    sinal_mv: list[float] = Field(min_length=100)
    inicio: datetime | None = None


class TextoConsulta(BaseModel):
    texto: str = Field(min_length=3)
    quando: datetime | None = None


class Reconhecimento(BaseModel):
    profissional: str = Field(min_length=2)


class Pergunta(BaseModel):
    pergunta: str = Field(min_length=3)
    paciente_id: str | None = None


# ----------------------------------------------------------------------------- consulta
@app.get("/", include_in_schema=False)
def painel():
    return FileResponse(ESTATICO / "index.html")


@app.get("/health")
def health():
    from src.audio import azure_ia

    return {"status": "ok", "pacientes": len(assistente.pacientes), "alertas": len(assistente.alertas.alertas),
            "azure_speech": azure_ia.fala_configurada(), "azure_text_analytics": azure_ia.linguagem_configurada()}


@app.get("/pacientes")
def pacientes():
    return assistente.situacao()


@app.get("/pacientes/{paciente_id}/resumo")
def resumo(paciente_id: str):
    if paciente_id not in assistente.pacientes:
        raise HTTPException(404, "paciente não encontrado")
    return {"paciente_id": paciente_id, "resumo": assistente.resumo_paciente(paciente_id)}


@app.get("/alertas")
def alertas(paciente_id: str | None = None, status: str | None = None, severidade_minima: str = "BAIXA"):
    try:
        sev = Severidade[severidade_minima.upper()]
    except KeyError:
        raise HTTPException(422, f"severidade inválida; use uma de {[s.name for s in Severidade]}")
    return [a.to_dict() for a in assistente.alertas.listar(paciente_id, status, sev)]


@app.post("/alertas/{alerta_id}/reconhecer")
def reconhecer(alerta_id: str, corpo: Reconhecimento):
    alerta = assistente.alertas.reconhecer(alerta_id, corpo.profissional)
    if alerta is None:
        raise HTTPException(404, "alerta não encontrado")
    return alerta.to_dict()


@app.post("/perguntar")
async def perguntar(corpo: Pergunta):
    return {"resposta": await run_in_threadpool(assistente.perguntar, corpo.pergunta, corpo.paciente_id)}


@app.get("/azure/diagnostico")
async def azure_diagnostico():
    """Testa as credenciais Azure de verdade e explica o que falta configurar."""
    from src.audio import azure_ia

    return await run_in_threadpool(azure_ia.diagnostico)


# ----------------------------------------------------------------------------- ingestão
def _df(itens: list[BaseModel]) -> pd.DataFrame:
    if not itens:
        raise HTTPException(422, "lista vazia")
    return pd.DataFrame([i.model_dump() for i in itens])


@app.post("/pacientes/{paciente_id}/sinais-vitais")
def sinais_vitais(paciente_id: str, leituras: list[LeituraVital]):
    """Envie a série completa (ou a janela recente) de leituras; os detectores usam o histórico."""
    return _saida(assistente.analisar_sinais_vitais(paciente_id, _df(leituras).dropna(axis=1, how="all")))


@app.post("/pacientes/{paciente_id}/prescricoes")
def prescricoes(paciente_id: str, eventos: list[EventoPrescricao]):
    return _saida(assistente.analisar_prescricoes(paciente_id, _df(eventos)))


@app.post("/pacientes/{paciente_id}/movimentacao")
def movimentacao(paciente_id: str, intervalos: list[IntervaloMovimento]):
    return _saida(assistente.analisar_movimentacao(paciente_id, _df(intervalos)))


@app.post("/pacientes/{paciente_id}/ecg")
def ecg(paciente_id: str, tracado: TracadoECG):
    try:
        return _saida(assistente.analisar_ecg(paciente_id, np.array(tracado.sinal_mv, dtype=np.float32),
                                              tracado.fs, tracado.inicio))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


def _temporario(arquivo: UploadFile, extensoes: set[str]) -> Path:
    sufixo = Path(arquivo.filename or "").suffix.lower()
    if sufixo not in extensoes:
        raise HTTPException(415, f"formato não suportado ({sufixo or 'sem extensão'}); use {sorted(extensoes)}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as tmp:
        shutil.copyfileobj(arquivo.file, tmp)
    return Path(tmp.name)


@app.post("/pacientes/{paciente_id}/audio")
async def audio(paciente_id: str, arquivo: UploadFile = File(...)):
    """Áudio de consulta em WAV PCM: biomarcadores vocais + transcrição (Azure Speech to Text)
    + termos críticos e sentimento (Azure Text Analytics)."""
    caminho = _temporario(arquivo, {".wav"})
    try:
        emitidos = await run_in_threadpool(assistente.analisar_audio, paciente_id, caminho)
    except Exception as e:  # WAV corrompido/compactado
        raise HTTPException(422, f"não foi possível ler o áudio: {e}")
    finally:
        caminho.unlink(missing_ok=True)
    analise = assistente.pacientes[paciente_id]["analises"]["audio"]
    return {**_saida(emitidos), **{k: analise[k] for k in
                                   ("features", "transcricao", "transcricao_origem", "analise_texto", "avisos") if k in analise}}


@app.post("/pacientes/{paciente_id}/transcricao")
async def transcricao(paciente_id: str, corpo: TextoConsulta):
    """Texto de consulta já transcrito: termos críticos + sentimento (Azure Text Analytics)."""
    emitidos = await run_in_threadpool(assistente.analisar_transcricao, paciente_id, corpo.texto, corpo.quando)
    analise = assistente.pacientes[paciente_id]["analises"]["transcricao"]
    return {**_saida(emitidos), **{k: analise[k] for k in ("analise_texto", "avisos") if k in analise}}


@app.post("/pacientes/{paciente_id}/video")
async def video(paciente_id: str, arquivo: UploadFile = File(...), backend: str = config.POSE_BACKEND):
    """Vídeo clínico (mp4/avi/mov/mkv). Processamento síncrono: use trechos curtos."""
    if backend not in ("yolo", "openpose"):
        raise HTTPException(422, "backend deve ser 'yolo' ou 'openpose'")
    caminho = _temporario(arquivo, {".mp4", ".avi", ".mov", ".mkv"})
    saida = config.SAIDAS / f"{paciente_id}_{datetime.now():%Y%m%d_%H%M%S}_anotado.mp4"
    try:
        emitidos = await run_in_threadpool(
            lambda: assistente.analisar_video(paciente_id, caminho, saida_anotada=saida, backend=backend))
    except FileNotFoundError as e:
        raise HTTPException(422, str(e))
    finally:
        caminho.unlink(missing_ok=True)
    return {**_saida(emitidos), "resumo": assistente.pacientes[paciente_id]["analises"]["video"],
            "video_anotado": f"/saidas/{saida.name}"}


@app.get("/saidas/{nome}")
def baixar_saida(nome: str):
    caminho = (config.SAIDAS / nome).resolve()
    if caminho.parent != config.SAIDAS.resolve() or not caminho.is_file():
        raise HTTPException(404, "arquivo não encontrado")
    return FileResponse(caminho)
