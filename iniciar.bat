@echo off
rem Prepara e inicia o Monitor Clinico em um computador novo (Windows, Python 3.11+ instalado).
rem Uso: de dois cliques neste arquivo ou rode "iniciar.bat" no terminal, dentro da pasta do projeto.
chcp 65001 >nul
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [1/4] Criando ambiente Python em .venv ...
    py -3 -m venv .venv || python -m venv .venv || goto :erro
    echo [2/4] Instalando dependencias ^(alguns minutos na primeira vez^) ...
    .venv\Scripts\python.exe -m pip install --upgrade pip -q
    .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :erro
)
if not exist data\pacientes\P001\paciente.json (
    echo [3/4] Gerando pacientes ficticios de demonstracao ...
    .venv\Scripts\python.exe scripts\gerar_dados_sinteticos.py || goto :erro
)
if not exist .env (
    copy .env.example .env >nul
    echo.
    echo  ATENCAO: criei o arquivo .env. Para usar a Azure, abra-o e preencha as 4 linhas AZURE_...
    echo  ^(passo a passo em GUIA_AZURE.md^). Sem isso o sistema roda, mas sem transcricao nem sentimento.
    echo.
)
echo [4/4] Verificando credenciais Azure ...
.venv\Scripts\python.exe scripts\validar_azure.py --diagnostico
echo.
echo Abrindo o painel em http://localhost:8000  ^(Ctrl+C para encerrar^)
start "" http://localhost:8000
.venv\Scripts\python.exe -m uvicorn api:app --port 8000
goto :eof

:erro
echo.
echo Algo falhou. Confira se o Python 3.11 ou mais novo esta instalado ^(python.org^) e tente de novo.
pause
