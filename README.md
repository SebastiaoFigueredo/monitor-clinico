# Monitor Clínico — Assistente Virtual de Monitoramento Hospitalar

Assistente que acompanha pacientes internados a partir de **vídeo**, **áudio** e **séries temporais**,
detecta o que foge do padrão esperado e **gera alertas automáticos para a equipe**.

> **Protótipo acadêmico (Tech Challenge 4).** Os alertas são achados de triagem para apoiar a equipe,
> não diagnósticos. Todos os pacientes de exemplo são fictícios; o único dado real é a base pública de
> ECG da PhysioNet. Não usar em assistência real sem validação clínica, regulatória e de privacidade (LGPD).

## O que ele faz

| Entrada | Técnica | Anomalias detectadas |
|---|---|---|
| **Vídeo clínico** | Pose humana com **YOLOv8-pose** ou **OpenPose** (modelo COCO da CMU via OpenCV DNN) + **YOLOv8** para objetos e áreas críticas | queda, saída do leito, permanência em zona restrita, agitação motora, imobilidade, postura inclinada, paciente ausente, objeto perfurocortante ao alcance |
| **Áudio de consulta — voz** | Biomarcadores vocais: F0, jitter, shimmer, HNR, pausas, tremor, tosse (NumPy puro) + comparação com a linha de base do próprio paciente | disfonia, tremor vocal, fala monótona, fala entrecortada/dispneia, tosse frequente, mudança vocal |
| **Áudio de consulta — fala** | Transcrição com **Azure Speech to Text**; **Azure Text Analytics** para sentimento, frases-chave e entidades de saúde; léxico de termos críticos | relato de dor torácica, dispneia, sangramento, sinal neurológico, síncope, risco de autolesão, problema com medicação etc.; sentimento negativo; piora do tom vs. consultas anteriores |
| **Sinais vitais** (FC, PA, SpO₂, FR, T) | limites clínicos com persistência · NEWS2 simplificado · z-score robusto móvel · CUSUM sobre resíduo sazonal · Isolation Forest | valor crítico, deterioração (escore), variação abrupta, tendência sustentada, combinação multivariada atípica |
| **ECG** (batimentos) | CNN 1D treinada na *ECG Fragment Database for the Exploration of Dangerous Arrhythmia* | ritmo ventricular grave (FV/TV/torsades), arritmia potencialmente perigosa, supraventricular |
| **Prescrições** | regras de segurança + estatística sobre o histórico | salto de dose (≥2x; ≥5x = provável erro de digitação), dose acima do máximo, suspensão precoce de antibiótico ou de uso contínuo, interação, duplicidade, prescrição instável, volume atípico de mudanças |
| **Movimentação na internação** | regras de duração + linha de base por paciente e por hora do dia | imobilidade prolongada (lesão por pressão), deambulação noturna, queda de atividade, agitação |

Toda anomalia vira um **alerta** com severidade, destinatários (enfermagem → médico plantonista → time de
resposta rápida; farmácia clínica para prescrições), ação sugerida e evidências. O gerenciador evita fadiga
de alarmes: deduplica por paciente/tipo dentro de uma janela (10 min por padrão), **escalona** se o quadro
piora e entrega em canais plugáveis (console, `outputs/alertas.jsonl`, webhook).

## Arquitetura

```
vídeo ──► pose (YOLOv8-pose | OpenPose) ─┐
          objetos/zonas (YOLOv8) ────────┼► video/analise.py (máquina de estados) ─┐
áudio ──► audio/features.py (voz) ─────────────► audio/analise.py ──────────────────┤
          azure_ia.py: Speech to Text ─► Text Analytics ─► texto.py (termos críticos) ┘
vitais ─► series/sinais_vitais.py   (detectores.py: z robusto, CUSUM, IForest) ────┤
ECG ────► series/ecg.py (CNN 1D) ───────────────────────────────────────────────────┼► Anomalia ─► alertas.py ─► equipe
prescr. ► series/prescricoes.py ────────────────────────────────────────────────────┤      (dedup, escalonamento, roteamento)
movim. ─► series/movimentacao.py ───────────────────────────────────────────────────┘
                         assistente.py orquestra tudo · api.py (FastAPI) · static/index.html (painel)
```

## Como rodar

**Jeito rápido (Windows):** dois cliques em `iniciar.bat` — cria o ambiente, instala tudo, confere as
credenciais Azure e abre o painel. Para configurar a Azure do zero, siga o [GUIA_AZURE.md](GUIA_AZURE.md).

Passo a passo manual:

```powershell
py -3.13 -m venv C:\venvs\tech4          # fora do OneDrive (caminhos longos / Smart App Control)
C:\venvs\tech4\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # GPU; sem GPU: pip install torch torchvision
pip install -r requirements.txt

python scripts/treinar_ecg.py              # ~1 min na GPU; grava models/ecg_cnn.pt e as métricas
python scripts/gerar_dados_sinteticos.py   # cria 4 pacientes fictícios em data/pacientes
python scripts/demo.py                     # analisa todos e imprime os alertas
pytest -q                                  # 32 testes

uvicorn api:app --port 8000                # painel em http://localhost:8000 · docs em /docs
```

Arquivos avulsos:

```powershell
python scripts/demo.py --video quarto.mp4 --paciente P010 --zonas data/zonas_exemplo.json
python scripts/demo.py --video quarto.mp4 --backend openpose     # antes: python scripts/baixar_openpose.py
python scripts/demo.py --audio consulta.wav --paciente P010
python scripts/demo.py --perguntar "Quem devo reavaliar primeiro?"   # requer ANTHROPIC_API_KEY
```

### Azure Speech to Text e Text Analytics

O passo a passo completo (conta gratuita, criação dos recursos, erros comuns e como compartilhar com quem
vai avaliar) está no [GUIA_AZURE.md](GUIA_AZURE.md). Em resumo: crie no portal da Azure um recurso **Speech** e um recurso **Language**, copie `.env.example` para `.env` e
preencha `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`, `AZURE_LANGUAGE_KEY` e `AZURE_LANGUAGE_ENDPOINT`. Depois:

```powershell
python scripts/validar_azure.py --diagnostico   # confere as chaves e diz o que corrigir
python scripts/validar_azure.py            # fala de teste (TTS) -> Speech to Text -> Text Analytics -> anomalias
python scripts/validar_azure.py --audio consulta.wav
python scripts/demo.py --audio consulta.wav --paciente P010
```

Fluxo de cada áudio (`src/audio/analise.py::analisar_audio`):

1. **Transcrição** — reconhecimento contínuo em pt-BR com tempo e confiança por trecho; os nomes de
   medicamentos entram como *phrase list*. Se existir um `.txt` com o mesmo nome do áudio, ele é usado no
   lugar (é assim que os pacientes fictícios funcionam, já que a voz sintética não é fala inteligível).
2. **Text Analytics** — sentimento do documento e por frase, frases-chave e, com `AZURE_ENTIDADES_SAUDE=1`,
   *Text Analytics for health* (sintomas e diagnósticos com detecção de negação). Textos longos são
   divididos em blocos de até 5.000 caracteres; as chamadas usam `disable_service_logs`.
3. **Termos críticos** — léxico em `src/audio/texto.py` (14 categorias com severidade e orientação) que
   ignora perguntas do profissional e negações; a negação apontada pela Azure tem prioridade.
4. **Alertas** — `relato_<categoria>`, `sentimento_negativo` (apenas quando a frase negativa tem conteúdo
   emocional) e `piora_do_sentimento` (comparação com as consultas anteriores do paciente).

Sem credenciais nada quebra: a análise acústica e o léxico local continuam, e a resposta traz um aviso
dizendo qual serviço não está configurado. Falha de rede/cota na Azure também vira aviso, não erro.

Os pesos `yolov8n.pt` e `yolov8n-pose.pt` são baixados automaticamente pelo Ultralytics para `models/`.
As zonas do quarto são polígonos em coordenadas normalizadas (`data/zonas_exemplo.json`); sem arquivo de
zonas, o leito é inferido da classe *bed* do YOLOv8.

### API

| Método | Rota | Uso |
|---|---|---|
| GET | `/pacientes` · `/pacientes/{id}/resumo` | situação da unidade (mais grave primeiro) e resumo de plantão |
| GET | `/alertas?paciente_id=&status=&severidade_minima=` | lista de alertas |
| POST | `/alertas/{id}/reconhecer` | profissional assume o alerta |
| POST | `/pacientes/{id}/sinais-vitais` · `/prescricoes` · `/movimentacao` · `/ecg` | séries em JSON |
| POST | `/pacientes/{id}/audio` · `/video` | upload de WAV / vídeo (o áudio devolve transcrição, sentimento e biomarcadores) |
| POST | `/pacientes/{id}/transcricao` | texto de consulta já transcrito: termos críticos + sentimento |
| POST | `/perguntar` | pergunta em linguagem natural sobre os alertas (Claude; sem credencial devolve o resumo automático) |

## Pacientes de demonstração e resultado esperado

| Paciente | Anomalias plantadas | Detectado |
|---|---|---|
| P001 | deterioração tipo sepse nas últimas 10 h, antibiótico suspenso em 2 dias, atividade diurna a 25%, voz dispneica | tendências (PA, FC, T) **horas antes** dos limites, NEWS2 5→12, limites críticos, suspensão precoce, queda de atividade, pausas/tosse |
| P002 | nenhuma (controle) | **nenhum alerta** |
| P003 | heparina 10x, AINE + anticoagulante, dessaturação de 30 min, 90 min fora do leito de madrugada, queda (vídeo com pixels reais), disfonia nova | todos |
| P004 | bradicardia progressiva, TV no ECG, haloperidol + amiodarona, dois opioides, 5,5 h imóvel, agitação (vídeo simulado), tremor vocal novo | todos |

## Validação e limitações (leia antes de confiar nos números)

- **ECG.** A base tem só 1.016 fragmentos de 2 s de **22 registros**. Em validação cruzada 5x *agrupada por
  registro* (nenhum paciente em treino e teste ao mesmo tempo): sensibilidade de 95% para arritmia
  ventricular e 95% para ritmo crítico, mas especificidade de **74%** e acurácia de 62% nas 6 classes —
  ritmo sinusal com bloqueio de ramo é confundido com TV. Serve como demonstração; para uso real precisaria
  de muito mais dados. As métricas ficam em `models/ecg_metricas.json`. Os cabeçalhos da base declaram
  250 Hz, mas os fragmentos têm 721 amostras para 2 s: o código usa 360 Hz. As tiras de ECG dos pacientes
  de demonstração são montadas com fragmentos que também estão no treino do modelo final.
- **Vídeo.** A lógica de eventos é testada com sequências sintéticas de pontos-chave (5 cenários) e o
  pipeline completo (YOLOv8-pose **e** OpenPose) com um vídeo de pixels reais em que a foto de uma pessoa é
  rotacionada para simular a queda. **Não foi validado com filmagens reais de quartos hospitalares**
  (oclusão por lençóis, câmera no teto, pouca luz e várias pessoas mudam bastante o resultado). O backend
  OpenPose decodifica uma pessoa só e roda em CPU (~1 s por quadro); o YOLOv8-pose é multi-pessoa e usa GPU.
- **Áudio.** Validado apenas com voz sintética paramétrica (`src/audio/sintese.py`), não com gravações de
  pacientes. Os limiares de jitter/shimmer/HNR são referências clássicas de análise acústica para vogal
  sustentada; em fala corrida real tendem a gerar mais falsos positivos.
- **Azure.** A integração com Speech to Text e Text Analytics foi executada contra a Azure real em
  15/11/2026 (recursos no plano gratuito F0): fala sintetizada pelo próprio Azure → transcrição → sentimento,
  frases-chave e entidades de saúde em português → alertas, tanto pelo script quanto pelo upload na API. O
  resultado está em `docs/evidencia_azure.md`. Só foi testada com fala sintética (TTS), não com gravações de
  consultas. **O sentimento da Azure sozinho não serve de alerta clínico:** ele classificou como negativas
  frases como "Não tive febre nem falta de ar" e "a dor diminuiu bastante"; por isso o alerta
  `sentimento_negativo` só dispara quando a frase negativa tem conteúdo emocional (lista de palavras em
  `src/audio/texto.py`). O léxico de termos críticos é pequeno e baseado em expressões regulares: não cobre
  sinônimos nem ironia, e a regra de negação é simples. Os testes automatizados usam clientes simulados e
  nunca chamam a nuvem. Usar a Azure envia áudio e texto da consulta para a nuvem: com pacientes reais isso
  exige contrato adequado, região definida e consentimento.
- **Prescrições.** A tabela de medicamentos, doses máximas e interações em `src/series/prescricoes.py` é
  ilustrativa e pequena; em produção viria do sistema de prescrição/farmácia.
- **Sinais vitais.** O NEWS2 é simplificado (sem nível de consciência nem O₂ suplementar). A detecção de
  tendência precisa de ~1,5 dia de histórico para descontar o ritmo circadiano.
- **Assistente em linguagem natural.** O caminho com Claude (`/perguntar`) foi implementado, mas só o
  comportamento sem credencial foi exercitado nos testes.
- **Privacidade.** Vídeo e áudio de pacientes são dados sensíveis: o protótipo processa tudo localmente,
  mas não implementa autenticação, criptografia nem trilha de auditoria.
