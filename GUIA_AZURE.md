# Guia: como testar o projeto com a Azure (≈15 minutos, sem custo)

O projeto usa dois serviços da Azure nos áudios de consulta:

| Serviço | Para quê | Recurso a criar no portal | Plano gratuito |
|---|---|---|---|
| **Azure Speech to Text** | transcrever o áudio | **Speech** (Fala) | **F0** |
| **Azure Text Analytics** | sentimento, frases-chave, entidades de saúde | **Language** (Linguagem) | **F0** |

Os planos F0 são gratuitos e bastam com folga para este trabalho (na data deste guia: cerca de 5 horas de
áudio e 5.000 registros de texto por mês — confira os valores atuais na página de preços da Azure). Só é
permitido um recurso F0 de cada tipo por assinatura.

## 1. Conta na Azure

- **Estudante:** <https://azure.microsoft.com/free/students> — entra com o e-mail da instituição, ganha
  crédito e **não pede cartão de crédito**.
- **Conta gratuita comum:** <https://azure.microsoft.com/free> — pede cartão para verificação, mas os
  planos F0 não geram cobrança.

## 2. Criar o recurso Speech

1. Em <https://portal.azure.com>, clique em **Criar um recurso** e procure por **Speech** (Fala).
2. Preencha: assinatura; grupo de recursos (crie um, ex.: `tech4`); **Região: Brazil South**;
   nome qualquer (ex.: `tech4-speech`); **Tipo de preço: Free F0**.
3. **Examinar + criar** → **Criar**. Quando terminar, **Ir para o recurso**.
4. No menu à esquerda, **Chaves e Ponto de Extremidade**. Anote a **CHAVE 1** e o **Local/Região**
   (ex.: `brazilsouth`).

## 3. Criar o recurso Language

1. **Criar um recurso** → procure por **Language** (Serviço de linguagem). Na tela de recursos extras,
   apenas **Continuar para criar o recurso** (não precisa marcar nada).
2. Mesmo grupo de recursos; **Região: Brazil South** (se o F0 não estiver disponível nela, use East US);
   nome (ex.: `tech4-language`); **Tipo de preço: Free F0**; marque o aviso de IA responsável.
3. Crie, vá para o recurso e abra **Chaves e Ponto de Extremidade**. Anote a **CHAVE 1** e o
   **Ponto de extremidade** (algo como `https://tech4-language.cognitiveservices.azure.com/`).

## 4. Colocar as chaves no projeto

Na pasta do projeto, copie `.env.example` para `.env` e preencha (sem aspas, sem `#` na frente):

```
AZURE_SPEECH_KEY=cole-a-chave-do-recurso-speech
AZURE_SPEECH_REGION=brazilsouth
AZURE_LANGUAGE_KEY=cole-a-chave-do-recurso-language
AZURE_LANGUAGE_ENDPOINT=https://tech4-language.cognitiveservices.azure.com/
```

## 5. Testar

```powershell
python scripts/validar_azure.py --diagnostico   # confere as chaves e diz exatamente o que corrigir
python scripts/validar_azure.py                 # fala de teste -> transcrição -> sentimento -> alertas
```

O segundo comando grava `docs/evidencia_azure.md` com o resultado real — serve de evidência no trabalho.

Depois, no painel (`iniciar.bat` ou `uvicorn api:app --port 8000`), a caixa **Testar uma consulta** mostra
o estado das credenciais e permite **gravar pelo microfone**, **enviar um arquivo de áudio** (qualquer
formato que o navegador abra; ele é convertido para WAV) ou **colar um texto**. Se mudar o `.env`, reinicie
o servidor.

Frases boas para testar: *"Doutor, estou com dor no peito e muita falta de ar."* · *"Parei de tomar a
varfarina e hoje vomitei sangue."* · *"Não aguento mais, estou muito triste."* · *"Estou ótimo, sem dor e
sem febre."* (esta não deve gerar alerta).

## Erros comuns

| Mensagem do diagnóstico | Causa provável |
|---|---|
| Speech: HTTP 401 | chave errada, ou `AZURE_SPEECH_REGION` diferente da região do recurso |
| Speech: não conectou | região escrita por extenso (`Brazil South`) — use `brazilsouth` |
| Text Analytics: chave recusada (401) | chave de outro recurso (ex.: a do Speech) |
| Text Analytics: não conectou ao endpoint | endpoint incompleto; precisa começar com `https://` e terminar em `.cognitiveservices.azure.com/` |
| Entidades de saúde indisponíveis | plano/região sem *Text Analytics for health*: defina `AZURE_ENTIDADES_SAUDE=0` (o resto continua) |

## Como professores e colegas podem testar

1. **Com as chaves de vocês (mais simples para quem avalia).** Enviem a pasta do projeto e, em separado
   (e-mail ou mensagem privada), o arquivo `.env`. Quem recebe coloca o `.env` na pasta e dá dois cliques
   em `iniciar.bat`. Nunca publiquem o `.env` no GitHub (ele já está no `.gitignore`); depois da avaliação,
   gerem novas chaves no portal (**Chaves e Ponto de Extremidade → Regenerar**) para invalidar as antigas.
2. **Cada um com a própria conta.** Seguir este guia leva uns 15 minutos.
3. **Sem Azure.** O projeto roda inteiro sem chaves (vídeo, voz, sinais vitais, ECG, prescrições e termos
   críticos por léxico local); só ficam de fora a transcrição e o sentimento, e o painel avisa isso. Para
   esses casos, incluam `docs/evidencia_azure.md` na entrega.

`iniciar.bat` prepara tudo num computador novo com Windows e Python 3.11+: cria o ambiente, instala as
dependências, confere a Azure e abre o painel.
