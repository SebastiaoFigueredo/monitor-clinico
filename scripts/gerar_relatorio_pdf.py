"""Gera docs/Relatorio_Monitor_Clinico.pdf — relatório resumido do projeto (requer: pip install reportlab)."""
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
SAIDA = RAIZ / "docs" / "Relatorio_Monitor_Clinico.pdf"
AZUL, CINZA = colors.HexColor("#0b4f8a"), colors.HexColor("#5d6b7c")

# Arial do Windows (acentos e símbolos); sem ela, cai para Helvetica
FONTE, NEGRITO, ITALICO = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
try:
    for nome, arq in (("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf")):
        pdfmetrics.registerFont(TTFont(nome, rf"C:\Windows\Fonts\{arq}"))
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
    FONTE, NEGRITO, ITALICO = "Arial", "Arial-Bold", "Arial-Italic"
except Exception:
    pass

base = getSampleStyleSheet()
P = ParagraphStyle("p", parent=base["Normal"], fontName=FONTE, fontSize=9.6, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=P, fontName=NEGRITO, fontSize=13, leading=16, textColor=AZUL, spaceBefore=10, spaceAfter=5, alignment=0, keepWithNext=1)
ITEM = ParagraphStyle("item", parent=P, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
CEL = ParagraphStyle("cel", parent=P, fontSize=8.4, leading=10.6, alignment=0, spaceAfter=0)
CELB = ParagraphStyle("celb", parent=CEL, fontName=NEGRITO, textColor=colors.white)
REF = ParagraphStyle("ref", parent=P, fontSize=8.5, leading=11.2, leftIndent=16, firstLineIndent=-16, alignment=0, spaceAfter=3)
LEG = ParagraphStyle("leg", parent=P, fontName=ITALICO, fontSize=8.2, textColor=CINZA, alignment=1)


def tabela(linhas, larguras):
    dados = [[Paragraph(c, CELB if i == 0 else CEL) for c in linha] for i, linha in enumerate(linhas)]
    t = Table(dados, colWidths=larguras, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3dd")),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return t


def itens(*textos):
    return [Paragraph(t, ITEM, bulletText="•") for t in textos]


def quadro_do_video() -> Path | None:
    """Extrai um quadro do vídeo anotado da queda para ilustrar o relatório."""
    video = RAIZ / "outputs" / "validacao_queda_anotado.mp4"
    destino = RAIZ / "docs" / "quadro_queda.jpg"
    if destino.exists():
        return destino
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.75))
        ok, quadro = cap.read()
        if ok:
            destino.parent.mkdir(exist_ok=True)
            destino.write_bytes(cv2.imencode(".jpg", quadro)[1].tobytes())  # imwrite falha com acento no caminho
            return destino
    except Exception:
        pass
    return None


def rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONTE, 7.8)
    canvas.setFillColor(CINZA)
    canvas.drawString(2 * cm, 1.15 * cm, "Monitor Clínico — Tech Challenge 4 · protótipo acadêmico, não usar em assistência real")
    canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, f"{doc.page}")
    canvas.restoreState()


def main():
    m = json.loads((RAIZ / "models" / "ecg_metricas.json").read_text(encoding="utf-8"))
    pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    h = []

    h += [Paragraph("Monitor Clínico", ParagraphStyle("t", parent=H1, fontSize=22, leading=26, spaceBefore=0)),
          Paragraph("Assistente virtual de monitoramento hospitalar por vídeo, áudio e séries temporais",
                    ParagraphStyle("st", parent=P, fontSize=11.5, leading=15, textColor=CINZA, alignment=0)),
          Paragraph("Relatório resumido · Tech Challenge 4 · Sebastião Figueredo dos Santos · novembro de 2026",
                    ParagraphStyle("au", parent=P, fontSize=9, textColor=CINZA, alignment=0, spaceAfter=10))]

    h.append(Paragraph("1. Objetivo", H1))
    h.append(Paragraph(
        "O Monitor Clínico é um protótipo de assistente virtual para unidades de internação. Ele processa vídeos "
        "clínicos, áudios de consultas e séries temporais do paciente, detecta o que foge do padrão esperado e gera "
        "alertas automáticos para a equipe, com severidade, destinatários, ação sugerida e as evidências do achado. "
        "É um trabalho acadêmico: os alertas são achados de triagem, não diagnósticos, e todos os pacientes de "
        "demonstração são fictícios.", P))

    h.append(Paragraph("2. O que o sistema faz", H1))
    h.append(tabela([
        ["Entrada", "Técnica", "Anomalias detectadas"],
        ["Vídeo clínico", "Pose humana com YOLOv8-pose ou OpenPose; objetos e áreas críticas com YOLOv8; máquina de estados sobre os pontos-chave",
         "Queda, saída do leito, zona restrita, agitação, imobilidade, paciente ausente, objeto perfurocortante"],
        ["Áudio — voz", "F0, jitter, shimmer, HNR, pausas, tremor e tosse; comparação com consultas anteriores do paciente",
         "Disfonia, tremor vocal, fala monótona, fala entrecortada, mudança vocal"],
        ["Áudio — fala", "Azure Speech to Text (transcrição); Azure Text Analytics (sentimento, frases-chave, entidades de saúde); léxico de termos críticos",
         "Relato de dor torácica, dispneia, sangramento, risco de autolesão, problema com medicação; sofrimento emocional"],
        ["Sinais vitais", "Limites clínicos com persistência, NEWS2 simplificado, z-score robusto, CUSUM sobre resíduo sazonal, Isolation Forest",
         "Valor crítico, deterioração, variação abrupta, tendência sustentada, combinação atípica"],
        ["ECG", "Rede neural convolucional 1D treinada em base pública de arritmias", "Ritmo ventricular grave, arritmia potencialmente perigosa, supraventricular"],
        ["Prescrições", "Regras de segurança e estatística sobre o histórico", "Salto de dose, dose acima do máximo, suspensão precoce, interação, duplicidade"],
        ["Movimentação", "Regras de duração e linha de base por paciente e hora do dia", "Imobilidade prolongada, deambulação noturna, queda de atividade, agitação"],
    ], [2.5 * cm, 8.0 * cm, 6.5 * cm]))
    h.append(Spacer(1, 6))
    h.append(Paragraph(
        "Toda anomalia é convertida em alerta por um gerenciador que roteia por severidade (enfermagem, médico "
        "plantonista, time de resposta rápida; farmácia clínica para prescrições), agrupa repetições numa janela "
        "de 10 minutos para reduzir fadiga de alarmes e renotifica quando o quadro piora. Os alertas ficam "
        "disponíveis em uma API (FastAPI) e em um painel web, onde podem ser reconhecidos pelo profissional.", P))

    h.append(Paragraph("3. Como o Monitor Clínico foi criado", H1))
    h.append(Paragraph(
        "O projeto foi desenvolvido em Python entre 14 e 15 de novembro de 2026, de forma iterativa, com apoio do "
        "assistente de programação Claude Code (Anthropic): os requisitos foram passados em linguagem natural e o "
        "código, os testes e a documentação foram escritos, executados e corrigidos em ciclos curtos. As etapas:", P))
    h += itens(
        "<b>Ambiente e base.</b> Ambiente virtual com PyTorch (GPU), Ultralytics e OpenCV; definição das estruturas "
        "comuns (anomalia, alerta, severidade) e do gerenciador de alertas.",
        "<b>Séries temporais.</b> Detectores genéricos (z-score robusto, CUSUM, Isolation Forest) e módulos de sinais "
        "vitais, prescrições e movimentação. A CNN de ECG foi treinada na base pública da PhysioNet.",
        "<b>Áudio.</b> Extração de biomarcadores vocais apenas com NumPy e um sintetizador de voz paramétrico para "
        "gerar áudios de teste com alterações controladas.",
        "<b>Vídeo.</b> Dois backends de pose (YOLOv8-pose e OpenPose via OpenCV DNN), detecção de objetos e zonas com "
        "YOLOv8 e a lógica de eventos, testada com cenários simulados e com um vídeo de validação.",
        "<b>Integração.</b> Orquestrador, quatro pacientes fictícios com anomalias plantadas, API, painel e testes.",
        "<b>Nuvem.</b> Criação dos recursos Speech e Language na Azure (plano gratuito F0) e integração com "
        "Azure Speech to Text e Azure Text Analytics, validada com chamadas reais.")
    h.append(Paragraph(
        "Rodar o sistema de verdade revelou problemas que foram corrigidos: o detector de tendência disparava com o "
        "ritmo circadiano normal e passou a comparar com o mesmo horário dos dias anteriores; a detecção de arritmia "
        "passou a agregar por grupo de risco; o OpenCV 5 deixou de ler modelos Caffe e foi fixada a versão 4; e o "
        "sentimento da Azure classificava como negativas frases com termos médicos, como “Não tive febre”, de modo "
        "que o alerta passou a exigir conteúdo emocional.", P))

    img = quadro_do_video()
    if img:
        h.append(KeepTogether([Image(str(img), width=9.2 * cm, height=6.9 * cm),
                               Paragraph("Figura 1 — Quadro do vídeo de validação: pontos-chave do YOLOv8-pose e alerta de queda.", LEG)]))

    h.append(Paragraph("4. Resultados", H1))
    h.append(tabela([
        ["Paciente fictício", "Anomalias plantadas", "Resultado"],
        ["P001", "Deterioração tipo sepse, antibiótico suspenso em 2 dias, queda de atividade, voz dispneica",
         "Tendências de pressão, frequência cardíaca e temperatura horas antes dos limites críticos; NEWS2 de 5 a 12"],
        ["P002 (controle)", "Nenhuma", "Nenhum alerta"],
        ["P003", "Heparina 10x, anti-inflamatório com anticoagulante, dessaturação, deambulação noturna, queda em vídeo", "Todas detectadas"],
        ["P004", "Bradicardia progressiva, taquicardia ventricular, interações, 5,5 h imóvel, tremor vocal novo", "Todas detectadas"],
    ], [3.0 * cm, 7.6 * cm, 6.4 * cm]))
    h.append(Spacer(1, 6))
    h += itens(
        f"<b>ECG.</b> Validação cruzada em 5 partes separando pacientes ({m['n_fragmentos']} fragmentos de "
        f"{m['n_registros']} registros): sensibilidade de {pct(m['sensibilidade_arritmia_ventricular'])} para arritmia "
        f"ventricular, especificidade de {pct(m['especificidade_arritmia_ventricular'])} e acurácia de "
        f"{pct(m['acuracia_6_classes'])} nas 6 classes.",
        "<b>Azure.</b> Fala de teste transcrita com confiança entre 0,88 e 0,99 por trecho; sentimento, frases-chave e "
        "entidades de saúde em português, com negação (“febre” negada), gerando os alertas esperados.",
        "<b>Testes.</b> 32 testes automatizados passam; eles não dependem de GPU nem de nuvem.")

    h.append(Paragraph("5. Limitações", H1))
    h += itens(
        "A base de ECG tem só 22 registros; a especificidade de 74% indica muitos falsos alarmes fora da demonstração.",
        "O vídeo não foi validado com filmagens reais de quartos hospitalares: a queda foi testada com a foto de uma "
        "pessoa rotacionada e os demais eventos com esqueletos simulados.",
        "A análise acústica foi validada com voz sintética; a integração Azure, só com fala sintetizada (TTS).",
        "A tabela de medicamentos e o léxico de termos críticos são pequenos e ilustrativos.",
        "Não há autenticação, criptografia nem trilha de auditoria; com pacientes reais, o envio de áudio e texto à "
        "nuvem exigiria contrato, consentimento e adequação à LGPD.")

    h.append(Paragraph("6. Fontes da pesquisa", H1))
    refs = [
        "JOCHER, G.; CHAURASIA, A.; QIU, J. <i>Ultralytics YOLOv8</i>. 2023. Disponível em: https://github.com/ultralytics/ultralytics.",
        "CAO, Z.; HIDALGO, G.; SIMON, T.; WEI, S.-E.; SHEIKH, Y. OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. "
        "<i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, v. 43, n. 1, p. 172–186, 2021.",
        "LIN, T.-Y. et al. Microsoft COCO: Common Objects in Context. In: <i>European Conference on Computer Vision (ECCV)</i>, 2014.",
        "NEMIRKO, A.; MANILO, L.; TATARINOVA, A.; ALEKSEEV, B.; EVDAKOVA, E. <i>ECG Fragment Database for the Exploration of Dangerous "
        "Arrhythmia</i> (versão 1.0.0). PhysioNet, 2022. https://doi.org/10.13026/kpfg-xs25.",
        "GOLDBERGER, A. L. et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic "
        "signals. <i>Circulation</i>, v. 101, n. 23, p. e215–e220, 2000.",
        "ROYAL COLLEGE OF PHYSICIANS. <i>National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in "
        "the NHS</i>. Londres: RCP, 2017.",
        "LIU, F. T.; TING, K. M.; ZHOU, Z.-H. Isolation Forest. In: <i>IEEE International Conference on Data Mining (ICDM)</i>, 2008, p. 413–422.",
        "PAGE, E. S. Continuous inspection schemes. <i>Biometrika</i>, v. 41, n. 1/2, p. 100–115, 1954.",
        "BOERSMA, P. Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound. "
        "<i>Proceedings of the Institute of Phonetic Sciences</i>, Amsterdã, v. 17, p. 97–110, 1993.",
        "TEIXEIRA, J. P.; OLIVEIRA, C.; LOPES, C. Vocal acoustic analysis – jitter, shimmer and HNR parameters. <i>Procedia Technology</i>, "
        "v. 9, p. 1112–1122, 2013. https://doi.org/10.1016/j.protcy.2013.12.124.",
        "PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, v. 12, p. 2825–2830, 2011.",
        "PASZKE, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In: <i>Advances in Neural Information "
        "Processing Systems (NeurIPS)</i>, 2019.",
        "MICROSOFT. <i>Azure AI Speech — Speech to text</i>. Documentação. Disponível em: https://learn.microsoft.com/azure/ai-services/speech-service/. "
        "Acesso em: 15 nov. 2026.",
        "MICROSOFT. <i>Azure AI Language — análise de sentimento, extração de frases-chave e Text Analytics for health</i>. Documentação. "
        "Disponível em: https://learn.microsoft.com/azure/ai-services/language-service/. Acesso em: 15 nov. 2026.",
        "INSTITUTE FOR SAFE MEDICATION PRACTICES (ISMP). <i>List of High-Alert Medications in Acute Care Settings</i>. Disponível em: https://www.ismp.org.",
        "BRASIL. Lei nº 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais (LGPD).",
    ]
    h += [Paragraph(f"[{i}] {r}", REF) for i, r in enumerate(refs, 1)]
    h.append(Paragraph(
        "As doses máximas e interações medicamentosas usadas no protótipo são um conjunto ilustrativo montado para a "
        "demonstração, sem fonte farmacológica única, e não devem ser usadas como referência clínica.", LEG))

    SAIDA.parent.mkdir(exist_ok=True)
    SimpleDocTemplate(str(SAIDA), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm,
                      title="Monitor Clínico — Relatório resumido", author="Sebastião Figueredo dos Santos",
                      ).build(h, onFirstPage=rodape, onLaterPages=rodape)
    print(SAIDA)


if __name__ == "__main__":
    main()
