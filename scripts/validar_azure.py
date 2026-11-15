"""Valida a integração REAL com a Azure (consome cota dos seus recursos).

    python scripts/validar_azure.py --diagnostico      # só confere as chaves e diz o que corrigir
    python scripts/validar_azure.py                    # gera uma fala de teste com o Azure TTS,
                                                       # transcreve (Speech to Text) e analisa (Text Analytics)
    python scripts/validar_azure.py --audio consulta.wav
    python scripts/validar_azure.py --texto "Estou com dor no peito e muito triste."

Requer no .env: AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, AZURE_LANGUAGE_KEY, AZURE_LANGUAGE_ENDPOINT.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src import config  # noqa: E402
from src.audio import azure_ia  # noqa: E402
from src.audio.texto import anomalias_do_texto  # noqa: E402

ROTEIRO = ("Doutor, desde ontem estou com muita falta de ar e um aperto no peito. "
           "Eu parei de tomar a varfarina na semana passada. "
           "Não tive febre. Estou muito triste e não aguento mais ficar internado.")


def sintetizar(texto: str, destino: Path, voz: str = "pt-BR-AntonioNeural") -> Path:
    """Gera um WAV 16 kHz mono com o Azure Text to Speech (mesmo recurso do Speech to Text)."""
    import azure.cognitiveservices.speech as speechsdk

    cfg = speechsdk.SpeechConfig(subscription=config.AZURE_SPEECH_KEY, region=config.AZURE_SPEECH_REGION)
    cfg.speech_synthesis_voice_name = voz
    cfg.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
    r = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None).speak_text_async(texto).get()
    if r.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"síntese falhou: {r.cancellation_details.error_details}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.audio_data)
    return destino


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio")
    ap.add_argument("--texto")
    ap.add_argument("--diagnostico", action="store_true", help="só testa as credenciais e sai")
    args = ap.parse_args()
    print("Diagnóstico das credenciais:")
    for item in azure_ia.diagnostico():
        print(f"  [{'OK' if item['ok'] else 'FALHA'}] {item['servico']}: {item['mensagem']}")
    if args.diagnostico:
        return
    relatorio = ["# Evidência de execução com a Azure real", ""]

    texto = args.texto
    if texto is None:
        if not azure_ia.fala_configurada():
            sys.exit("\nSem Azure Speech não há como transcrever. Siga o GUIA_AZURE.md e rode de novo "
                     "(ou use --texto para testar só a análise de texto).")
        audio = Path(args.audio) if args.audio else sintetizar(ROTEIRO, config.DADOS / "falas" / "teste_azure.wav")
        if not args.audio:
            print(f"\nRoteiro falado (TTS):\n  {ROTEIRO}\nÁudio: {audio}")
        t = azure_ia.transcrever(audio, termos_dica=["varfarina", "heparina", "amiodarona"])
        relatorio += [f"**Áudio:** `{audio.name}`", "", "## Transcrição (Azure Speech to Text)", ""]
        relatorio += [f"- [{tr.inicio_s:.1f}–{tr.fim_s:.1f} s, confiança {tr.confianca}] {tr.texto}" for tr in t.trechos]
        print("\nTranscrição (Speech to Text):")
        for tr in t.trechos:
            conf = f"{tr.confianca:.2f}" if tr.confianca is not None else "-"
            print(f"  [{tr.inicio_s:6.1f}–{tr.fim_s:6.1f}s  conf {conf}] {tr.texto}")
        texto = t.texto

    analise = None
    if azure_ia.linguagem_configurada():
        try:
            analise = azure_ia.analisar_texto(texto)
        except RuntimeError as e:
            print(f"\nText Analytics falhou (veja o diagnóstico no topo): {str(e).splitlines()[0]}")
    if analise is not None:
        print(f"\nSentimento (Text Analytics): {analise.sentimento}  "
              f"[positivo {analise.positivo:.2f} · neutro {analise.neutro:.2f} · negativo {analise.negativo:.2f}]")
        for f in analise.frases_negativas:
            print(f"  negativa ({f.negativo:.2f}): {f.texto}")
        print("Frases-chave:", ", ".join(analise.frases_chave) or "-")
        for e in analise.entidades_saude:
            print(f"  entidade de saúde: {e.texto} [{e.categoria}]{' (NEGADA)' if e.negada else ''}")
        relatorio += ["", "## Azure Text Analytics", "",
                      f"- Sentimento: **{analise.sentimento}** (positivo {analise.positivo:.2f}, neutro "
                      f"{analise.neutro:.2f}, negativo {analise.negativo:.2f})",
                      f"- Frases-chave: {', '.join(analise.frases_chave) or '-'}"]
        relatorio += [f"- Entidade de saúde: {e.texto} [{e.categoria}]{' (negada)' if e.negada else ''}"
                      for e in analise.entidades_saude]
    elif not azure_ia.linguagem_configurada():
        print("\n(Text Analytics não configurado: só o léxico local de termos críticos.)")

    print("\nAnomalias geradas:")
    anomalias = anomalias_do_texto(texto, "TESTE", analise=analise)
    for a in anomalias:
        print(f"  [{a.severidade.name}] {a.tipo}: {a.descricao}")
    if analise is not None or args.texto is None:  # houve chamada real à Azure: guarda a evidência
        from datetime import datetime

        relatorio += ["", "## Anomalias geradas", ""] + [f"- [{a.severidade.name}] {a.tipo}: {a.descricao}" for a in anomalias]
        relatorio.insert(1, f"\nGerado em {datetime.now():%d/%m/%Y %H:%M} por `scripts/validar_azure.py`.")
        destino = config.RAIZ / "docs" / "evidencia_azure.md"
        destino.parent.mkdir(exist_ok=True)
        destino.write_text("\n".join(relatorio), encoding="utf-8")
        print(f"\nEvidência salva em {destino} (pode ir junto com o trabalho).")


if __name__ == "__main__":
    main()
