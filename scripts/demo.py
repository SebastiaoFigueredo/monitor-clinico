"""Demonstração ponta a ponta: analisa todos os pacientes de data/pacientes e imprime os alertas.

    python scripts/demo.py                 # todos os pacientes
    python scripts/demo.py P003            # um paciente
    python scripts/demo.py --video caminho.mp4 --paciente P010   # um vídeo real avulso
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src import config  # noqa: E402
from src.assistente import AVISO, AssistenteClinico  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pacientes", nargs="*", help="IDs (padrão: todos)")
    ap.add_argument("--video", help="arquivo de vídeo real para analisar")
    ap.add_argument("--audio", help="arquivo WAV de consulta para analisar")
    ap.add_argument("--paciente", default="AVULSO")
    ap.add_argument("--backend", default=config.POSE_BACKEND, choices=["yolo", "openpose"])
    ap.add_argument("--zonas", help="JSON com as zonas do quarto (ver README)")
    ap.add_argument("--perguntar", help="pergunta em linguagem natural ao assistente (usa Claude)")
    args = ap.parse_args()

    if config.ALERTAS_ARQUIVO.exists():
        config.ALERTAS_ARQUIVO.unlink()
    assistente = AssistenteClinico()

    if args.video or args.audio:
        if args.video:
            from src.video.objetos import carregar_zonas

            zonas = carregar_zonas(args.zonas) if args.zonas else None
            saida = config.SAIDAS / f"{Path(args.video).stem}_anotado.mp4"
            assistente.analisar_video(args.paciente, args.video, saida_anotada=saida, backend=args.backend, zonas=zonas)
            print(f"\nVídeo anotado: {saida}")
        if args.audio:
            assistente.analisar_audio(args.paciente, args.audio)
            a = assistente.pacientes[args.paciente]["analises"]["audio"]
            print("\nBiomarcadores vocais:", a["features"])
            if a.get("transcricao"):
                print(f"Transcrição ({a['transcricao_origem']}): {a['transcricao']}")
            if a.get("analise_texto"):
                t = a["analise_texto"]
                print(f"Sentimento: {t['sentimento']} (negativo {t['negativo']:.0%}) | frases-chave: {', '.join(t['frases_chave'])}")
            for aviso in a.get("avisos", []):
                print("Aviso:", aviso)
    else:
        pastas = sorted(p for p in (config.DADOS / "pacientes").glob("*") if (p / "paciente.json").exists())
        if not pastas:
            sys.exit("Nenhum paciente em data/pacientes. Rode antes: python scripts/gerar_dados_sinteticos.py")
        for pasta in pastas:
            if args.pacientes and pasta.name not in args.pacientes:
                continue
            print(f"\n=== Analisando {pasta.name} ===")
            assistente.analisar_pasta(pasta)

    print("\n" + "=" * 100 + "\nSITUAÇÃO DA UNIDADE (mais grave primeiro)\n" + "=" * 100)
    for s in assistente.situacao():
        print(f"{s['id']}  leito {s.get('leito', '-'):6}  pior: {s['pior_severidade']:<8} "
              f"abertos: {s['alertas_abertos']:>2}  {s['por_severidade']}")
    for s in assistente.situacao():
        print("\n" + assistente.resumo_paciente(s["id"]))
    if args.perguntar:
        print("\n" + "=" * 100 + f"\nPergunta: {args.perguntar}\n")
        print(assistente.perguntar(args.perguntar))
    print(f"\nAlertas gravados em {config.ALERTAS_ARQUIVO}\n{AVISO}")


if __name__ == "__main__":
    main()
