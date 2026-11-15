"""Baixa o modelo COCO do OpenPose (CMU) para models/openpose/ (~200 MB).

O servidor original da CMU fica fora do ar com frequência; se todos os endereços falharem,
baixe `pose_iter_440000.caffemodel` manualmente e coloque na pasta, ou informe outro
endereço:  python scripts/baixar_openpose.py --url https://.../pose_iter_440000.caffemodel
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

PROTO = ("https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/"
         "models/pose/coco/pose_deploy_linevec.prototxt")
PESOS = [
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
    "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_iter_440000.caffemodel",
]


def baixar(url: str, destino: Path, minimo_bytes: int) -> bool:
    print(f"baixando {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "monitor-clinico"})
        with urllib.request.urlopen(req, timeout=30) as r, open(destino, "wb") as f:
            while bloco := r.read(1 << 20):
                f.write(bloco)
    except OSError as e:
        print(f"  falhou: {e}")
        destino.unlink(missing_ok=True)
        return False
    if destino.stat().st_size < minimo_bytes:
        print(f"  arquivo incompleto ({destino.stat().st_size} bytes)")
        destino.unlink()
        return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="endereço alternativo do .caffemodel")
    args = ap.parse_args()
    config.OPENPOSE_PROTO.parent.mkdir(parents=True, exist_ok=True)
    if not config.OPENPOSE_PROTO.exists() and not baixar(PROTO, config.OPENPOSE_PROTO, 10_000):
        sys.exit("não foi possível baixar o prototxt")
    if not config.OPENPOSE_PESOS.exists():
        if not any(baixar(u, config.OPENPOSE_PESOS, 100_000_000) for u in ([args.url] if args.url else PESOS)):
            sys.exit("não foi possível baixar os pesos — veja as instruções no topo deste arquivo")
    print("OpenPose pronto em", config.OPENPOSE_PESOS.parent)
