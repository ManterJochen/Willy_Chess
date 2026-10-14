"""
Download the two models Willy_Chess builds on, once per machine:

- rtdetr (PekingU/rtdetr_r50vd, 0.2 GB): the base the piece detector is fine-tuned from (src/pieces/train.py).
- sam2 (facebook/sam2-hiera-large, 0.9 GB): the piece masks during a game.

Both land where Willy looks for them (assets/models/hf in the Willy checkout), so training and the
game find them without any config. Willy's fetch script checks TLS against the operating system's
certificate store (truststore), which is what a company proxy needs; without it the download fails
with CERTIFICATE_VERIFY_FAILED. An interrupted download resumes.

    python -m src.pieces.fetch
"""
import subprocess
import sys
from pathlib import Path

import willy

FETCH = Path(willy.__file__).resolve().parents[1] / "scripts" / "model_weights" / "fetch.py"
MODELS = ("rtdetr", "sam2")


def main() -> int:
    return subprocess.call([sys.executable, str(FETCH), *MODELS])


if __name__ == "__main__":
    sys.exit(main())
