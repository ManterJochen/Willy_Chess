"""
Train the chess piece detector (RT-DETR) with Workaholic-Willy's DetectorTraining.

The dataset is the YOLO set in data/ with the 12 classes "white pawn" ... "black king".
The best epoch lands in docs/weights/chess_pieces_rtdetr, its model.safetensors through Git LFS.

Run on the workstation (CUDA GPU), from the repository root:
    python -m src.pieces.train
    python -m src.pieces.train --data data/chess --epochs 80 --image-size 800
    python -m src.pieces.train --smoke      # 2 epochs on 64 images: proves the chain, not a model
"""
import argparse
from pathlib import Path

from willy import DetectorTraining

from src.mapping.frame import CLASSES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_OUT = REPO_ROOT / "docs" / "weights" / "chess_pieces_rtdetr"


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
        argparse.Namespace: The training settings; None leaves a setting to Willy's recipe.
    """
    parser = argparse.ArgumentParser(description="Train the chess piece detector (RT-DETR via Workaholic-Willy).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="YOLO dataset folder (default: data/)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output folder (default: docs/weights/chess_pieces_rtdetr, with --smoke ..._smoke)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Maximum epochs; early stopping still applies (default: the recipe's)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Images per batch (default: 8)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Training size in px, a multiple of 32 (default: 640)")
    parser.add_argument("--device", default=None,
                        help='"cuda", "cpu" or "mps" (default: the first one available)')
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the last epoch in the output folder")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs on 64 images: checks the setup, says nothing about quality")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke_out = DEFAULT_OUT.with_name(DEFAULT_OUT.name + "_smoke")
    out_dir = args.out or (smoke_out if args.smoke else DEFAULT_OUT)

    run = DetectorTraining.from_dataset(
        dataset=args.data,
        tier="smoke" if args.smoke else "full",
        epochs=args.epochs,
        batch=args.batch,
        image_size=args.image_size,
        classes=CLASSES,          # all 12: a dataset whose names differ is refused, not half trained
        out_dir=out_dir,
        format="yolo",
        device=args.device,
        resume=args.resume,
    )
    print(run.describe())

    # Images and boxes per class and split, before anything trains.
    print(run.probe())

    report = run.train()
    print(report)

    written = run.write_report(report)
    print(f"Model in {out_dir}, report in {written} and {written.with_name('report.html')}")


if __name__ == "__main__":
    main()
