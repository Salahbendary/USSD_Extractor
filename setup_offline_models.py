"""
setup_offline_models.py
=======================
Run this script ONCE on a machine with internet access.
It downloads the PaddleOCR English models into  ./paddle_models/
so the extractor can run fully offline afterwards.

Usage:
    python setup_offline_models.py
    python setup_offline_models.py --model-dir /custom/path
"""

import argparse
import sys
from pathlib import Path


def download_models(model_dir: str = "./paddle_models"):
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PaddleOCR Offline Model Downloader")
    print(f"  Target folder: {model_path.resolve()}")
    print("=" * 60)

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("\n❌  PaddleOCR is not installed.")
        print("    Run:  pip install paddlepaddle paddleocr")
        sys.exit(1)

    print("\n⬇  Downloading English OCR models (det + rec + cls)...")
    print("   This happens only once. Models are cached locally.\n")

    # Instantiating PaddleOCR triggers automatic model download
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=True,
        use_gpu=False,
        # Force models to be saved in our custom folder
        det_model_dir=str(model_path / "det"),
        rec_model_dir=str(model_path / "rec"),
        cls_model_dir=str(model_path / "cls"),
    )

    # Quick smoke test
    import numpy as np
    dummy = np.ones((100, 300, 3), dtype=np.uint8) * 255
    result = ocr.ocr(dummy, cls=True)
    print("\n✅  Models downloaded and tested successfully!")
    print(f"\n   Models saved to: {model_path.resolve()}")
    print("\n   Run the extractor with:")
    print(f"   python extractor.py -i ./images -o results.csv "
          f"--model-dir {model_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", "-m", default="./paddle_models",
                        help="Where to save models (default: ./paddle_models)")
    args = parser.parse_args()
    download_models(args.model_dir)
