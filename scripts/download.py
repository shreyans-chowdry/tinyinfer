"""Download TinyLlama-1.1B-Chat (config, tokenizer, safetensors) to a local folder.

    python scripts/download.py --out ~/.cache/tinyllama
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out", default=str(Path.home() / ".cache" / "tinyllama"))
    args = ap.parse_args()
    path = snapshot_download(
        args.repo,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
        local_dir=args.out,
    )
    print(path)


if __name__ == "__main__":
    main()
