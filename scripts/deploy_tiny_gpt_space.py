#!/usr/bin/env python3
"""Deploy the TinyGPT Space to Hugging Face — one command.

    HF_TOKEN=hf_xxx python scripts/deploy_tiny_gpt_space.py

Creates (or reuses) https://huggingface.co/spaces/Abol0pati/tiny-gpt-fa
and uploads `hf-space-tiny-gpt/`. The Space is **static** (free tier): the
browser runs the vendored `src/` stack in Pyodide/WebAssembly against the
committed 2 MB checkpoint, so no CPU/Gradio hosting is needed. Requires
`pip install huggingface_hub` and a token with write access.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "Abol0pati/tiny-gpt-fa"
SPACE_DIR = Path(__file__).resolve().parent.parent / "hf-space-tiny-gpt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""),
                        help="HF token (or set the HF_TOKEN env var)")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    if not args.token:
        sys.exit("No token: pass --token hf_... or set the HF_TOKEN env var.")

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    url = f"https://huggingface.co/spaces/{args.repo_id}"
    print(f"creating/reusing static Space {args.repo_id} ...")
    api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk="static",
                    private=False, exist_ok=True)
    print(f"uploading {SPACE_DIR} ...")
    api.upload_folder(repo_id=args.repo_id, repo_type="space",
                      folder_path=str(SPACE_DIR),
                      commit_message="TinyGPT live demo — NumPy autograd + Persian char-level transformer")
    print(f"done -> {url}  (first build takes ~1-2 minutes)")


if __name__ == "__main__":
    main()
