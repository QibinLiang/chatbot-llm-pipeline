#!/usr/bin/env python3
"""
Pre-download default models with progress output.

Covers the default ASR model (faster-whisper) and the default LLM model.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List


def _resolve_asr_repo_id(model_name: str) -> str:
    known = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
    if "/" in model_name:
        return model_name
    if model_name in known:
        return f"Systran/faster-whisper-{model_name}"
    return model_name


def _download_repo_files(repo_id: str, revision: str | None) -> None:
    try:
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore
    except Exception as exc:
        print("Missing dependency: huggingface_hub. Install faster-whisper first.", file=sys.stderr)
        raise SystemExit(1) from exc

    api = HfApi()
    files: List[str] = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
    if not files:
        print(f"No files found for {repo_id}", file=sys.stderr)
        return

    total = len(files)
    print(f"Downloading model files for {repo_id} ({total} files)")
    for idx, filename in enumerate(files, start=1):
        print(f"[{idx}/{total}] {filename}")
        hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, repo_type="model")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download default models with progress")
    parser.add_argument("--asr-model", default=os.environ.get("ASR_MODEL", "small"), help="ASR model name or repo id")
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"), help="LLM model repo id")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM model download")
    parser.add_argument("--revision", default=None, help="Model revision (optional)")
    args = parser.parse_args()

    asr_repo = _resolve_asr_repo_id(args.asr_model)
    _download_repo_files(asr_repo, args.revision)
    if not args.skip_llm and args.llm_model:
        _download_repo_files(args.llm_model, args.revision)


if __name__ == "__main__":
    main()
