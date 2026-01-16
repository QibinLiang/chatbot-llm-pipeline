import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() in {".json"}:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if config_path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("YAML config requires PyYAML") from exc
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    raise ValueError(f"Unsupported config format: {config_path.suffix}")
