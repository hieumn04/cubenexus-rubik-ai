from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Keep the production model inside this service so the same path works on
# Windows, Linux, Docker, and Railway.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best.pt"


def _read_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _read_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RubikAiSettings:
    service_name: str
    model_path: Path
    model_version: str
    enable_model_load: bool
    scanner_required_stable_observations: int
    scanner_stability_timeout_seconds: float
    scanner_stable_grid_match_min_cells: int

    @staticmethod
    def load() -> "RubikAiSettings":
        model_path = Path(os.getenv("AI_RUBIK_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        model_version = os.getenv("AI_RUBIK_MODEL_VERSION", "rubik-sticker-v1")
        service_name = os.getenv("AI_RUBIK_SERVICE_NAME", "CubeNexus AI Rubik Service")

        return RubikAiSettings(
            service_name=service_name,
            model_path=model_path,
            model_version=model_version,
            enable_model_load=_read_bool("AI_RUBIK_ENABLE_MODEL_LOAD", True),
            scanner_required_stable_observations=max(1, _read_int("AI_SCANNER_REQUIRED_STABLE_OBSERVATIONS", 2)),
            scanner_stability_timeout_seconds=max(1.5, _read_float("AI_SCANNER_STABILITY_TIMEOUT_SECONDS", 30.0)),
            scanner_stable_grid_match_min_cells=max(5, min(9, _read_int("AI_SCANNER_STABLE_GRID_MATCH_MIN_CELLS", 7))),
        )
