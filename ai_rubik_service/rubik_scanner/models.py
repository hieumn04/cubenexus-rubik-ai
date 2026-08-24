from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


ColorName = str
Box = tuple[int, int, int, int]
GridMatrix = list[list[ColorName]]
ConfidenceMatrix = list[list[float]]


class ScanState(str, Enum):
    WAITING_FOR_FACE = "WAITING_FOR_FACE"
    SCANNING_FACE = "SCANNING_FACE"
    FACE_LOCKED = "FACE_LOCKED"
    MOVE_TO_NEXT_FACE = "MOVE_TO_NEXT_FACE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class StickerDetection:
    xyxy: Box
    conf: float
    cls_id: int
    cls_name: ColorName
    area: float
    center_x: float
    center_y: float


@dataclass
class GridBuildResult:
    ordered_boxes: list[Box]
    ordered_colors: list[ColorName]
    ordered_confidences: list[float]
    grid_matrix: GridMatrix
    confidence_matrix: ConfidenceMatrix
    average_confidence: float


@dataclass
class FrameDetectionResult:
    ok: bool
    grid: GridMatrix | None = None
    confidence_matrix: ConfidenceMatrix | None = None
    avg_confidence: float = 0.0
    detected_stickers: int = 0
    frame_index: int = 0
    reason: str | None = None
    ordered_boxes: list[Box] = field(default_factory=list)


@dataclass
class FaceScanResult:
    status: str
    face_index: int
    face_name: str
    grid: GridMatrix | None = None
    cell_confidences: ConfidenceMatrix | None = None
    overall_confidence: float = 0.0
    valid_frames: int = 0
    reason: str | None = None
    possible_duplicate: bool = False
    duplicate_similarity: float = 0.0


@dataclass
class CubeScanOutput:
    status: str
    faces: dict[str, GridMatrix]
    color_counts: dict[str, int]
    overall_confidence: float
    captured_at: str
    face_order: list[str]
    validation: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(face_order: list[str]) -> "CubeScanOutput":
        return CubeScanOutput(
            status="INCOMPLETE",
            faces={},
            color_counts={},
            overall_confidence=0.0,
            captured_at=datetime.now(UTC).isoformat(),
            face_order=face_order,
            validation={},
        )
