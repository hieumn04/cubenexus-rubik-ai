from __future__ import annotations

import time
from collections import Counter

from .models import FaceScanResult, FrameDetectionResult


class FaceConsensusBuffer:
    def __init__(
        self,
        face_index: int,
        face_name: str,
        scan_seconds: float = 5.0,
        min_valid_frames: int = 12,
        cell_majority_threshold: float = 0.60,
        face_stability_threshold: float = 0.70,
    ) -> None:
        self.face_index = face_index
        self.face_name = face_name
        self.scan_seconds = float(scan_seconds)
        self.min_valid_frames = int(min_valid_frames)
        self.cell_majority_threshold = float(cell_majority_threshold)
        self.face_stability_threshold = float(face_stability_threshold)
        self.started_at: float | None = None
        self.frames: list[FrameDetectionResult] = []

    def start(self) -> None:
        self.started_at = time.time()
        self.frames = []

    def add(self, frame_result: FrameDetectionResult) -> None:
        if frame_result.ok and frame_result.grid and frame_result.confidence_matrix:
            self.frames.append(frame_result)

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.time() - self.started_at

    def is_finished(self) -> bool:
        return self.started_at is not None and self.elapsed() >= self.scan_seconds

    def finalize(self) -> FaceScanResult:
        valid_frames = len(self.frames)
        if valid_frames < self.min_valid_frames:
            return FaceScanResult(
                status="FAILED",
                face_index=self.face_index,
                face_name=self.face_name,
                valid_frames=valid_frames,
                reason="LOW_VALID_FRAME_COUNT",
            )

        output_grid: list[list[str]] = [["unknown"] * 3 for _ in range(3)]
        output_confidences: list[list[float]] = [[0.0] * 3 for _ in range(3)]
        cell_stability_scores: list[float] = []

        for row in range(3):
            for col in range(3):
                votes = Counter(frame.grid[row][col] for frame in self.frames if frame.grid)
                if not votes:
                    return FaceScanResult(
                        status="FAILED",
                        face_index=self.face_index,
                        face_name=self.face_name,
                        valid_frames=valid_frames,
                        reason="MISSING_CELL_VOTES",
                    )

                majority_color, majority_count = votes.most_common(1)[0]
                majority_ratio = majority_count / valid_frames
                mean_conf = self._mean_cell_confidence(row, col, majority_color)

                output_grid[row][col] = majority_color
                output_confidences[row][col] = mean_conf
                cell_stability_scores.append(majority_ratio)

                if majority_ratio < self.cell_majority_threshold:
                    return FaceScanResult(
                        status="FAILED",
                        face_index=self.face_index,
                        face_name=self.face_name,
                        valid_frames=valid_frames,
                        reason="LOW_STABILITY",
                    )

        overall_confidence = sum(cell_stability_scores) / len(cell_stability_scores)
        if overall_confidence < self.face_stability_threshold:
            return FaceScanResult(
                status="FAILED",
                face_index=self.face_index,
                face_name=self.face_name,
                valid_frames=valid_frames,
                reason="LOW_FACE_STABILITY",
                overall_confidence=overall_confidence,
            )

        return FaceScanResult(
            status="PASSED",
            face_index=self.face_index,
            face_name=self.face_name,
            grid=output_grid,
            cell_confidences=output_confidences,
            overall_confidence=overall_confidence,
            valid_frames=valid_frames,
        )

    def _mean_cell_confidence(self, row: int, col: int, color: str) -> float:
        values = []
        for frame in self.frames:
            if not frame.grid or not frame.confidence_matrix:
                continue
            if frame.grid[row][col] == color:
                values.append(frame.confidence_matrix[row][col])
        if not values:
            return 0.0
        return sum(values) / len(values)
