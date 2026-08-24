from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import cv2

from .color_utils import color_count_summary, grid_similarity, validate_color_counts
from .consensus import FaceConsensusBuffer
from .models import CubeScanOutput, FaceScanResult, FrameDetectionResult, ScanState


class RubikScanStateMachine:
    def __init__(self, face_order: list[str], scan_seconds: float, outputs_dir: Path) -> None:
        self.face_order = face_order
        self.scan_seconds = float(scan_seconds)
        self.outputs_dir = outputs_dir
        self.state = ScanState.WAITING_FOR_FACE
        self.current_index = 0
        self.faces: dict[str, FaceScanResult] = {}
        self.status_text = "Show face 1 and press SPACE to start scan."
        self.countdown_text = ""
        self.consensus_buffer: FaceConsensusBuffer | None = None
        self.completed_output: CubeScanOutput | None = None

    @property
    def current_face(self) -> str:
        return self.face_order[min(self.current_index, len(self.face_order) - 1)]

    def handle_key(self, key: int) -> bool:
        if key in (27, ord("q"), ord("Q")):
            return False

        if key == ord(" "):
            self._handle_space()
        elif key in (ord("r"), ord("R")):
            self.retry_current_face()
        elif key in (ord("b"), ord("B")):
            self.go_back()
        elif key in (ord("c"), ord("C")):
            self.clear_all()

        return True

    def update(self, frame, frame_result: FrameDetectionResult) -> None:
        if self.state != ScanState.SCANNING_FACE or self.consensus_buffer is None:
            return

        self.consensus_buffer.add(frame_result)
        remaining = max(0.0, self.scan_seconds - self.consensus_buffer.elapsed())
        self.countdown_text = f"Scanning {self.current_face}: {remaining:.1f}s"

        if self.consensus_buffer.is_finished():
            result = self.consensus_buffer.finalize()
            if result.status == "PASSED" and result.grid:
                similarity = self._max_similarity(result.grid)
                if similarity >= 0.80:
                    result.possible_duplicate = True
                    result.duplicate_similarity = similarity
                    self.status_text = "This face looks similar to a previous face. Rotate and retry."
                    self.state = ScanState.FAILED
                    self.countdown_text = ""
                    return

                self.faces[result.face_name] = result
                self.status_text = f"Face {result.face_name} captured. Press SPACE for next side."
                self.state = ScanState.FACE_LOCKED
                self.countdown_text = ""
                self._save_debug_frame(frame, result.face_name)
                if len(self.faces) == len(self.face_order):
                    self._complete()
            else:
                self.status_text = self._reason_to_text(result.reason)
                self.state = ScanState.FAILED
                self.countdown_text = ""

    def _handle_space(self) -> None:
        if self.state in {ScanState.WAITING_FOR_FACE, ScanState.FAILED}:
            self._start_scan()
            return

        if self.state == ScanState.FACE_LOCKED:
            if len(self.faces) == len(self.face_order):
                self.state = ScanState.COMPLETED
                self.status_text = "Cube scan completed."
                return

            self.current_index = min(self.current_index + 1, len(self.face_order) - 1)
            self.state = ScanState.WAITING_FOR_FACE
            self.status_text = f"Show face {self.current_face} and press SPACE to start scan."
            return

    def _start_scan(self) -> None:
        self.consensus_buffer = FaceConsensusBuffer(
            face_index=self.current_index,
            face_name=self.current_face,
            scan_seconds=self.scan_seconds,
        )
        self.consensus_buffer.start()
        self.state = ScanState.SCANNING_FACE
        self.status_text = f"Scanning face {self.current_face} for {self.scan_seconds:.0f} seconds..."

    def retry_current_face(self) -> None:
        current_face = self.current_face
        self.faces.pop(current_face, None)
        self.state = ScanState.WAITING_FOR_FACE
        self.status_text = f"Retry face {current_face}. Press SPACE to start scan."
        self.countdown_text = ""
        self.consensus_buffer = None

    def go_back(self) -> None:
        if self.current_index == 0:
            self.status_text = "Already at first face."
            return

        current_face = self.current_face
        self.faces.pop(current_face, None)
        self.current_index -= 1
        previous_face = self.current_face
        self.faces.pop(previous_face, None)
        self.state = ScanState.WAITING_FOR_FACE
        self.status_text = f"Back to face {previous_face}. Press SPACE to scan again."
        self.countdown_text = ""
        self.consensus_buffer = None

    def clear_all(self) -> None:
        self.faces.clear()
        self.current_index = 0
        self.state = ScanState.WAITING_FOR_FACE
        self.status_text = "All scan state cleared. Show face U and press SPACE."
        self.countdown_text = ""
        self.consensus_buffer = None
        self.completed_output = None

    def _max_similarity(self, grid) -> float:
        scores = []
        for face_name, result in self.faces.items():
            if face_name == self.current_face or not result.grid:
                continue
            scores.append(grid_similarity(grid, result.grid))
        return max(scores, default=0.0)

    def _complete(self) -> None:
        faces_payload = {face_name: result.grid for face_name, result in self.faces.items() if result.grid}
        counts = color_count_summary(faces_payload)
        valid, validation = validate_color_counts(faces_payload)
        avg_conf = sum(result.overall_confidence for result in self.faces.values()) / max(1, len(self.faces))

        output = CubeScanOutput(
            status="COMPLETED" if valid else "INVALID_COLOR_COUNT",
            faces=faces_payload,
            color_counts=counts,
            overall_confidence=avg_conf,
            captured_at=datetime.now(UTC).isoformat(),
            face_order=self.face_order,
            validation={key: value for key, value in validation.items()},
        )
        self.completed_output = output

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.outputs_dir / "cube_scan_result.json"
        output_path.write_text(json.dumps(asdict(output), indent=2), encoding="utf-8")

        self.state = ScanState.COMPLETED
        self.status_text = f"Completed. JSON saved to {output_path}."
        self.countdown_text = ""

    def _save_debug_frame(self, frame, face_name: str) -> None:
        if frame is None or getattr(frame, "size", 0) == 0:
            return
        debug_dir = self.outputs_dir / "debug_frames"
        debug_dir.mkdir(parents=True, exist_ok=True)
        filename = debug_dir / f"{int(time.time())}_{face_name}.png"
        cv2.imwrite(str(filename), frame)

    @staticmethod
    def _reason_to_text(reason: str | None) -> str:
        mapping = {
            "LOW_VALID_FRAME_COUNT": "Need more valid frames. Retry this face.",
            "LOW_STABILITY": "Scan unstable, retry.",
            "LOW_FACE_STABILITY": "Face stability too low, retry.",
            "MISSING_CELL_VOTES": "Need 9 stickers consistently visible.",
        }
        return mapping.get(reason, "Scan failed. Retry current face.")
