from __future__ import annotations

import base64
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rubik_scanner.color_utils import color_count_summary, grid_similarity, validate_color_counts
from rubik_scanner.grid_builder import SpatialGridBuilder
from rubik_scanner.models import FrameDetectionResult, StickerDetection

from .schemas import (
    AiCheckResponse,
    AnalyzeFrameResponse,
    CubeScanValidationResponse,
    ScannerFaceResult,
    ScannerPreviewResponse,
    ScannerSessionResponse,
    ScannerStickerObservation,
)
from .settings import RubikAiSettings

@dataclass
class InferenceTimings:
    decode_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class ScannerSessionFace:
    center_color: str
    grid3x3: list[list[str]]
    stickers: list[dict[str, Any]]
    overall_confidence: float
    valid_frames: int
    captured_at: datetime


@dataclass
class ScannerObservation:
    center_color: str
    grid3x3: list[list[str]]
    stickers: list[dict[str, Any]]
    overall_confidence: float
    key: str


@dataclass
class ScannerBurstState:
    started_at: float | None = None
    stable_observation_count: int = 0
    last_observation_key: str | None = None
    last_observation: ScannerObservation | None = None
    observation_counts: dict[str, int] = field(default_factory=dict)
    scan_generation: int = 0
    target_face_index: int = 1
    request_id: str | None = None
    scanner_state: str = "POSITION_FACE"
    message: str = "Hold one complete face inside the frame."

    def reset(self, message: str, scanner_state: str = "POSITION_FACE") -> None:
        self.started_at = None
        self.stable_observation_count = 0
        self.last_observation_key = None
        self.last_observation = None
        self.observation_counts.clear()
        self.request_id = None
        self.scanner_state = scanner_state
        self.message = message


@dataclass
class ScannerSessionState:
    session_id: str
    started_at: datetime
    captured_faces: list[ScannerSessionFace]
    metadata: dict[str, Any]
    completed_at: datetime | None = None
    last_face_scan: ScannerSessionFace | None = None
    last_scan_status: str | None = None
    last_scan_reason: str | None = None
    burst: ScannerBurstState = field(default_factory=ScannerBurstState)


class RubikAiService:
    def __init__(self, settings: RubikAiSettings) -> None:
        self._settings = settings
        self._model_loaded = False
        self._load_error: str | None = None
        self._model = None
        self._grid_builder = SpatialGridBuilder(min_confidence=0.35)
        self._scanner_sessions: dict[str, ScannerSessionState] = {}
        self._inference_gate = threading.BoundedSemaphore(value=1)
        self._torch = None
        self._try_load_model()

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _try_load_model(self) -> None:
        if not self._settings.enable_model_load:
            self._load_error = "Model loading disabled for Phase 2 skeleton."
            return

        if not self._settings.model_path.exists():
            self._load_error = f"Model file not found: {self._settings.model_path}"
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self._settings.model_path))
            try:
                import torch

                self._torch = torch
            except Exception:
                self._torch = None

            try:
                self._model.fuse()
            except Exception:
                pass
            self._warm_up_model()
            self._model_loaded = True
            self._load_error = None
        except Exception as exc:
            self._model_loaded = False
            self._load_error = str(exc)

    def _warm_up_model(self) -> None:
        if self._model is None:
            return
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            context = self._torch.inference_mode() if self._torch is not None else nullcontext()
            with context:
                self._model.predict(
                    source=dummy,
                    imgsz=640,
                    conf=0.35,
                    verbose=False,
                    device="cpu",
                    half=False,
                )
        except Exception:
            pass

    def run_check(self, check_type: str, payload: dict[str, Any] | None = None) -> AiCheckResponse:
        payload = payload or {}
        analyzed = self._analyze_payload(payload)
        now = datetime.now(UTC)

        if analyzed["frame"] is None:
            return AiCheckResponse(
                checkType=check_type,
                status="AI_CHECK_UNAVAILABLE" if not self._model_loaded else "FAILED",
                confidence=0.0,
                detectedCube=False,
                detectedStickers=0,
                grid3x3=None,
                reason=analyzed["reason"],
                modelVersion=self._settings.model_version,
                modelLoaded=self._model_loaded,
                createdAt=now,
            )

        frame_result, _ = self._infer_frame_detailed(analyzed["frame"], 0)
        if check_type == "PRE_CHECK":
            if frame_result.ok and frame_result.grid is not None:
                return AiCheckResponse(
                    checkType=check_type,
                    status="PASSED",
                    confidence=frame_result.avg_confidence,
                    detectedCube=True,
                    detectedStickers=frame_result.detected_stickers,
                    grid3x3=frame_result.grid,
                    reason=None,
                    modelVersion=self._settings.model_version,
                    modelLoaded=self._model_loaded,
                    createdAt=now,
                )

            return AiCheckResponse(
                checkType=check_type,
                status="FAILED",
                confidence=0.0,
                detectedCube=False,
                detectedStickers=frame_result.detected_stickers,
                grid3x3=None,
                reason=frame_result.reason,
                modelVersion=self._settings.model_version,
                modelLoaded=self._model_loaded,
                createdAt=now,
            )

        if check_type == "SCRAMBLE_CHECK":
            return AiCheckResponse(
                checkType=check_type,
                status="NEEDS_REVIEW",
                confidence=frame_result.avg_confidence if frame_result.ok and frame_result.grid else 0.0,
                detectedCube=frame_result.ok and frame_result.grid is not None,
                detectedStickers=frame_result.detected_stickers,
                grid3x3=frame_result.grid,
                expectedScramble=str(payload.get("scrambleSequence") or ""),
                detectedState=self._grid_to_state(frame_result.grid),
                isScrambleMatched=None,
                reason="Cube color scan available, but full scramble state comparison is not implemented yet.",
                modelVersion=self._settings.model_version,
                modelLoaded=self._model_loaded,
                createdAt=now,
            )

        return AiCheckResponse(
            checkType=check_type,
            status="NEEDS_REVIEW",
            confidence=frame_result.avg_confidence if frame_result.ok and frame_result.grid else 0.0,
            detectedCube=frame_result.ok and frame_result.grid is not None,
            detectedStickers=frame_result.detected_stickers,
            grid3x3=frame_result.grid,
            isSolved=None,
            reason="Single-frame finish validation is available, but full solved-state verification is not implemented yet.",
            modelVersion=self._settings.model_version,
            modelLoaded=self._model_loaded,
            createdAt=now,
        )

    def analyze_frame(self, payload: dict[str, Any] | None = None) -> AnalyzeFrameResponse:
        payload = payload or {}
        analyzed = self._analyze_payload(payload)
        notes: list[str] = []
        if self._load_error:
            notes.append(self._load_error)
        if analyzed["reason"]:
            notes.append(analyzed["reason"])

        if analyzed["frame"] is None:
            return AnalyzeFrameResponse(
                status="FAILED",
                modelVersion=self._settings.model_version,
                labels=[],
                notes=notes or ["No valid image payload was provided."],
                detectedStickers=0,
                grid3x3=None,
                confidence=0.0,
            )

        frame_result, _ = self._infer_frame_detailed(analyzed["frame"], 0)
        return AnalyzeFrameResponse(
            status="DETECTED" if frame_result.ok and frame_result.grid else "NOT_DETECTED",
            modelVersion=self._settings.model_version,
            labels=self._flatten(frame_result.grid),
            notes=notes + ([frame_result.reason] if frame_result.reason else []),
            detectedStickers=frame_result.detected_stickers,
            grid3x3=frame_result.grid,
            confidence=frame_result.avg_confidence,
        )

    def validate_cube_scan(self, faces: dict[str, list[list[str]]]) -> CubeScanValidationResponse:
        valid, validation = validate_color_counts(faces)
        return CubeScanValidationResponse(
            status="VALID_COLOR_COUNT" if valid else "INVALID_COLOR_COUNT",
            colorCounts=color_count_summary(faces),
            validation=validation,
        )

    def start_scanner_test_session(self, metadata: dict[str, Any] | None = None) -> ScannerSessionResponse:
        session = ScannerSessionState(
            session_id=str(uuid4()),
            started_at=datetime.now(UTC),
            captured_faces=[],
            metadata=metadata or {},
        )
        session.burst.reset("Hold one complete face inside the frame.", "POSITION_FACE")
        self._scanner_sessions[session.session_id] = session
        return self._build_session_response(session, message=session.burst.message)

    def get_scanner_test_session(self, session_id: str) -> ScannerSessionResponse:
        session = self._require_scanner_session(session_id)
        return self._build_session_response(session, message=session.burst.message)

    def preview_scanner_test_frame(self, session_id: str, payload: dict[str, Any] | None = None) -> ScannerPreviewResponse:
        session = self._require_scanner_session(session_id)
        analyzed = self._analyze_payload(payload or {})
        if analyzed["frame"] is None:
            return self._build_preview_response(
                session,
                scanner_state="AI_UNAVAILABLE" if not self._model_loaded else "CAMERA_ERROR",
                status="FAILED",
                reason=analyzed["reason"],
                timings=analyzed["timings"],
            )

        frame_result, timings = self._infer_frame_detailed(analyzed["frame"], 0)
        return self._build_preview_response(
            session,
            scanner_state="POSITION_FACE" if frame_result.ok else "RETRY",
            status="DETECTED" if frame_result.ok else "RETRY",
            frame_result=frame_result,
            timings=timings,
            reason=None if frame_result.ok else frame_result.reason,
        )

    def observe_scanner_test_frame(self, session_id: str, payload: dict[str, Any] | None = None) -> ScannerPreviewResponse:
        session = self._require_scanner_session(session_id)
        analyzed = self._analyze_payload(payload or {})
        burst = session.burst
        identity = self._scanner_request_identity(session, payload or {})

        if identity["scan_session_id"] != session.session_id:
            return self._build_preview_response(
                session,
                scanner_state="RETRY",
                status="RETRY",
                reason="Scanner session mismatch. Restart scanning.",
                timings=analyzed["timings"],
                identity=identity,
            )

        if analyzed["frame"] is None:
            burst.reset("AI unavailable. Retry the current face.", "AI_UNAVAILABLE" if not self._model_loaded else "CAMERA_ERROR")
            session.last_scan_status = burst.scanner_state
            session.last_scan_reason = analyzed["reason"]
            return self._build_preview_response(
                session,
                scanner_state=burst.scanner_state,
                status=burst.scanner_state,
                reason=analyzed["reason"],
                timings=analyzed["timings"],
                identity=identity,
            )

        if burst.started_at is None:
            burst.started_at = time.perf_counter()
            burst.stable_observation_count = 0
            burst.last_observation_key = None
            burst.last_observation = None
            burst.scan_generation = identity["scan_generation"]
            burst.target_face_index = identity["target_face_index"]
            burst.request_id = identity["request_id"]
            burst.scanner_state = "SCANNING"
            burst.message = f"Scanning 0/{self._settings.scanner_required_stable_observations} stable observations."

        if identity["scan_generation"] != burst.scan_generation or identity["target_face_index"] != burst.target_face_index:
            return self._build_preview_response(
                session,
                scanner_state=burst.scanner_state,
                status="STALE",
                reason="Stale scan response ignored.",
                timings=analyzed["timings"],
                identity=identity,
            )

        burst.request_id = identity["request_id"]

        elapsed_ms = (time.perf_counter() - burst.started_at) * 1000.0
        if elapsed_ms > self._settings.scanner_stability_timeout_seconds * 1000.0:
            burst.reset("Detection unstable. The face changed too much between frames. Adjust the cube and retry.", "RETRY")
            session.last_scan_status = "RETRY"
            session.last_scan_reason = burst.message
            return self._build_preview_response(session, scanner_state="RETRY", status="RETRY", reason=burst.message, timings=analyzed["timings"], identity=identity)

        frame_result, timings = self._infer_frame_detailed(analyzed["frame"], 0)
        if frame_result.reason == "AI service is busy. Retry shortly.":
            burst.scanner_state = "AI_BUSY"
            burst.message = "AI service is busy. Retry shortly."
            session.last_scan_status = "AI_BUSY"
            session.last_scan_reason = burst.message
            return self._build_preview_response(
                session,
                scanner_state="AI_BUSY",
                status="AI_BUSY",
                timings=timings,
                reason=burst.message,
                identity=identity,
            )

        if not frame_result.ok or frame_result.grid is None:
            burst.scanner_state = "SCANNING"
            burst.message = frame_result.reason or "Keep all 9 stickers visible."
            return self._build_preview_response(
                session,
                scanner_state="SCANNING",
                status="SCANNING",
                frame_result=frame_result,
                timings=timings,
                reason=burst.message,
                identity=identity,
            )

        observation = self._to_observation(frame_result)

        if burst.last_observation and self._observations_are_stable_match(burst.last_observation, observation):
            burst.stable_observation_count += 1
            if observation.overall_confidence >= burst.last_observation.overall_confidence:
                burst.last_observation = observation
                burst.last_observation_key = observation.key
        else:
            burst.stable_observation_count = 1
            burst.last_observation = observation
            burst.last_observation_key = observation.key

        if burst.stable_observation_count >= self._settings.scanner_required_stable_observations:
            duplicate_center = self._find_duplicate_center(session, observation.center_color)
            if duplicate_center is not None:
                burst.reset("Face already scanned. Show another face.", "DUPLICATE_FACE")
                session.last_scan_status = "DUPLICATE_FACE"
                session.last_scan_reason = burst.message
                return self._build_preview_response(
                    session,
                    scanner_state="DUPLICATE_FACE",
                    status="DUPLICATE_FACE",
                    frame_result=frame_result,
                    timings=timings,
                    reason=f"Center color {observation.center_color} already belongs to face {duplicate_center}.",
                    identity=identity,
                )

            return self._accept_observation(
                session=session,
                burst=burst,
                observation=burst.last_observation or observation,
                frame_result=frame_result,
                timings=timings,
                identity=identity,
                accepted_reason="Face detected. You may relax your hand.",
                stable_observation_count=self._settings.scanner_required_stable_observations,
            )

        burst.scanner_state = "STABLE" if burst.stable_observation_count >= 2 else "SCANNING"
        burst.message = f"Scanning {burst.stable_observation_count}/{self._settings.scanner_required_stable_observations} stable observations."
        return self._build_preview_response(
            session,
            scanner_state=burst.scanner_state,
            status=burst.scanner_state,
            frame_result=frame_result,
            timings=timings,
            reason=burst.message,
            stable_observation_count=burst.stable_observation_count,
            identity=identity,
        )

    def scan_scanner_test_face(self, session_id: str, payload: dict[str, Any] | None = None) -> ScannerSessionResponse:
        session = self._require_scanner_session(session_id)
        frames_base64 = list((payload or {}).get("framesBase64") or [])
        preview_response: ScannerPreviewResponse | None = None
        for frame_base64 in frames_base64:
            preview_response = self.observe_scanner_test_frame(session_id, {"imageBase64": frame_base64})
            if preview_response.scanner_state in {"ACCEPTED", "DUPLICATE_FACE", "AI_BUSY", "AI_UNAVAILABLE", "CAMERA_ERROR", "RETRY"}:
                break
        if preview_response is None:
            session.burst.reset("Detection unstable. Adjust the cube and retry.", "RETRY")
        return self._build_session_response(session, message=session.burst.message)

    def retry_scanner_test_face(self, session_id: str) -> ScannerSessionResponse:
        session = self._require_scanner_session(session_id)
        session.last_face_scan = None
        session.last_scan_status = "RETRY"
        session.last_scan_reason = "Retry current face."
        session.burst.scan_generation += 1
        session.burst.reset("Detection unstable. Adjust the cube and retry.", "RETRY")
        return self._build_session_response(session, message=session.burst.message)

    def reset_scanner_test_session(self, session_id: str) -> ScannerSessionResponse:
        session = self._require_scanner_session(session_id)
        session.captured_faces.clear()
        session.completed_at = None
        session.last_face_scan = None
        session.last_scan_status = "RESET"
        session.last_scan_reason = None
        session.burst.scan_generation += 1
        session.burst.target_face_index = 1
        session.burst.reset("Hold one complete face inside the frame.", "POSITION_FACE")
        return self._build_session_response(session, message=session.burst.message)

    def _analyze_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        evidence = payload.get("evidence") or {}
        image_bytes = payload.get("imageBytes")
        image_base64 = payload.get("imageBase64") or evidence.get("image_base64") or evidence.get("imageBase64")
        image_url = payload.get("imageUrl") or evidence.get("image_url") or evidence.get("imageUrl") or evidence.get("frame_url") or evidence.get("storage_key")
        frame, decode_ms = self._load_frame(image_bytes=image_bytes, image_base64=image_base64, image_url=image_url)
        timings = InferenceTimings(decode_ms=decode_ms, total_ms=decode_ms)
        return {
            "frame": frame,
            "reason": None if frame is not None else self._build_missing_image_reason(),
            "timings": timings,
        }

    def _build_missing_image_reason(self) -> str:
        if not self._model_loaded:
            return self._load_error or "AI model is unavailable."
        return "No supported image payload was provided."

    def _load_frame(self, image_bytes: bytes | bytearray | None, image_base64: str | None, image_url: str | None) -> tuple[Any, float]:
        t0 = time.perf_counter()
        if image_bytes:
            try:
                data = np.frombuffer(bytes(image_bytes), dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is not None and frame.size > 0:
                    return frame, (time.perf_counter() - t0) * 1000.0
            except Exception:
                return None, (time.perf_counter() - t0) * 1000.0

        if image_base64:
            try:
                raw = base64.b64decode(image_base64)
                data = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is not None and frame.size > 0:
                    return frame, (time.perf_counter() - t0) * 1000.0
            except Exception:
                return None, (time.perf_counter() - t0) * 1000.0

        if image_url:
            image_path = Path(image_url)
            if image_path.exists():
                frame = cv2.imread(str(image_path))
                if frame is not None and frame.size > 0:
                    return frame, (time.perf_counter() - t0) * 1000.0
        return None, (time.perf_counter() - t0) * 1000.0

    def _infer_frame_detailed(self, frame, frame_index: int) -> tuple[FrameDetectionResult, InferenceTimings]:
        timings = InferenceTimings()
        total_started = time.perf_counter()
        if not self._model_loaded or self._model is None:
            return FrameDetectionResult(
                ok=False,
                detected_stickers=0,
                frame_index=frame_index,
                reason=self._load_error or "AI model is unavailable.",
            ), timings

        if not self._inference_gate.acquire(blocking=False):
            return FrameDetectionResult(
                ok=False,
                detected_stickers=0,
                frame_index=frame_index,
                reason="AI service is busy. Retry shortly.",
            ), timings

        try:
            preprocess_started = time.perf_counter()
            timings.preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

            inference_started = time.perf_counter()
            context = self._torch.inference_mode() if self._torch is not None else nullcontext()
            with context:
                results = self._model.predict(
                    source=frame,
                    imgsz=640,
                    conf=0.35,
                    verbose=False,
                    device="cpu",
                    half=False,
                )
            timings.inference_ms = (time.perf_counter() - inference_started) * 1000.0

            postprocess_started = time.perf_counter()
            detections = self._to_detections(results)
            built = self._grid_builder.build(detections, frame.shape[1], frame.shape[0])
            timings.postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
            timings.total_ms = (time.perf_counter() - total_started) * 1000.0

            if built is None:
                return FrameDetectionResult(
                    ok=False,
                    detected_stickers=len(detections),
                    frame_index=frame_index,
                    reason="Need 9 stickers visible in a stable 3x3 layout.",
                ), timings

            return FrameDetectionResult(
                ok=True,
                grid=built.grid_matrix,
                confidence_matrix=built.confidence_matrix,
                avg_confidence=built.average_confidence,
                detected_stickers=len(detections),
                frame_index=frame_index,
                reason=None,
                ordered_boxes=built.ordered_boxes,
            ), timings
        finally:
            self._inference_gate.release()

    def _to_detections(self, results) -> list[StickerDetection]:
        detections: list[StickerDetection] = []
        if not results:
            return detections

        result0 = results[0]
        if result0.boxes is None:
            return detections

        for box in result0.boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = str(self._model.names.get(cls_id, str(cls_id))).lower()
            area = float(max(1, x2 - x1) * max(1, y2 - y1))
            detections.append(
                StickerDetection(
                    xyxy=(x1, y1, x2, y2),
                    conf=conf,
                    cls_id=cls_id,
                    cls_name=cls_name,
                    area=area,
                    center_x=(x1 + x2) / 2.0,
                    center_y=(y1 + y2) / 2.0,
                )
            )
        return detections

    def _to_observation(self, frame_result: FrameDetectionResult) -> ScannerObservation:
        grid = frame_result.grid or []
        stickers = []
        for row in range(3):
            for col in range(3):
                index = row * 3 + col
                bbox = list(frame_result.ordered_boxes[index]) if index < len(frame_result.ordered_boxes) else [0, 0, 0, 0]
                confidence = float(frame_result.confidence_matrix[row][col]) if frame_result.confidence_matrix else 0.0
                stickers.append({"color": grid[row][col], "confidence": confidence, "bbox": bbox})
        center_color = grid[1][1]
        key = f"{center_color}|{'/'.join(cell for row in grid for cell in row)}"
        return ScannerObservation(
            center_color=center_color,
            grid3x3=grid,
            stickers=stickers,
            overall_confidence=frame_result.avg_confidence,
            key=key,
        )

    def _accept_observation(
        self,
        session: ScannerSessionState,
        burst: ScannerBurstState,
        observation: ScannerObservation,
        frame_result: FrameDetectionResult,
        timings: InferenceTimings,
        identity: dict[str, Any],
        accepted_reason: str,
        stable_observation_count: int,
    ) -> ScannerPreviewResponse:
        face = ScannerSessionFace(
            center_color=observation.center_color,
            grid3x3=observation.grid3x3,
            stickers=observation.stickers,
            overall_confidence=observation.overall_confidence,
            valid_frames=stable_observation_count,
            captured_at=datetime.now(UTC),
        )
        session.captured_faces.append(face)
        session.last_face_scan = face
        session.last_scan_status = "ACCEPTED"
        session.last_scan_reason = None
        burst.reset("Face accepted. Rotate to a different center color.", "ACCEPTED")
        if len(session.captured_faces) >= 6:
            session.completed_at = datetime.now(UTC)
            burst.scanner_state = "ACCEPTED"
            burst.message = "Six-face scan completed."
        return self._build_preview_response(
            session,
            scanner_state="ACCEPTED",
            status="ACCEPTED",
            frame_result=frame_result,
            timings=timings,
            reason=accepted_reason,
            stable_observation_count=stable_observation_count,
            identity=identity,
        )

    def _observations_are_stable_match(self, left: ScannerObservation, right: ScannerObservation) -> bool:
        if left.center_color != right.center_color:
            return False

        matching_cells = 0
        for row in range(3):
            for col in range(3):
                if left.grid3x3[row][col] == right.grid3x3[row][col]:
                    matching_cells += 1

        return matching_cells >= self._settings.scanner_stable_grid_match_min_cells

    def _build_preview_response(
        self,
        session: ScannerSessionState,
        scanner_state: str,
        status: str,
        frame_result: FrameDetectionResult | None = None,
        timings: InferenceTimings | None = None,
        reason: str | None = None,
        stable_observation_count: int | None = None,
        identity: dict[str, Any] | None = None,
    ) -> ScannerPreviewResponse:
        timings = timings or InferenceTimings()
        identity = identity or self._scanner_request_identity(session, {})
        center_color = frame_result.grid[1][1] if frame_result and frame_result.ok and frame_result.grid else None
        return ScannerPreviewResponse(
            status=status,
            scannerState=scanner_state,
            scanSessionId=str(identity["scan_session_id"]),
            scanGeneration=int(identity["scan_generation"]),
            requestId=identity["request_id"],
            targetFaceIndex=int(identity["target_face_index"]),
            requestedFaceIndex=min(len(session.captured_faces) + 1, 6),
            requestedFaceLabel=self._requested_face_label(session),
            centerColor=center_color,
            grid3x3=frame_result.grid if frame_result else None,
            stickers=self._frame_stickers(frame_result),
            detectedStickers=frame_result.detected_stickers if frame_result else 0,
            confidence=frame_result.avg_confidence if frame_result and frame_result.ok else 0.0,
            inferMs=timings.inference_ms,
            decodeMs=timings.decode_ms,
            preprocessMs=timings.preprocess_ms,
            postprocessMs=timings.postprocess_ms,
            totalMs=timings.total_ms,
            stableObservationCount=stable_observation_count if stable_observation_count is not None else session.burst.stable_observation_count,
            requiredStableObservations=self._settings.scanner_required_stable_observations,
            modelVersion=self._settings.model_version,
            reason=reason,
        )

    def _build_session_response(self, session: ScannerSessionState, message: str) -> ScannerSessionResponse:
        faces = [self._to_face_result(face) for face in session.captured_faces]
        raw_sticker_state = [cell for face in session.captured_faces for row in face.grid3x3 for cell in row]
        status = "COMPLETED" if session.completed_at is not None and len(session.captured_faces) == 6 else "IN_PROGRESS"
        return ScannerSessionResponse(
            sessionId=session.session_id,
            status=status,
            scannerState=session.burst.scanner_state,
            message=message,
            scanGeneration=session.burst.scan_generation,
            requestedFaceIndex=min(len(session.captured_faces) + 1, 6),
            requestedFaceLabel=self._requested_face_label(session),
            capturedFaceCount=len(session.captured_faces),
            rawStickerCount=len(raw_sticker_state),
            orientationResolved=False,
            modelVersion=self._settings.model_version,
            startedAt=session.started_at,
            completedAt=session.completed_at,
            faces=faces,
            rawStickerState=raw_sticker_state,
            lastFaceScan=self._to_face_result(session.last_face_scan) if session.last_face_scan else None,
            lastScanStatus=session.last_scan_status,
            lastScanReason=session.last_scan_reason,
        )

    def _requested_face_label(self, session: ScannerSessionState) -> str:
        return f"Face {min(len(session.captured_faces) + 1, 6)} of 6"

    def _to_face_result(self, face: ScannerSessionFace) -> ScannerFaceResult:
        return ScannerFaceResult(
            centerColor=face.center_color,
            grid3x3=face.grid3x3,
            stickers=[ScannerStickerObservation(**sticker) for sticker in face.stickers],
            overallConfidence=face.overall_confidence,
            validFrames=face.valid_frames,
            capturedAt=face.captured_at,
        )

    def _find_duplicate_center(self, session: ScannerSessionState, center_color: str) -> int | None:
        for index, face in enumerate(session.captured_faces, start=1):
            if face.center_color == center_color:
                return index
        return None

    def _frame_stickers(self, frame_result: FrameDetectionResult | None) -> list[ScannerStickerObservation]:
        if frame_result is None or not frame_result.ok or not frame_result.grid or not frame_result.confidence_matrix or not frame_result.ordered_boxes:
            return []
        stickers: list[ScannerStickerObservation] = []
        for row in range(3):
            for col in range(3):
                index = row * 3 + col
                stickers.append(
                    ScannerStickerObservation(
                        color=frame_result.grid[row][col],
                        confidence=float(frame_result.confidence_matrix[row][col]),
                        bbox=list(frame_result.ordered_boxes[index]),
                    )
                )
        return stickers

    def _require_scanner_session(self, session_id: str) -> ScannerSessionState:
        session = self._scanner_sessions.get(session_id)
        if session is None:
            raise KeyError(f"Scanner session {session_id} was not found.")
        return session

    def _scanner_request_identity(self, session: ScannerSessionState, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = payload.get("metadata") or {}
        requested_face_index = min(len(session.captured_faces) + 1, 6)
        scan_session_id = (
            payload.get("scanSessionId")
            or metadata.get("scanSessionId")
            or session.session_id
        )
        scan_generation = payload.get("scanGeneration")
        if scan_generation is None:
            scan_generation = metadata.get("scanGeneration")
        if scan_generation is None:
            scan_generation = session.burst.scan_generation

        target_face_index = payload.get("targetFaceIndex")
        if target_face_index is None:
            target_face_index = metadata.get("targetFaceIndex")
        if target_face_index is None:
            target_face_index = requested_face_index

        request_id = payload.get("requestId")
        if request_id is None:
            request_id = metadata.get("requestId")

        return {
            "scan_session_id": str(scan_session_id),
            "scan_generation": int(scan_generation),
            "request_id": None if request_id is None else str(request_id),
            "target_face_index": int(target_face_index),
        }

    @staticmethod
    def _flatten(grid: list[list[str]] | None) -> list[str]:
        if not grid:
            return []
        return [cell for row in grid for cell in row]

    @staticmethod
    def _grid_to_state(grid: list[list[str]] | None) -> str | None:
        if not grid:
            return None
        return "".join(cell[:1].upper() for row in grid for cell in row)
