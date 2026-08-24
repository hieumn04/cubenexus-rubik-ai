from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EvidenceReference(BaseModel):
    image_url: str | None = None
    video_url: str | None = None
    frame_url: str | None = None
    storage_key: str | None = None
    image_base64: str | None = None


class AiCheckRequest(BaseModel):
    match_id: str = Field(..., alias="matchId")
    player_id: str = Field(..., alias="playerId")
    scramble_sequence: str | None = Field(default=None, alias="scrambleSequence")
    evidence: EvidenceReference | None = None
    image_url: str | None = Field(default=None, alias="imageUrl")
    image_base64: str | None = Field(default=None, alias="imageBase64")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class AnalyzeFrameRequest(BaseModel):
    image_url: str | None = Field(default=None, alias="imageUrl")
    image_base64: str | None = Field(default=None, alias="imageBase64")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class ScannerSessionStartRequest(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScannerFrameRequest(BaseModel):
    image_url: str | None = Field(default=None, alias="imageUrl")
    image_base64: str | None = Field(default=None, alias="imageBase64")
    metadata: dict[str, Any] = Field(default_factory=dict)
    scan_session_id: str | None = Field(default=None, alias="scanSessionId")
    scan_generation: int | None = Field(default=None, alias="scanGeneration")
    request_id: str | None = Field(default=None, alias="requestId")
    target_face_index: int | None = Field(default=None, alias="targetFaceIndex")

    model_config = {"populate_by_name": True}


class ScannerFaceScanRequest(BaseModel):
    frames_base64: list[str] = Field(default_factory=list, alias="framesBase64")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class AiCheckResponse(BaseModel):
    check_type: str = Field(..., alias="checkType")
    status: str
    confidence: float
    detected_cube: bool = Field(..., alias="detectedCube")
    detected_stickers: int = Field(..., alias="detectedStickers")
    grid3x3: list[list[str]] | None = None
    reason: str | None = None
    expected_scramble: str | None = Field(default=None, alias="expectedScramble")
    detected_state: str | None = Field(default=None, alias="detectedState")
    is_scramble_matched: bool | None = Field(default=None, alias="isScrambleMatched")
    is_solved: bool | None = Field(default=None, alias="isSolved")
    model_version: str = Field(..., alias="modelVersion")
    model_loaded: bool = Field(..., alias="modelLoaded")
    created_at: datetime = Field(..., alias="createdAt")

    model_config = {"populate_by_name": True}


class AnalyzeFrameResponse(BaseModel):
    status: str
    model_version: str = Field(..., alias="modelVersion")
    labels: list[str]
    notes: list[str]
    detected_stickers: int = Field(..., alias="detectedStickers")
    grid3x3: list[list[str]] | None = Field(default=None, alias="grid3x3")
    confidence: float = 0.0

    model_config = {"populate_by_name": True}


class ScannerStickerObservation(BaseModel):
    color: str
    confidence: float
    bbox: list[int]


class ScannerPreviewResponse(BaseModel):
    status: str
    scanner_state: str = Field(..., alias="scannerState")
    scan_session_id: str = Field(..., alias="scanSessionId")
    scan_generation: int = Field(default=0, alias="scanGeneration")
    request_id: str | None = Field(default=None, alias="requestId")
    target_face_index: int = Field(..., alias="targetFaceIndex")
    requested_face_index: int = Field(..., alias="requestedFaceIndex")
    requested_face_label: str = Field(..., alias="requestedFaceLabel")
    center_color: str | None = Field(default=None, alias="centerColor")
    grid3x3: list[list[str]] | None = Field(default=None, alias="grid3x3")
    stickers: list[ScannerStickerObservation] = Field(default_factory=list)
    detected_stickers: int = Field(..., alias="detectedStickers")
    confidence: float = 0.0
    infer_ms: float = Field(..., alias="inferMs")
    decode_ms: float = Field(default=0.0, alias="decodeMs")
    preprocess_ms: float = Field(default=0.0, alias="preprocessMs")
    postprocess_ms: float = Field(default=0.0, alias="postprocessMs")
    total_ms: float = Field(default=0.0, alias="totalMs")
    stable_observation_count: int = Field(default=0, alias="stableObservationCount")
    required_stable_observations: int = Field(default=3, alias="requiredStableObservations")
    model_version: str = Field(..., alias="modelVersion")
    reason: str | None = None

    model_config = {"populate_by_name": True}


class ScannerFaceResult(BaseModel):
    center_color: str = Field(..., alias="centerColor")
    grid3x3: list[list[str]] = Field(..., alias="grid3x3")
    stickers: list[ScannerStickerObservation]
    overall_confidence: float = Field(..., alias="overallConfidence")
    valid_frames: int = Field(..., alias="validFrames")
    captured_at: datetime = Field(..., alias="capturedAt")

    model_config = {"populate_by_name": True}


class ScannerSessionResponse(BaseModel):
    session_id: str = Field(..., alias="sessionId")
    status: str
    scanner_state: str = Field(..., alias="scannerState")
    message: str
    scan_generation: int = Field(default=0, alias="scanGeneration")
    requested_face_index: int = Field(..., alias="requestedFaceIndex")
    requested_face_label: str = Field(..., alias="requestedFaceLabel")
    captured_face_count: int = Field(..., alias="capturedFaceCount")
    raw_sticker_count: int = Field(..., alias="rawStickerCount")
    orientation_resolved: bool = Field(..., alias="orientationResolved")
    model_version: str = Field(..., alias="modelVersion")
    started_at: datetime = Field(..., alias="startedAt")
    completed_at: datetime | None = Field(default=None, alias="completedAt")
    faces: list[ScannerFaceResult]
    raw_sticker_state: list[str] = Field(default_factory=list, alias="rawStickerState")
    last_face_scan: ScannerFaceResult | None = Field(default=None, alias="lastFaceScan")
    last_scan_status: str | None = Field(default=None, alias="lastScanStatus")
    last_scan_reason: str | None = Field(default=None, alias="lastScanReason")

    model_config = {"populate_by_name": True}


class CubeScanFaceMapRequest(BaseModel):
    faces: dict[str, list[list[str]]]


class CubeScanValidationResponse(BaseModel):
    status: str
    color_counts: dict[str, int] = Field(..., alias="colorCounts")
    validation: dict[str, Any]

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    status: str
    service_name: str = Field(..., alias="serviceName")
    model_path: str = Field(..., alias="modelPath")
    model_exists: bool = Field(..., alias="modelExists")
    model_version: str = Field(..., alias="modelVersion")
    model_loaded: bool = Field(..., alias="modelLoaded")

    model_config = {"populate_by_name": True}
