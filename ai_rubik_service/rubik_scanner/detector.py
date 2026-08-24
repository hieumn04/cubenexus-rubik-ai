from __future__ import annotations

import cv2
from ultralytics import YOLO

from .grid_builder import SpatialGridBuilder
from .models import FrameDetectionResult, StickerDetection


class RubikStickerDetector:
    def __init__(self, model_path: str, imgsz: int, conf: float, device: str, infer_every: int) -> None:
        self._model = YOLO(model_path)
        try:
            self._model.fuse()
        except Exception:
            pass

        self._imgsz = imgsz
        self._conf = conf
        self._device = device
        self._infer_every = max(1, int(infer_every))
        self._frame_counter = 0
        self._last_result = None
        self._grid_builder = SpatialGridBuilder(min_confidence=conf)
        self.last_infer_ms = 0.0

    def detect(self, frame, frame_index: int) -> FrameDetectionResult:
        self._frame_counter += 1
        if self._frame_counter % self._infer_every == 0 or self._last_result is None:
            t0 = cv2.getTickCount()
            self._last_result = self._model.predict(
                source=frame,
                imgsz=self._imgsz,
                conf=self._conf,
                verbose=False,
                device=self._device,
                half=False,
            )
            elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
            self.last_infer_ms = elapsed * 1000.0

        detections = self._to_detections(self._last_result)
        built = self._grid_builder.build(detections, frame.shape[1], frame.shape[0])
        if built is None:
            return FrameDetectionResult(
                ok=False,
                detected_stickers=len(detections),
                frame_index=frame_index,
                reason="Need 9 stickers",
            )

        return FrameDetectionResult(
            ok=True,
            grid=built.grid_matrix,
            confidence_matrix=built.confidence_matrix,
            avg_confidence=built.average_confidence,
            detected_stickers=len(detections),
            frame_index=frame_index,
            ordered_boxes=built.ordered_boxes,
        )

    def draw_debug(self, frame, frame_result: FrameDetectionResult) -> None:
        for box, color in zip(frame_result.ordered_boxes, [cell for row in (frame_result.grid or []) for cell in row]):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, color, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, color, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def _to_detections(self, results) -> list[StickerDetection]:
        detections: list[StickerDetection] = []
        if not results:
            return detections

        r0 = results[0]
        if r0.boxes is None:
            return detections

        for box in r0.boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = str(self._model.names.get(cls_id, str(cls_id))).lower()
            area = float(max(1, x2 - x1) * max(1, y2 - y1))
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            detections.append(
                StickerDetection(
                    xyxy=(x1, y1, x2, y2),
                    conf=conf,
                    cls_id=cls_id,
                    cls_name=cls_name,
                    area=area,
                    center_x=center_x,
                    center_y=center_y,
                )
            )

        return detections
