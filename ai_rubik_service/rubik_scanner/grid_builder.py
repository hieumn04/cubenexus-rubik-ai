from __future__ import annotations

from typing import Sequence

from .models import GridBuildResult, StickerDetection


class SpatialGridBuilder:
    def __init__(self, min_confidence: float) -> None:
        self._min_confidence = float(min_confidence)

    def build(self, detections: Sequence[StickerDetection], frame_width: int, frame_height: int) -> GridBuildResult | None:
        filtered = [det for det in detections if det.conf >= self._min_confidence]
        if len(filtered) < 7:
            return None

        selected = self._select_best_nine(filtered, frame_width, frame_height)
        if len(selected) != 9:
            return None

        selected.sort(key=lambda item: item.center_y)
        rows = [selected[0:3], selected[3:6], selected[6:9]]

        ordered_boxes = []
        ordered_colors = []
        ordered_confidences = []
        grid_matrix = []
        confidence_matrix = []

        for row in rows:
            row_sorted = sorted(row, key=lambda item: item.center_x)
            grid_row = []
            conf_row = []
            for det in row_sorted:
                ordered_boxes.append(det.xyxy)
                ordered_colors.append(det.cls_name)
                ordered_confidences.append(det.conf)
                grid_row.append(det.cls_name)
                conf_row.append(det.conf)
            grid_matrix.append(grid_row)
            confidence_matrix.append(conf_row)

        if len(ordered_boxes) != 9:
            return None

        avg_conf = sum(ordered_confidences) / len(ordered_confidences)
        return GridBuildResult(
            ordered_boxes=ordered_boxes,
            ordered_colors=ordered_colors,
            ordered_confidences=ordered_confidences,
            grid_matrix=grid_matrix,
            confidence_matrix=confidence_matrix,
            average_confidence=avg_conf,
        )

    def _select_best_nine(self, detections: Sequence[StickerDetection], frame_width: int, frame_height: int) -> list[StickerDetection]:
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        def rank(det: StickerDetection) -> tuple[float, float, float]:
            center_distance = abs(det.center_x - frame_center_x) + abs(det.center_y - frame_center_y)
            return (
                det.conf,
                -center_distance,
                -abs(det.area),
            )

        return sorted(detections, key=rank, reverse=True)[:9]
