import re
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from model.armor_pose_detector import ArmorPoseONNXDetector
from model.ultralytics_yolo_detector import UltralyticsYOLODetector
from tracker.CascadeMatchTracker.type import SingleDetectionResult, State, TrackingState


class BaseDetector:
    def __init__(self, config, pixel_world_transform, visualize=False):
        super().__init__()
        self.device = config["device"]
        self.config = config
        self.visualize = visualize
        self.pixel_world_transform = pixel_world_transform

        self.class_names = config["armor_detector"]["class_names"]
        self.class_name_to_id = {
            class_name: class_id for class_id, class_name in enumerate(self.class_names)
        }

        self.raw_name_mapping = config["armor_detector"].get("raw_name_mapping", {})
        self.color_name_mapping = {
            int(key): str(value)
            for key, value in config["armor_detector"].get(
                "color_name_mapping",
                {"0": "R", "1": "B", "2": "G", "3": "P"},
            ).items()
        }
        self.object_name_mapping = {
            str(key): str(value)
            for key, value in config["armor_detector"].get(
                "object_name_mapping",
                {"0": "7", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5"},
            ).items()
        }

        self.car_detector = UltralyticsYOLODetector(
            model_path=config["car_detector"]["weights_path"],
            img_size=config["car_detector"]["img_size"],
            max_det=config["car_detector"]["max_det"],
            conf_thres=config["car_detector"]["conf_thres"],
            iou_thres=config["car_detector"]["iou_thres"],
            device=self.device,
            classes=config["car_detector"].get("classes"),
        )

        self.armor_detector = ArmorPoseONNXDetector(
            model_path=config["armor_detector"]["weights_path"],
            conf_thres=config["armor_detector"]["conf_thres"],
            iou_thres=config["armor_detector"]["iou_thres"],
            max_det=config["armor_detector"]["max_det"],
            device=self.device,
        )
        self.raw_armor_class_names = self.armor_detector.get_class_names()
        self.detect_frame_index = 0
        self.detect_full_frame_interval = max(
            1, int(config["tracker"].get("detect_full_frame_interval", 3))
        )
        self.detect_roi_expand_ratio = float(
            config["tracker"].get("detect_roi_expand_ratio", 0.2)
        )
        self.detect_roi_min_margin = int(
            config["tracker"].get("detect_roi_min_margin", 24)
        )
        self.detect_roi_img_size = int(
            config["tracker"].get(
                "detect_roi_img_size", config["car_detector"]["img_size"]
            )
        )
        self.min_color_conf = float(config["tracker"].get("min_color_conf", 0.45))
        self.min_color_margin = float(config["tracker"].get("min_color_margin", 0.08))

        self.tracks = [
            TrackingState(class_id=class_id, name=class_name, max_frames=30)
            for class_id, class_name in enumerate(self.class_names)
        ]

    def _parse_raw_class(self, raw_class_name: str) -> Optional[Tuple[str, str]]:
        match = re.fullmatch(r"(s\d+)_o(\d+)", raw_class_name)
        if not match:
            return None
        return match.groups()

    def _map_detection_to_class_name(self, det: dict) -> Optional[str]:
        raw_class_name = det["class_name"]
        if raw_class_name in self.raw_name_mapping:
            mapped_name = self.raw_name_mapping[raw_class_name]
            if mapped_name in (None, "", "ignore", "IGNORE"):
                return None
            return str(mapped_name)

        if raw_class_name in self.class_name_to_id:
            return raw_class_name

        parsed = self._parse_raw_class(raw_class_name)
        if parsed is None:
            return None

        _, object_token = parsed
        color_prefix = self._resolve_color_prefix(det)
        object_suffix = self.object_name_mapping.get(object_token)
        if color_prefix not in ("B", "R") or not object_suffix:
            return None

        mapped_name = f"{color_prefix}{object_suffix}"
        return mapped_name if mapped_name in self.class_name_to_id else None

    def _resolve_color_prefix(self, det: dict) -> Optional[str]:
        color_scores = det.get("color_scores") or []
        if not color_scores:
            return self.color_name_mapping.get(det.get("color_id", -1))

        color_scores_np = np.asarray(color_scores, dtype=np.float32)
        best_idx = int(np.argmax(color_scores_np))
        best_score = float(color_scores_np[best_idx])
        second_score = 0.0
        if color_scores_np.size > 1:
            second_score = float(np.partition(color_scores_np, -2)[-2])

        if best_score < self.min_color_conf:
            return None
        if best_score - second_score < self.min_color_margin:
            return None
        return self.color_name_mapping.get(best_idx)

    def xyxy2pos3d(self, xyxy):
        x1, y1, x2, y2 = xyxy
        xo, yo = (x1 + x2) * 0.5, y2
        if (
            position_3d := self.pixel_world_transform.pixel_to_world([xo, yo])
        ) is None:
            position_3d = np.array([0.0, 0.0, 0.0])
        return position_3d.tolist()

    @staticmethod
    def _sanitize_box(box: List[float], width: int, height: int) -> Optional[List[int]]:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _expand_box(
        self, box: List[float], width: int, height: int
    ) -> Optional[List[int]]:
        sanitized = self._sanitize_box(box, width, height)
        if sanitized is None:
            return None

        x1, y1, x2, y2 = sanitized
        box_w = x2 - x1
        box_h = y2 - y1
        margin_x = max(self.detect_roi_min_margin, int(box_w * self.detect_roi_expand_ratio))
        margin_y = max(self.detect_roi_min_margin, int(box_h * self.detect_roi_expand_ratio))
        return self._sanitize_box(
            [x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y],
            width,
            height,
        )

    @staticmethod
    def _compute_iou(box_a: List[int], box_b: List[int]) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        denom = area_a + area_b - inter_area
        return inter_area / denom if denom > 0 else 0.0

    def _deduplicate_car_candidates(
        self, candidates: List[Dict]
    ) -> List[Dict]:
        kept: List[Dict] = []
        for candidate in sorted(candidates, key=lambda item: item["conf"], reverse=True):
            if any(
                self._compute_iou(candidate["bbox"], existing["bbox"]) > 0.6
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return kept

    def _get_track_roi_hints(self, width: int, height: int) -> List[Dict]:
        roi_hints: List[Dict] = []
        for track in self.tracks:
            if track.state not in (State.CONFIRMED, State.TENTATIVE, State.LOST):
                continue
            if track.car_box is None:
                continue

            roi_box = self._expand_box(track.car_box, width, height)
            if roi_box is None:
                continue

            roi_hints.append(
                {
                    "roi_box": roi_box,
                    "track_class_id": track.class_id,
                    "track_bot_id": track.bot_id if track.bot_id >= 0 else None,
                }
            )
        return roi_hints

    def _predict_cars_full_frame(self, img: np.ndarray) -> List[Dict]:
        img_height, img_width = img.shape[:2]
        car_detections, _ = self.car_detector.predict(img)
        candidates: List[Dict] = []
        for cls_id, xyxy, conf in car_detections:
            bbox = self._sanitize_box(xyxy, img_width, img_height)
            if bbox is None:
                continue
            candidates.append(
                {
                    "class_id": cls_id,
                    "bbox": bbox,
                    "conf": float(conf),
                    "track_bot_id": None,
                }
            )
        return candidates

    def _predict_cars_from_rois(self, img: np.ndarray) -> List[Dict]:
        img_height, img_width = img.shape[:2]
        roi_hints = self._get_track_roi_hints(img_width, img_height)
        if not roi_hints:
            return []

        roi_images = []
        valid_hints = []
        for hint in roi_hints:
            x1, y1, x2, y2 = hint["roi_box"]
            roi_img = img[y1:y2, x1:x2]
            if roi_img.size == 0:
                continue
            roi_images.append(roi_img)
            valid_hints.append(hint)

        if not roi_images:
            return []

        roi_detections_batch, _ = self.car_detector.predict_batch(
            roi_images,
            img_size_override=self.detect_roi_img_size,
        )

        candidates: List[Dict] = []
        for hint, roi_detections in zip(valid_hints, roi_detections_batch):
            roi_x1, roi_y1, _, _ = hint["roi_box"]
            for cls_id, xyxy, conf in roi_detections:
                local_box = self._sanitize_box(
                    xyxy,
                    hint["roi_box"][2] - hint["roi_box"][0],
                    hint["roi_box"][3] - hint["roi_box"][1],
                )
                if local_box is None:
                    continue

                bbox = self._sanitize_box(
                    [
                        local_box[0] + roi_x1,
                        local_box[1] + roi_y1,
                        local_box[2] + roi_x1,
                        local_box[3] + roi_y1,
                    ],
                    img_width,
                    img_height,
                )
                if bbox is None:
                    continue

                candidates.append(
                    {
                        "class_id": cls_id,
                        "bbox": bbox,
                        "conf": float(conf),
                        "track_bot_id": hint["track_bot_id"],
                    }
                )
        return candidates

    def _get_car_candidates(self, img: np.ndarray) -> List[Dict]:
        self.detect_frame_index += 1
        run_full_frame = (
            self.detect_frame_index == 1
            or self.detect_full_frame_interval <= 1
            or self.detect_frame_index % self.detect_full_frame_interval == 0
        )

        candidates = (
            self._predict_cars_full_frame(img)
            if run_full_frame
            else self._predict_cars_from_rois(img)
        )
        if not candidates and not run_full_frame:
            candidates = self._predict_cars_full_frame(img)
        return self._deduplicate_car_candidates(candidates)

    def _select_armor_detection(self, armor_detections: List[dict]):
        if not armor_detections:
            return None, -1, ""

        best_any = max(armor_detections, key=lambda det: det["confidence"])
        mapped_candidates = []
        for det in armor_detections:
            mapped_name = self._map_detection_to_class_name(det)
            if mapped_name is None:
                continue
            mapped_candidates.append(
                (self.class_name_to_id[mapped_name], mapped_name, det)
            )

        if not mapped_candidates:
            return best_any, -1, best_any["class_name"]

        class_id, mapped_name, det = max(
            mapped_candidates, key=lambda item: item[2]["confidence"]
        )
        return det, class_id, mapped_name

    def detect(
        self, img
    ) -> tuple[List[SingleDetectionResult], Optional[np.ndarray]]:
        detections = []
        img_height, img_width = img.shape[:2]
        car_candidates = self._get_car_candidates(img)
        if not car_candidates:
            return [], img.copy()

        crop_imgs = []
        car_boxes = []
        car_confs = []
        car_bot_ids = []

        for idx, candidate in enumerate(car_candidates):
            car_box = candidate["bbox"]
            x1, y1, x2, y2 = car_box
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            car_boxes.append(car_box)
            car_confs.append(float(candidate["conf"]))
            car_bot_ids.append(
                candidate["track_bot_id"]
                if candidate["track_bot_id"] is not None
                else idx
            )
            crop_imgs.append(crop)

        if not crop_imgs:
            return [], img.copy()

        armor_detections_batch, _ = self.armor_detector.predict_batch(crop_imgs)

        for bot_id, car_box, car_conf, armor_detections in zip(
            car_bot_ids, car_boxes, car_confs, armor_detections_batch
        ):
            position_3d = self.xyxy2pos3d(car_box)
            selected_armor, class_id, mapped_class_name = self._select_armor_detection(
                armor_detections
            )

            armor_box = [0.0, 0.0, 0.0, 0.0]
            armor_corners = None
            class_conf = 0.0

            raw_class_name = ""
            if selected_armor is not None:
                armor_box = selected_armor["bbox"]
                armor_corners = selected_armor["keypoints"]
                class_conf = float(selected_armor["confidence"])
                raw_class_name = selected_armor["class_name"]

            detections.append(
                SingleDetectionResult(
                    class_id=class_id,
                    class_conf=class_conf,
                    bot_id=bot_id,
                    car_box=car_box,
                    armor_box=armor_box,
                    car_conf=car_conf,
                    pos_3d=position_3d,
                    time_stamp=time.time(),
                    raw_class_name=raw_class_name if raw_class_name else mapped_class_name,
                    armor_corners=armor_corners,
                )
            )

        vis_img = None
        if self.visualize:
            vis_img = self._get_visualized_img(img, detections)
        return detections, vis_img

    def _get_visualized_img(self, img, detections):
        vis_img = img.copy()
        for detection in detections:
            x1_car, y1_car, x2_car, y2_car = map(int, detection.car_box)
            cv2.rectangle(vis_img, (x1_car, y1_car), (x2_car, y2_car), (0, 255, 0), 2)
            cv2.putText(
                vis_img,
                f"Car: {detection.car_conf:.2f} ID:{detection.bot_id}",
                (x1_car, max(20, y1_car - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            armor_box = detection.armor_box
            if len(armor_box) != 4:
                continue

            x1_armor, y1_armor, x2_armor, y2_armor = armor_box
            if x2_armor <= x1_armor or y2_armor <= y1_armor:
                continue

            x1_armor = int(x1_armor + detection.car_box[0])
            y1_armor = int(y1_armor + detection.car_box[1])
            x2_armor = int(x2_armor + detection.car_box[0])
            y2_armor = int(y2_armor + detection.car_box[1])

            if 0 <= detection.class_id < len(self.class_names):
                label = f"{self.class_names[detection.class_id]}: {detection.class_conf:.2f}"
                color = (255, 0, 0) if detection.class_id < 6 else (0, 0, 255)
            elif detection.raw_class_name:
                label = f"{detection.raw_class_name}: {detection.class_conf:.2f}"
                color = (0, 255, 255)
            else:
                label = f"Armor: {detection.class_conf:.2f}"
                color = (0, 255, 255)

            cv2.rectangle(
                vis_img, (x1_armor, y1_armor), (x2_armor, y2_armor), color, 2
            )
            cv2.putText(
                vis_img,
                label,
                (x1_armor, max(20, y1_armor - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            if detection.armor_corners and len(detection.armor_corners) == 8:
                corners = np.array(detection.armor_corners, dtype=np.float32).reshape(
                    -1, 2
                )
                corners[:, 0] += detection.car_box[0]
                corners[:, 1] += detection.car_box[1]
                corners = corners.astype(np.int32)
                cv2.polylines(vis_img, [corners], isClosed=True, color=color, thickness=2)

        return vis_img


if __name__ == "__main__":
    import yaml
    from transform.ray_renderer import PixelToWorld

    config = yaml.safe_load(open("config/params.yaml", "r"))
    pixel_world_transform = PixelToWorld.build_from_config(config)
    detector = BaseDetector(config, pixel_world_transform, visualize=True)

    img = cv2.imread("demo/demo1.jpg")
    result, vis_img = detector.detect(img)
    print(result)
    if vis_img is not None:
        cv2.imshow("Detection Result", vis_img)
        cv2.waitKey(0)
