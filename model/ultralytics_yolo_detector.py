import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _ensure_ultralytics_cache_dir() -> None:
    cache_dir = os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
    os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", cache_dir)
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp")
    os.makedirs(cache_dir, exist_ok=True)


class UltralyticsYOLODetector:
    def __init__(
        self,
        model_path: str,
        img_size: int = 1280,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 10,
        device: str = "cuda",
        classes: Optional[List[int]] = None,
    ):
        _ensure_ultralytics_cache_dir()

        self.model_path = model_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.device = self._resolve_device(device)

        self.model = YOLO(model_path)
        if Path(model_path).suffix != ".engine":
            self.model = self.model.to(self.device)
            self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[Tuple[int, List[float], float]], None]:
        detections_batch, _ = self.predict_batch([image])
        return detections_batch[0], None

    def _run_predict(
        self, images, img_size_override: Optional[int] = None
    ):
        return self.model.predict(
            images,
            imgsz=img_size_override or self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

    @staticmethod
    def _extract_detections(result) -> List[Tuple[int, List[float], float]]:
        detections: List[Tuple[int, List[float], float]] = []
        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        clss = result.boxes.cls.detach().cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = box.tolist()
            detections.append((int(cls_id), [x1, y1, x2, y2], float(conf)))

        return detections

    def predict_batch(
        self,
        images: List[np.ndarray],
        img_size_override: Optional[int] = None,
    ) -> Tuple[List[List[Tuple[int, List[float], float]]], None]:
        if not images:
            return [], None

        results = self._run_predict(images, img_size_override=img_size_override)
        all_detections = [self._extract_detections(result) for result in results]
        return all_detections, None

    @staticmethod
    def visualize_results(
        image: np.ndarray,
        detections: List[Tuple[int, List[float], float]],
        label: str = "car",
    ) -> np.ndarray:
        vis_image = image.copy()
        for _, bbox, conf in detections:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis_image,
                f"{label}: {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return vis_image
