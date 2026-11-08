import ast
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch


class ArmorPoseONNXDetector:
    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 5,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        if device == "cuda" and ort.get_device() == "GPU" and torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        batch_dim = input_shape[0]
        self.supports_batch = not isinstance(batch_dim, int) or batch_dim != 1
        self.max_batch_size = batch_dim if isinstance(batch_dim, int) and batch_dim > 1 else None
        self.input_height = int(input_shape[2])
        self.input_width = int(input_shape[3])

        meta = self.session.get_modelmeta().custom_metadata_map
        self.raw_class_names = self._parse_class_names(meta.get("names", "{}"))

    @staticmethod
    def _parse_class_names(raw_names: str) -> Dict[int, str]:
        try:
            parsed = ast.literal_eval(raw_names)
        except (SyntaxError, ValueError):
            return {}
        return {int(key): str(value) for key, value in parsed.items()}

    def get_class_names(self) -> Dict[int, str]:
        return self.raw_class_names

    def preprocess(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        h, w = image.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h))
        pad_w = (self.input_width - new_w) / 2.0
        pad_h = (self.input_height - new_h) / 2.0

        left = int(np.floor(pad_w))
        right = int(np.ceil(pad_w))
        top = int(np.floor(pad_h))
        bottom = int(np.ceil(pad_h))

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor, (scale, left, top)

    @staticmethod
    def _clip_bbox(bbox: List[float], width: int, height: int) -> List[float]:
        x1, y1, x2, y2 = bbox
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    def _scale_back_bbox(
        self, bbox: np.ndarray, scale: float, pad_left: float, pad_top: float, width: int, height: int
    ) -> List[float]:
        x1, y1, x2, y2 = bbox.tolist()
        bbox = [
            (x1 - pad_left) / scale,
            (y1 - pad_top) / scale,
            (x2 - pad_left) / scale,
            (y2 - pad_top) / scale,
        ]
        return self._clip_bbox(bbox, width, height)

    def _scale_back_keypoints(
        self, keypoints: np.ndarray, scale: float, pad_left: float, pad_top: float, width: int, height: int
    ) -> List[float]:
        scaled = []
        for x, y in keypoints.reshape(-1, 2):
            x = (float(x) - pad_left) / scale
            y = (float(y) - pad_top) / scale
            scaled.extend(
                [
                    max(0.0, min(float(width), x)),
                    max(0.0, min(float(height), y)),
                ]
            )
        return scaled

    def nms(self, detections: List[dict]) -> List[dict]:
        if not detections:
            return []

        boxes = np.array([det["bbox"] for det in detections], dtype=np.float32)
        confs = np.array([det["confidence"] for det in detections], dtype=np.float32)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(detections[i])

            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
                boxes[order[1:], 3] - boxes[order[1:], 1]
            )
            iou = inter / np.maximum(area_i + area_rest - inter, 1e-6)

            inds = np.where(iou <= self.iou_thres)[0]
            order = order[inds + 1]

        return keep[: self.max_det]

    def _decode_output(
        self,
        output: np.ndarray,
        image: np.ndarray,
        scale: float,
        pad_left: float,
        pad_top: float,
    ) -> List[dict]:
        detections = []
        height, width = image.shape[:2]
        for row in output:
            confidence = float(row[4])
            if confidence < self.conf_thres:
                continue

            raw_class_id = int(row[5])
            bbox = self._scale_back_bbox(
                row[:4], scale, pad_left, pad_top, width, height
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            color_scores = row[6:10].astype(np.float32).tolist()
            keypoints = self._scale_back_keypoints(
                row[10:18], scale, pad_left, pad_top, width, height
            )

            detections.append(
                {
                    "class_id": raw_class_id,
                    "class_name": self.raw_class_names.get(
                        raw_class_id, f"class_{raw_class_id}"
                    ),
                    "bbox": bbox,
                    "confidence": confidence,
                    "color_scores": color_scores,
                    "color_id": int(np.argmax(color_scores)),
                    "keypoints": keypoints,
                }
            )

        detections.sort(key=lambda det: det["confidence"], reverse=True)
        return self.nms(detections)

    def predict(self, image: np.ndarray) -> Tuple[List[dict], None]:
        tensor, (scale, pad_left, pad_top) = self.preprocess(image)
        output = self.session.run(
            [self.output_name], {self.input_name: tensor[None, ...]}
        )[0][0]
        return self._decode_output(output, image, scale, pad_left, pad_top), None

    def predict_batch(self, images: List[np.ndarray]) -> Tuple[List[List[dict]], None]:
        if not images:
            return [], None

        if len(images) == 1 or not self.supports_batch:
            return [self.predict(image)[0] for image in images], None

        if self.max_batch_size is not None and len(images) > self.max_batch_size:
            all_detections: List[List[dict]] = []
            for start in range(0, len(images), self.max_batch_size):
                chunk_detections, _ = self.predict_batch(
                    images[start : start + self.max_batch_size]
                )
                all_detections.extend(chunk_detections)
            return all_detections, None

        tensors = []
        metas = []
        for image in images:
            tensor, meta = self.preprocess(image)
            tensors.append(tensor)
            metas.append(meta)

        batch_tensor = np.stack(tensors, axis=0)
        outputs = self.session.run([self.output_name], {self.input_name: batch_tensor})[0]
        if outputs.ndim == 2:
            outputs = outputs[None, ...]

        all_detections = []
        for image, output, (scale, pad_left, pad_top) in zip(images, outputs, metas):
            all_detections.append(
                self._decode_output(output, image, scale, pad_left, pad_top)
            )
        return all_detections, None
