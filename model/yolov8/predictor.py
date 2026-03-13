import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict, Optional

class YOLOv8ONNXDetector:
    def __init__(
        self,
        model_path: str,
        img_size: Tuple[int, int] = (640, 640),
        conf_thres: float = 0.45,
        iou_thres: float = 0.45,
        max_det: int = 100,
        device: str = 'cuda'
    ):
        self.model_path = model_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        
        # 选择设备
        if device == 'cuda' and ort.get_device() == 'GPU':
            self.provider = ['CUDAExecutionProvider']
        else:
            self.provider = ['CPUExecutionProvider']
        
        # 加载模型
        self.session = ort.InferenceSession(model_path, providers=self.provider)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 获取模型实际输入尺寸
        input_shape = self.session.get_inputs()[0].shape
        self.is_classification = False
        
        # 检查是否为分类模型（固定224x224输入）
        if len(input_shape) == 4 and input_shape[2] == 224 and input_shape[3] == 224:
            self.is_classification = True
            self.img_size = (224, 224)
            print(f"Classification model detected, using fixed input size: {self.img_size}")
        else:
            print(f"Detection model detected, using input size: {self.img_size}")
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float, float]]:
        """预处理图像"""
        h, w = img.shape[:2]
        
        # 处理img_size参数
        if isinstance(self.img_size, int):
            img_h = img_w = self.img_size
        else:
            img_h, img_w = self.img_size
        
        # 计算缩放比例
        r_w = img_w / w
        r_h = img_h / h
        r = min(r_w, r_h)
        
        # 计算新的尺寸
        new_w = int(w * r)
        new_h = int(h * r)
        
        # 调整图像大小
        resized = cv2.resize(img, (new_w, new_h))
        
        # 计算填充
        pad_w = (img_w - new_w) // 2
        pad_h = (img_h - new_h) // 2
        
        # 填充图像
        padded = cv2.copyMakeBorder(
            resized, pad_h, img_h - new_h - pad_h,
            pad_w, img_w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # 转换为模型输入格式
        input_tensor = padded.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # 增加批次维度
        input_tensor = input_tensor.astype(np.float32) / 255.0  # 归一化
        
        # 对于分类模型，添加额外的预处理
        if self.is_classification:
            # 确保图像是RGB格式
            if img.shape[2] == 3:
                # 分类模型可能需要RGB格式
                pass
        
        # 返回更多信息用于后处理
        return input_tensor, (r, pad_w, pad_h, img_w, img_h)
    
    def postprocess(
        self, 
        outputs: np.ndarray, 
        scale_info: Tuple[float, float, float, float, float], 
        img_shape: Tuple[int, int]
    ) -> List[Tuple[int, List[float], float]]:
        """后处理模型输出"""
        if self.is_classification:
            # 处理分类模型输出
            detections = []
            for output in outputs:
                # 分类模型输出格式: [batch, num_classes]
                if len(output.shape) == 2:
                    class_probs = output[0]
                    

                    
                    class_id = np.argmax(class_probs)
                    conf = class_probs[class_id]
                    
                    if conf >= self.conf_thres:
                        # 分类模型返回整个图像的分类结果
                        # 使用整个图像作为边界框
                        h, w = img_shape[:2]
                        detections.append((class_id, [0, 0, w, h], float(conf)))
            return detections[:self.max_det]
        else:
            # 处理检测模型输出
            r, pad_w, pad_h, img_w, img_h = scale_info
            h, w = img_shape[:2]
            
            # 解析输出
            detections = []
            for output in outputs:
                # 检查输出形状
                if len(output.shape) == 3:
                    # 标准YOLO输出格式: [batch, num_detections, 5+] (x1, y1, x2, y2, conf, class_id, ...)
                    for det in output[0]:
                        # 确保检测结果有效
                        if len(det) >= 5:
                            conf = det[4]
                            if conf < self.conf_thres:
                                continue
                            
                            class_id = int(det[5]) if len(det) > 5 else 0
                            
                            # YOLOv8输出格式：[center_x, center_y, width, height]
                            center_x, center_y, width, height = det[:4]
                            
                            # 检查坐标是否已经归一化（0-1范围）
                            if max(center_x, center_y, width, height) <= 1.0:
                                # 归一化坐标，转换为模型输入尺寸
                                center_x = center_x * img_w
                                center_y = center_y * img_h
                                width = width * img_w
                                height = height * img_h
                            
                            # 转换为[x1, y1, x2, y2]格式
                            x1 = center_x - width / 2
                            y1 = center_y - height / 2
                            x2 = center_x + width / 2
                            y2 = center_y + height / 2
                            
                            # 从模型输入尺寸转换到原始图像尺寸
                            x1 = (x1 - pad_w) / r
                            y1 = (y1 - pad_h) / r
                            x2 = (x2 - pad_w) / r
                            y2 = (y2 - pad_h) / r
                            
                            # 确保边界框有效（x2 > x1, y2 > y1）
                            if x2 <= x1:
                                x1, x2 = x2, x1
                            if y2 <= y1:
                                y1, y2 = y2, y1
                            
                            # 确保边界框在图像范围内
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(w, x2)
                            y2 = min(h, y2)
                            
                            # 确保边界框仍然有效
                            if x2 > x1 and y2 > y1:
                                detections.append((class_id, [x1, y1, x2, y2], float(conf)))
            
            # 按置信度排序
            detections.sort(key=lambda x: x[2], reverse=True)
            
            # 应用NMS
            filtered_detections = self.nms(detections)
            
            # 限制最大检测数量
            return filtered_detections[:self.max_det]
    
    def nms(self, detections: List[Tuple[int, List[float], float]]) -> List[Tuple[int, List[float], float]]:
        """非极大值抑制"""
        if not detections:
            return []
        
        boxes = np.array([det[1] for det in detections])
        confs = np.array([det[2] for det in detections])
        class_ids = np.array([det[0] for det in detections])
        
        # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 按置信度排序
        order = confs.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(detections[i])
            
            # 计算与其他边界框的交集
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # 计算IoU
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 保留IoU小于阈值的边界框
            inds = np.where(iou <= self.iou_thres)[0]
            order = order[inds + 1]
        
        return keep
    
    def predict(self, img: np.ndarray) -> Tuple[List[Tuple[int, List[float], float]], Optional[np.ndarray]]:
        """预测图像"""
        # 预处理
        input_tensor, scale_info = self.preprocess(img)
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 调试信息
        if self.is_classification:
            print(f"Classification model output shape: {[o.shape for o in outputs]}")
            if outputs:
                print(f"First output sample: {outputs[0][0][:10]}")
        
        # 后处理
        detections = self.postprocess(outputs, scale_info, img.shape)
        
        return detections, None
    
    def predict_batch(self, imgs: List[np.ndarray]) -> Tuple[List[List[Tuple[int, List[float], float]]], Optional[List[np.ndarray]]]:
        """批量预测图像"""
        results = []
        for img in imgs:
            detections, _ = self.predict(img)
            results.append(detections)
        return results, None
