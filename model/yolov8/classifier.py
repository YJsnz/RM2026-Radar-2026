import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional

class YOLOv8ONNXClassifier:
    def __init__(
        self,
        model_path: str,
        img_size: Tuple[int, int] = (224, 224),
        conf_thres: float = 0.45,
        device: str = 'cuda',
        visualize: bool = False
    ):
        self.model_path = model_path
        self.input_size = img_size
        self.conf_thres = conf_thres
        self.device = device
        self.visualize = visualize
        
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
        if len(input_shape) == 4:
            self.input_size = (input_shape[2], input_shape[3])
            print(f"Classification model detected, using input size: {self.input_size}")
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 调整图像大小
        resized = cv2.resize(img, self.input_size)
        
        # 转换为RGB格式
        if img.shape[2] == 3:
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = resized
        
        # 转换为模型输入格式
        input_tensor = rgb_img.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # 增加批次维度
        input_tensor = input_tensor.astype(np.float32) / 255.0  # 归一化
        
        return input_tensor
    
    def predict(self, img: np.ndarray) -> Tuple[List[Tuple[int, List[float], float]], Optional[np.ndarray]]:
        """预测图像"""
        # 预处理
        input_tensor = self.preprocess(img)
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 后处理
        detections = self.postprocess(outputs, img.shape)
        
        # 可视化
        vis_img = None
        if self.visualize and detections:
            vis_img = self.visualize_results(img, detections)
        
        return detections, vis_img
    
    def postprocess(self, outputs: np.ndarray, img_shape: Tuple[int, int]) -> List[Tuple[int, List[float], float]]:
        """后处理模型输出"""
        detections = []
        for output in outputs:
            # 分类模型输出格式: [batch, num_classes]
            if len(output.shape) == 2:
                class_probs = output[0]
                # 分类模型返回7个类别的概率，取前4个作为机器人编号1-4
                # 找到前4个类别中概率最高的索引
                top4_probs = class_probs[:4]
                class_id = np.argmax(top4_probs)
                conf = top4_probs[class_id]
                
                if conf >= self.conf_thres:
                    # 分类模型返回整个图像的分类结果
                    # 使用整个图像作为边界框
                    h, w = img_shape[:2]
                    detections.append((class_id, [0, 0, w, h], float(conf)))
        return detections
    
    def visualize_results(self, image, detections, class_names=None):
        """可视化分类结果"""
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # 默认类别名称
        if class_names is None:
            class_names = {i: f'Class_{i}' for i in range(10)}  # 根据实际类别数调整

        # 颜色列表
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0)
        ]

        for detection in detections:
            cls_id, bbox, conf = detection
            x1, y1, x2, y2 = bbox

            # 选择颜色
            color = colors[cls_id % len(colors)]

            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 添加标签
            label = f"{class_names.get(cls_id, f'Class_{cls_id}')}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # 绘制标签背景
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # 绘制标签文本
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        return vis_image
    
    def predict_batch(self, images):
        """批量预测图像"""
        all_detections = []
        for image in images:
            detections, _ = self.predict(image)
            all_detections.append(detections)
        return all_detections
