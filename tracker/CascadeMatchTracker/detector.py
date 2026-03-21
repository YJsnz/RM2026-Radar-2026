from model.pfa_yolov5.predictor import YOLOv5Detector
from model.yolov12.predictor_with_tracker import PredictorWithTracker
from model.yolov8.predictor import YOLOv8ONNXDetector
from model.yolov8.classifier import YOLOv8ONNXClassifier
from tracker.CascadeMatchTracker.type import TrackingState, SingleDetectionResult
from typing import List, Optional
import numpy as np
import cv2
import time
from model.armor_detector import TwoStepArmorDetectorClassifier


class BaseDetector:

    def __init__(self, config, pixel_world_transform, visualize=False):
        super().__init__()
        self.device = config["device"]
        self.class_names = config["armor_detector"]["class_names"]
        self.config = config

        # 使用YOLOv8 ONNX模型
        yolov8_onnx_path = "yolov8/ONNX/"
        
        # 车辆检测器 (使用RM2025.onnx)
        self.car_detector = YOLOv8ONNXDetector(
            model_path=yolov8_onnx_path + "RM2025.onnx",
            img_size=config["car_detector"]["img_size"],
            max_det=config["car_detector"]["max_det"],
            conf_thres=config["car_detector"]["conf_thres"],
            iou_thres=config["car_detector"]["iou_thres"],
            device=self.device,
        )

        # 装甲板检测器 (使用armor_yolo.onnx)
        self.armor_detector = YOLOv8ONNXDetector(
            model_path=yolov8_onnx_path + "armor_yolo.onnx",
            img_size=config["armor_detector"]["img_size"],
            max_det=config["armor_detector"]["max_det"],
            conf_thres=config["armor_detector"]["conf_thres"],
            iou_thres=config["armor_detector"]["iou_thres"],
            device=self.device,
        )

        # 分类器 (使用classify.onnx)
        self.classifier = YOLOv8ONNXClassifier(
            model_path=yolov8_onnx_path + "classify.onnx",
            conf_thres=0.3,  # 使用合理的阈值
            device=self.device,
        )

        self.pixel_world_transform = pixel_world_transform
        self.visualize = visualize

        ## Initialize tracking state
        self.tracks = [
            TrackingState(class_id=class_id, name=class_name, max_frames=30)
            for class_id, class_name in enumerate(self.class_names[:10])
        ]
    
    def xyxy2pos3d(self, xyxy):
        """
        将车辆边界框坐标转换为3D世界坐标
        
        注意：xyxy坐标应该是在原始图像尺寸上的坐标
        如果检测器对图像进行了resize，需要先将坐标转换回原始图像尺寸
        
        Args:
            xyxy: 车辆边界框坐标 [x1, y1, x2, y2]，应该是原始图像尺寸上的坐标
            
        Returns:
            position_3d: 3D世界坐标 [x, y, z]
        """
        x1, y1, x2, y2 = xyxy
        # 使用车辆底部中心点进行坐标转换
        xo, yo = (x1 + x2) * 0.5, y2
        # 像素坐标(u, v)传递给pixel_to_world
        # 注意：pixel_to_world期望的坐标应该与相机内参矩阵对应的图像尺寸匹配
        if (
            position_3d := self.pixel_world_transform.pixel_to_world([xo, yo])
        ) is None:
            position_3d = np.array([0.0, 0.0, 0.0])
        return position_3d.tolist()

    def detect(
        self, img
    ) -> tuple[List[SingleDetectionResult], Optional[np.ndarray]]:
        """
        Preform full detection pipeline on the input image.
        Args:
            img: Input image in BGR format.
        Returns:
            A tuple containing a list of detection results and an optional image with visualizations.
        """

        detections = []
        img_copy = img
        car_detections, _ = self.car_detector.predict(img_copy)

        num_car_detections = len(car_detections)
        if num_car_detections == 0:
            return [], img_copy

        crop_imgs = []
        car_boxes = []
        car_confs = []
        
        for i, car_detection in enumerate(car_detections):
            class_id, xyxy, conf = car_detection
            x1, y1, x2, y2 = map(int, xyxy)
            car_box = [x1, y1, x2, y2]
            car_boxes.append(car_box)
            car_confs.append(conf)
            
            # 确保边界框有效
            if x1 < x2 and y1 < y2:
                crop_img = img_copy[y1:y2, x1:x2]
                crop_imgs.append(crop_img)
            else:
                crop_imgs.append(np.zeros((10, 10, 3), dtype=np.uint8))

        armor_detections, _ = self.armor_detector.predict_batch(crop_imgs)

        for i, (armor_detections, car_box, car_conf) in enumerate(zip(armor_detections, car_boxes, car_confs)):
            track_id = i  # 使用索引作为临时track_id
            
            position_3d = self.xyxy2pos3d(car_box)

            if len(armor_detections) == 0:
                detection_result = SingleDetectionResult(
                    class_id=-1,
                    class_conf=0.0,
                    bot_id=track_id,
                    car_box=car_box,
                    armor_box=[0.0, 0.0, 0.0, 0.0],
                    car_conf=car_conf,
                    pos_3d=position_3d,
                    time_stamp=time.time(),
                )
                detections.append(detection_result)
                continue

            max_armor_confidence_detection = max(
                armor_detections, key=lambda x: x[2], default=None
            )

            if max_armor_confidence_detection:
                armor_class_id, armor_box, armor_conf = max_armor_confidence_detection
                
                # 参考另一个项目的逻辑
                class_id = -1
                class_conf = 0.0
                
                # 1. 首先获取装甲板区域的图像
                armor_x1, armor_y1, armor_x2, armor_y2 = armor_box
                armor_x1 = max(0, int(armor_x1))
                armor_y1 = max(0, int(armor_y1))
                
                # 使用裁剪后的图像尺寸作为边界
                armor_roi = None
                if len(crop_imgs[i]) > 0:
                    armor_x2 = min(crop_imgs[i].shape[1], int(armor_x2))
                    armor_y2 = min(crop_imgs[i].shape[0], int(armor_y2))
                    
                    if armor_x2 > armor_x1 and armor_y2 > armor_y1:
                        armor_roi = crop_imgs[i][armor_y1:armor_y2, armor_x1:armor_x2]
                
                # 2. 颜色检测函数
                # 注意：这里的图像来源于 OpenCV 的裁剪，因此是 BGR 格式
                # channels[0]=B, channels[1]=G, channels[2]=R
                def get_color(img):
                    if img is None or img.size == 0:
                        return 1  # 默认灰色
                    channels = cv2.split(img)
                    if len(channels) != 3:
                        return 1

                    # 对于 BGR 格式：channels[0]=B, channels[2]=R
                    # 计算各通道均值（正确索引）
                    blue_mean = cv2.mean(channels[0])[0]  # B通道
                    red_mean = cv2.mean(channels[2])[0]  # R通道

                    # 直接比较红色和蓝色通道的均值
                    if red_mean > blue_mean:
                        return 1  # 红色
                    else:
                        return 0  # 蓝色
                
                # 3. 对装甲板区域进行分类
                class_id = -1
                class_conf = 0.0
                if armor_roi is not None:
                    # 使用分类器对装甲板区域进行分类
                    classify_detections, _ = self.classifier.predict(armor_roi)
                    if classify_detections:
                        # 分类模型直接返回类别ID和置信度
                        classify_id, _, classify_conf = classify_detections[0]
                        
                        # 4. 获取装甲板区域的颜色
                        color = get_color(armor_roi)
                        
                        # 5. 直接使用分类模型的输出进行编号分配
                        # 按照分类模型的实际输出确定机器人编号
                        # 分类模型返回0-3表示机器人编号1-4
                        robot_number = classify_id + 1  # 转换为1-based编号
                        
                        # 确保编号在有效范围内
                        if robot_number < 1:
                            robot_number = 1
                        elif robot_number > 4:
                            robot_number = 4
                        
                        # 根据颜色确定最终类别
                        # color: 0=蓝色, 1=红色, 其他=灰色
                        if color == 1:
                            # 红色机器人
                            if robot_number == 1:
                                class_id = 0  # R1
                            elif robot_number == 2:
                                class_id = 1  # R2
                            elif robot_number == 3:
                                class_id = 2  # R3
                            elif robot_number == 4:
                                class_id = 3  # R4
                        elif color == 0:
                            # 蓝色机器人
                            if robot_number == 1:
                                class_id = 5  # B1
                            elif robot_number == 2:
                                class_id = 6  # B2
                            elif robot_number == 3:
                                class_id = 7  # B3
                            elif robot_number == 4:
                                class_id = 8  # B4
                        
                        # 确保分类置信度正确赋值
                        class_conf = classify_conf
                
                # 确保类别ID在有效范围内
                if class_id < 0 or class_id >= len(self.class_names):
                    class_id = -1
                    class_conf = 0.0
                
                detection_result = SingleDetectionResult(
                    class_id=class_id,
                    class_conf=class_conf,
                    bot_id=track_id,
                    car_box=car_box,
                    armor_box=list(map(int, armor_box)),
                    car_conf=car_conf,
                    pos_3d=position_3d,  # Placeholder for 3D position
                    time_stamp=time.time(),
                )
                detections.append(detection_result)
        vis_img = None
        if self.visualize:
            vis_img = self._get_visualized_img(img_copy, detections)
        return detections, vis_img

    def _get_visualized_img(self, img, detections):
        # 使用原图尺寸直接绘制，避免每帧 resize 带来的开销
        vis_img = img.copy()

        # 缩放因子设为 1（坐标已经是原图尺寸）
        scale_x = 1.0
        scale_y = 1.0
        height, width = vis_img.shape[:2]

        for detection in detections:
            ## Draw car bounding box
            x1_car, y1_car, x2_car, y2_car = map(int, detection.car_box)
            label = f"Car: {detection.car_conf:.2f} ID: {detection.bot_id}"
            cv2.rectangle(vis_img, (x1_car, y1_car), (x2_car, y2_car), (0, 255, 0), 2)
            cv2.putText(
                vis_img,
                label,
                (x1_car, y1_car - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            ## Draw armor detection if any
            class_id = detection.class_id
            # 检查装甲板边界框是否有效，而不是检查class_id
            armor_box = detection.armor_box
            if len(armor_box) == 4:
                x1_armor, y1_armor, x2_armor, y2_armor = map(int, armor_box)
                # 确保装甲板边界框有效
                if x2_armor > x1_armor and y2_armor > y1_armor:
                    x1_armor = int(x1_armor + detection.car_box[0])
                    y1_armor = int(y1_armor + detection.car_box[1])
                    x2_armor = int(x2_armor + detection.car_box[0])
                    y2_armor = int(y2_armor + detection.car_box[1])

                    # 确定标签和颜色
                    if class_id >= 0 and class_id < len(self.class_names):
                        label = f"{self.class_names[class_id]}: {detection.class_conf:.2f}"
                        if class_id < 4:  # red
                            color = (0, 0, 255)  # 红色框
                        elif class_id < 9:  # blue
                            color = (255, 0, 0)  # 蓝色框
                        else:  # grey
                            color = (255, 255, 255)  # 白色框
                    else:
                        label = f"Armor: {detection.class_conf:.2f}"
                        color = (0, 255, 255)  # 黄色框

                    cv2.rectangle(
                        vis_img, (x1_armor, y1_armor), (x2_armor, y2_armor), color, 2
                    )
                    cv2.putText(
                        vis_img,
                        label,
                        (x1_armor, y1_armor - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
        return vis_img


if __name__ == "__main__":
    import yaml
    from transform.ray_renderer import PixelToWorld
    import numpy as np

    config = yaml.safe_load(open("config/params.yaml", "r"))
    pixel_world_transform = PixelToWorld.build_from_config(config)
    tracker = BaseDetector(config, pixel_world_transform, visualize=True)

    img = cv2.imread("test_assets/ustgz_0.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result, vis_img = tracker.detect(img)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    vis_img = cv2.resize(vis_img, (1536, 1024))
    cv2.imshow("Detection Result", vis_img)
    cv2.waitKey(0)
