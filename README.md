# RM2026 Radar YSU System

## 项目概述

RM2026 Radar YSU System是一个基于计算机视觉和深度学习的雷达系统，用于RoboMaster比赛中的目标检测、跟踪和定位。该系统使用多种深度学习模型实现实时目标检测和跟踪，并通过相机标定和几何变换将检测结果投影到场地坐标系中。

## 环境要求

### 硬件要求
- NVIDIA GPU（支持CUDA 11.8）
- 海康相机（或其他兼容相机）
- 串口设备（用于裁判系统通信）
- 足够的存储空间（至少5GB）

### 软件要求
- Ubuntu 22.04 LTS
- CUDA 11.8
- cuDNN 9.1.0
- ROS2 Humble
- Python 3.10

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd RM2026-Radar
```

### 2. 创建虚拟环境
```bash
# 使用conda创建虚拟环境
conda create -n rm2026-radar python=3.10
conda activate rm2026-radar
```

### 3. 安装依赖包
```bash
# 安装PyTorch（带CUDA支持）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖包
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python==4.8.0.76 pyqt5 open3d filterpy seaborn ultralytics onnxruntime-gpu pydantic pyserial
```

### 4. 安装海康相机SDK
下载并安装海康相机SDK（MVS）：
```bash
sudo dpkg -i MVS-4.6.3_x86_64_20260205.deb
```

### 5. 安装ROS2 Humble
```bash
sudo apt update
sudo apt install -y ros-humble-ros-base ros-humble-rclpy
```

## 配置说明

### 主要配置文件
- `config/params.yaml`：系统核心参数配置
- `config/device.yaml`：设备配置
- `config/botsort.yaml`：BotSort跟踪器配置
- `config/bytetrack.yaml`：ByteTrack跟踪器配置
- `config/guess_pts.yaml`：点预测配置

### 关键配置参数

#### 1. 视频推理配置
```yaml
debug:
  inference_video: true  # 是否启用视频推理
  video_source: "p1.mp4"  # 视频源路径
  streaming_video: false  # 是否录制视频
```

#### 2. 模型配置
```yaml
car_detector:
  weights_path: "yolo/Car-yolov8s.pt"
  img_size: 1280  # 车辆检测器输入尺寸
  conf_thres: 0.5  # 置信度阈值
  iou_thres: 0.5  # IOU阈值
  max_det: 10  # 最大检测数量

armor_detector:
  weights_path: "yolo/praysky_coord_noe2e_0331_576x768.onnx"
  conf_thres: 0.25  # 装甲识别阈值
  iou_thres: 0.4  # IOU阈值
  max_det: 5  # 每辆车内最多保留的装甲候选数
  class_names: ["B1", "B2", "B3", "B4", "B5", "B7", "R1", "R2", "R3", "R4", "R5", "R7"]
```

#### 3. 设备配置
```yaml
device: "cuda"  # 运行设备（cuda或cpu）
```

## 运行方法

### 启动项目
```bash
# 确保虚拟环境已激活
conda activate rm2026-radar

# 运行启动脚本
bash start.sh
```

### 视频推理模式
1. 修改`config/params.yaml`中的`video_source`为视频文件路径
2. 设置`inference_video: true`
3. 运行启动脚本

### 相机实时模式
1. 修改`config/params.yaml`中的`inference_video: false`
2. 确保海康相机已连接
3. 运行启动脚本

## 项目结构

```
RM2026-Radar/
├── config/                # 配置文件目录
│   ├── params.yaml        # 系统参数配置
│   ├── device.yaml        # 设备配置
│   ├── botsort.yaml       # BotSort跟踪器配置
│   ├── bytetrack.yaml     # ByteTrack跟踪器配置
│   └── guess_pts.yaml     # 点预测配置
├── driver/                # 驱动目录
│   ├── hik_camera/        # 海康相机驱动
│   │   ├── MvImport/      # 海康相机SDK Python接口
│   │   ├── hik.py         # 海康相机实现
│   │   └── mock_hik.py    # 模拟相机实现
│   └── referee/           # 裁判系统驱动
│       ├── referee_comm.py # 裁判系统通信
│       ├── serial_comm.py  # 串口通信
│       └── usb.py          # USB设备管理
├── interface/             # 界面目录
│   ├── core.py            # 核心界面实现
│   └── interfactive_display.py # 交互式显示组件
├── model/                 # 模型目录
│   ├── pfa_yolov5/        # YOLOv5模型实现
│   ├── yolov8/            # YOLOv8模型实现
│   ├── yolov12/           # YOLOv12模型实现
│   ├── armor_detector.py  # 装甲板检测器
│   └── armor_light_classifier.py # 装甲板灯分类器
├── tracker/               # 跟踪器目录
│   └── CascadeMatchTracker/ # 级联匹配跟踪器
├── transform/             # 坐标转换目录
│   ├── ray_renderer.py    # 射线渲染
│   └── solidwork2uwb.py   # SolidWork到UWB坐标转换
├── utils/                 # 工具目录
│   ├── config.py          # 配置工具
│   └── average_meter.py   # 平均计数器
├── field/                 # 场地相关文件
├── demo/                  # 演示数据
├── run_scripts/           # 运行脚本
├── start.sh               # 启动脚本
├── main.py                # 主入口文件
└── main_event_loop.py     # 主事件循环
```

## 核心功能

### 1. 两阶段目标识别
- **第一层车辆检测**：使用 `yolo/Car-yolov8s.pt` 在全图检测机器人车辆
- **第二层装甲识别**：对每个车辆裁剪区域运行 `yolo/praysky_coord_noe2e_0331_576x768.onnx`
- **目标映射**：将第二层模型输出映射为 `B1-B5-B7 / R1-R5-R7`，再交给跟踪、投影和裁判系统

### 2. 目标跟踪
- **级联匹配跟踪**：基于检测结果和历史轨迹进行目标跟踪
- **Kalman滤波**：使用Kalman滤波器预测目标位置
- **Bot ID关联**：为每个检测到的目标分配唯一ID

### 3. 场地定位
- **相机标定**：使用相机内参和外参进行坐标转换
- **射线投影**：将像素坐标转换为场地坐标系
- **UWB坐标转换**：将场地坐标转换为UWB坐标系

### 4. 裁判系统通信
- **串口通信**：与裁判系统进行实时通信
- **数据打包**：将检测结果打包为裁判系统格式
- **状态上报**：实时上报目标状态

### 5. 可视化界面
- **实时显示**：显示检测结果和跟踪状态
- **场地投影**：在场地模型上显示目标位置
- **性能监控**：显示帧率和系统状态

## 模型说明

### 1. 车辆检测器
- **模型**：YOLOv8 PyTorch（`Car-yolov8s.pt`）
- **输入尺寸**：1280x1280
- **输出**：车辆边界框和置信度

### 2. 装甲识别模型
- **模型**：YOLO Pose ONNX（`praysky_coord_noe2e_0331_576x768.onnx`）
- **输入尺寸**：576x768
- **输出**：装甲板边界框、12 类目标类别、小/大装甲类别、4 类颜色属性和四点角点坐标
- **类别语义**：`s0/s1` 表示小/大装甲，`o0/o1/o2/o3/o4/o5/o6/o7` 分别表示哨兵、1、2、3、4、5、前哨站、基地

## 性能优化

### 1. GPU加速
- 使用CUDA进行模型推理
- 批量处理提高吞吐量
- 模型量化减少内存使用

### 2. 代码优化
- 多线程处理
- 内存管理优化
- 计算缓存

### 3. 配置优化
- 调整模型输入尺寸
- 优化置信度阈值
- 合理设置最大检测数量

## 常见问题和解决方案

### 1. 串口连接失败
- **原因**：串口设备未连接或权限不足
- **解决方案**：
  - 检查串口设备连接
  - 确保用户有访问串口的权限
  - 修改`config/params.yaml`中的串口配置

### 2. 相机无法获取图像
- **原因**：相机未连接或SDK未正确安装
- **解决方案**：
  - 检查相机连接
  - 确认海康SDK已正确安装
  - 尝试重新启动相机

### 3. 推理帧率低
- **原因**：模型输入尺寸过大或硬件性能不足
- **解决方案**：
  - 降低模型输入尺寸
  - 优化代码性能
  - 考虑使用更强大的GPU

### 4. CUDA相关错误
- **原因**：CUDA版本不匹配或驱动未正确安装
- **解决方案**：
  - 确保CUDA 11.8和cuDNN 9.1.0正确安装
  - 检查GPU驱动是否最新
  - 验证CUDA环境变量配置

## 注意事项

1. **Qt版本冲突**：系统使用系统安装的Qt库（5.15.3），避免与其他Qt库冲突
2. **CUDA环境**：确保CUDA 11.8和cuDNN 9.1.0正确安装
3. **串口设备**：确保裁判系统串口设备正确连接
4. **海康相机**：确保海康相机SDK正确安装并连接相机
5. **模型文件**：确保模型文件存在于正确路径
6. **视频文件**：确保视频文件格式兼容且路径正确

## 依赖包版本

以下是虚拟环境中安装的主要依赖包及其版本：

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.7.1+cu118 | 深度学习框架 |
| torchvision | 0.22.1+cu118 | 计算机视觉库 |
| opencv-python | 4.8.0.76 | 图像处理 |
| onnxruntime-gpu | 1.23.2 | ONNX模型推理 |
| ultralytics | 8.4.21 | YOLO模型实现 |
| open3d | 0.19.0 | 3D点云处理 |
| filterpy | 1.4.5 | 滤波算法 |
| pyserial | 3.5 | 串口通信 |
| PyQt5 | 5.15.11 | GUI界面 |
| numpy | 1.26.4 | 数值计算 |
| scipy | 1.15.3 | 科学计算 |
| pandas | 2.3.3 | 数据处理 |

## 许可证

本项目使用MIT许可证。详见LICENSE文件。

## 贡献

欢迎贡献代码和提出建议！请通过GitHub提交Pull Request或Issue。

## 联系方式

如有问题或建议，请联系项目维护者。
