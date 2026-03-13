# RM2025 Radar Algorithm System

## 项目简介

RM2025 Radar Algorithm System 是一个用于 RoboMaster 2025 赛事的雷达算法系统，主要功能包括：
- 车辆检测与追踪
- 装甲板检测与分类
- 多目标跟踪
- 机器人编号识别
- 与裁判系统通信

## 环境要求

### 操作系统
- **推荐**：Ubuntu 22.04 LTS
- **支持**：Ubuntu 20.04 LTS

### 软件依赖
1. **ROS 2**：Humble 或 Iron 版本
2. **Python**：3.9 或更高版本
3. **CUDA**（可选，用于GPU加速）：11.7 或更高版本
4. **ONNX Runtime**：1.14.1 或更高版本

## 依赖安装

### 1. 安装 ROS 2

按照 ROS 2 官方文档安装 Humble 版本：
```bash
sudo apt update && sudo apt install curl gnupg lsb-release

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop-full

# 设置环境变量
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. 安装 Python 依赖

```bash
pip install --upgrade pip
pip install onnxruntime opencv-python numpy scikit-learn matplotlib seaborn PyQt5
```

### 3. 安装系统依赖

```bash
sudo apt update
sudo apt install libopencv-dev python3-opencv socat
```

### 4. 安装相机驱动（可选）

如果使用大恒相机，需要安装 MVS SDK：
```bash
# 下载并安装大恒相机 SDK
# 安装完成后，相机驱动会位于 /opt/MVS/
```

## 项目结构

```
RM2025-Radar-Algorithm-main/
├── config/              # 配置文件
│   ├── params.yaml      # 主要参数配置
│   └── device.yaml      # 设备配置
├── driver/              # 驱动程序
│   ├── hik_camera/      # 海康相机驱动
│   └── referee/         # 裁判系统通信
├── model/               # 模型相关
│   ├── digit_classifier/ # 数字分类器
│   ├── yolov8/          # YOLOv8 模型
│   │   └── ONNX/         # ONNX 格式模型
├── tracker/             # 追踪器
│   └── CascadeMatchTracker/ # 级联匹配追踪器
├── run_scripts/         # 运行脚本
│   ├── run.sh           # 主启动脚本
│   ├── camera_capture.sh # 相机控制脚本
│   └── create_serial_loopback.sh # 串口环回脚本
├── main.py              # 主入口文件
└── main_event_loop.py   # 主事件循环
```

## 模型文件

确保 `model/yolov8/ONNX/` 目录下有以下模型文件：
- `RM2025.onnx` - 车辆检测模型
- `armor_yolo.onnx` - 装甲板检测模型
- `classify.onnx` - 装甲板分类模型

## 如何启动项目

### 方法一：使用 run.sh 脚本（推荐）

```bash
cd RM2025-Radar-Algorithm-main
chmod +x run_scripts/run.sh
./run_scripts/run.sh
```

### 方法二：手动启动

1. **激活 ROS 2 环境**：
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **设置设备权限**：
   ```bash
   # 为串口设备添加权限（根据实际设备调整）
   sudo chmod 666 /dev/ttyUSB0
   ```

3. **启动项目**：
   ```bash
   python main.py --config "config/params.yaml" --device_config "config/device.yaml"
   ```

## 故障排除

### 1. ROS 2 相关错误
- **症状**：导入 `rclpy` 失败
- **解决方案**：确保 ROS 2 环境正确激活，运行 `source /opt/ros/humble/setup.bash`

### 2. 模型加载错误
- **症状**：ONNX 模型加载失败
- **解决方案**：
  - 检查模型文件路径是否正确
  - 确保 ONNX Runtime 版本兼容
  - 验证模型文件是否完整

### 3. 设备访问错误
- **症状**：无法打开摄像头或串口
- **解决方案**：
  - 检查设备连接是否正常
  - 确保有足够的权限访问设备
  - 对于串口，运行 `sudo chmod 666 /dev/ttyUSB0`

### 4. 分类错误
- **症状**：装甲板分类不正确（如 R1 被分类为 R4）
- **解决方案**：
  - 检查分类模型是否为最新版本
  - 考虑重新训练分类模型
  - 调整分类阈值参数

## 相关脚本说明

### 1. run.sh
- **功能**：主启动脚本
- **用途**：设置环境变量并启动项目
- **使用**：`./run_scripts/run.sh`

### 2. camera_capture.sh
- **功能**：启动大恒相机 SDK 软件
- **用途**：用于相机参数配置和调试
- **使用**：`./run_scripts/camera_capture.sh`

### 3. create_serial_loopback.sh
- **功能**：创建虚拟串口环回设备
- **用途**：用于调试串口通信，无需实际物理设备
- **使用**：`./run_scripts/create_serial_loopback.sh`

## 测试模式

如果只想测试装甲板识别功能，而不需要完整的 ROS 2 环境，可以运行：

```bash
python test_armor_recognition.py
```

这会处理 `demo/demo1.jpg` 图像并生成识别结果。

## 联系方式

如果遇到问题，请联系项目维护者获取帮助。

---

**注意**：本项目为 RoboMaster 2025 赛事专用，使用前请确保遵守相关比赛规则。
