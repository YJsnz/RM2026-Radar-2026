# RM2026 Radar YSU System
## 环境要求

### 硬件要求
- NVIDIA GPU（支持CUDA 11.8）
- 海康相机
- 串口设备（用于裁判系统通信）

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
cd RM2025-Radar-Algorithm-main
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

## 运行方法

### 启动项目
```bash
# 确保虚拟环境已激活
conda activate rm2026-radar

# 运行启动脚本
bash start.sh
```

### 配置文件
- `config/params.yaml`：系统参数配置
- `config/device.yaml`：设备配置

## 项目结构

```
RM2025-Radar-Algorithm-main/
├── config/                # 配置文件目录
│   ├── params.yaml        # 系统参数配置
│   └── device.yaml        # 设备配置
├── driver/                # 驱动目录
│   ├── hik_camera/        # 海康相机驱动
│   │   ├── MvImport/      # 海康相机SDK Python接口
│   │   ├── hik.py         # 海康相机实现
│   │   └── mock_hik.py    # 模拟相机实现
│   └── referee/           # 裁判系统驱动
│       ├── referee_comm.py # 裁判系统通信
│       └── serial_comm.py  # 串口通信
├── interface/             # 界面目录
│   ├── core.py            # 核心界面实现
│   └── interfactive_display.py # 交互式显示组件
├── model/                 # 模型目录
│   └── pfa_yolov5/        # YOLOv5模型实现
├── utils/                 # 工具目录
│   └── config.py          # 配置工具
├── field/                 # 场地相关文件
├── demo/                  # 演示数据
├── start.sh               # 启动脚本
└── main.py                # 主入口文件
```

## 核心功能

1. **目标检测**：使用YOLOv5模型进行实时目标检测
2. **目标跟踪**：基于检测结果进行目标跟踪
3. **场地定位**：将检测到的目标投影到场地坐标系
4. **裁判系统通信**：与裁判系统进行实时通信
5. **相机控制**：支持相机参数调整和图像获取
6. **可视化界面**：实时显示检测结果和场地投影

## 注意事项

1. **Qt版本冲突**：系统使用系统安装的Qt库（5.15.3），避免与其他Qt库冲突
2. **CUDA环境**：确保CUDA 11.8和cuDNN 9.1.0正确安装
3. **串口设备**：确保裁判系统串口设备正确连接
4. **海康相机**：确保海康相机SDK正确安装并连接相机

## 依赖包版本

以下是虚拟环境中安装的所有依赖包及其版本：

| 包名 | 版本 |
|------|------|
| addict | 2.4.0 |
| annotated-types | 0.7.0 |
| apturl | 0.5.2 |
| argcomplete | 1.8.1 |
| asttokens | 3.0.1 |
| attrs | 25.4.0 |
| beautifulsoup4 | 4.10.0 |
| blinker | 1.9.0 |
| Brlapi | 0.8.3 |
| catkin-pkg-modules | 1.1.0 |
| certifi | 2020.6.20 |
| chardet | 4.0.0 |
| click | 8.3.1 |
| colorama | 0.4.4 |
| coloredlogs | 15.0.1 |
| comm | 0.2.3 |
| command-not-found | 0.3 |
| ConfigArgParse | 1.7.5 |
| contourpy | 1.3.2 |
| cryptography | 3.4.8 |
| cupshelpers | 1.0 |
| cycler | 0.12.1 |
| dash | 4.0.0 |
| dbus-python | 1.2.18 |
| decorator | 5.2.1 |
| defer | 1.0.6 |
| distro | 1.7.0 |
| distro-info | 1.1+ubuntu0.2 |
| docutils | 0.17.1 |
| empy | 3.3.4 |
| exceptiongroup | 1.3.1 |
| executing | 2.2.1 |
| fastjsonschema | 2.21.2 |
| filelock | 3.20.0 |
| filterpy | 1.4.5 |
| flake8 | 4.0.1 |
| Flask | 3.1.3 |
| flatbuffers | 25.12.19 |
| fonttools | 4.62.0 |
| fsspec | 2025.12.0 |
| html5lib | 1.1 |
| httplib2 | 0.20.2 |
| humanfriendly | 10.0 |
| idna | 3.3 |
| importlib-metadata | 4.6.4 |
| iniconfig | 1.1.1 |
| ipython | 8.38.0 |
| ipywidgets | 8.1.8 |
| itsdangerous | 2.2.0 |
| jedi | 0.19.2 |
| jeepney | 0.7.1 |
| Jinja2 | 3.1.6 |
| joblib | 1.5.3 |
| jsonschema | 4.26.0 |
| jsonschema-specifications | 2025.9.1 |
| jupyter_core | 5.9.1 |
| jupyterlab_widgets | 3.0.16 |
| keyring | 23.5.0 |
| kiwisolver | 1.5.0 |
| language-selector | 0.1 |
| lark | 1.1.1 |
| launchpadlib | 1.10.16 |
| lazr.restfulclient | 0.14.4 |
| lazr.uri | 1.0.6 |
| louis | 3.20.0 |
| lxml | 4.8.0 |
| macaroonbakery | 1.3.1 |
| MarkupSafe | 3.0.2 |
| matplotlib | 3.10.8 |
| matplotlib-inline | 0.2.1 |
| mccabe | 0.6.1 |
| more-itertools | 8.10.0 |
| mpmath | 1.3.0 |
| narwhals | 2.18.0 |
| nbformat | 5.10.4 |
| nest-asyncio | 1.6.0 |
| netifaces | 0.11.0 |
| networkx | 3.4.2 |
| numpy | 1.26.4 |
| nvidia-cublas-cu11 | 11.11.3.6 |
| nvidia-cuda-cupti-cu11 | 11.8.87 |
| nvidia-cuda-nvrtc-cu11 | 11.8.89 |
| nvidia-cuda-runtime-cu11 | 11.8.89 |
| nvidia-cudnn-cu11 | 9.1.0.70 |
| nvidia-cufft-cu11 | 10.9.0.58 |
| nvidia-curand-cu11 | 10.3.0.86 |
| nvidia-cusolver-cu11 | 11.4.1.48 |
| nvidia-cusparse-cu11 | 11.7.5.86 |
| nvidia-nccl-cu11 | 2.21.5 |
| nvidia-nvtx-cu11 | 11.8.86 |
| oauthlib | 3.2.0 |
| olefile | 0.46 |
| onnxruntime-gpu | 1.23.2 |
| open3d | 0.19.0 |
| opencv-python | 4.8.0.76 |
| packaging | 21.3 |
| pandas | 2.3.3 |
| parso | 0.8.6 |
| pexpect | 4.8.0 |
| pillow | 12.1.1 |
| pip | 22.0.2 |
| platformdirs | 4.9.4 |
| plotly | 6.6.0 |
| pluggy | 0.13.0 |
| polars | 1.39.0 |
| polars-runtime-32 | 1.39.0 |
| prompt_toolkit | 3.0.52 |
| protobuf | 3.12.4 |
| psutil | 7.2.2 |
| ptyprocess | 0.7.0 |
| pure_eval | 0.2.3 |
| py | 1.10.0 |
| pycairo | 1.20.1 |
| pycodestyle | 2.8.0 |
| pycups | 2.0.1 |
| pydantic | 2.12.5 |
| pydantic_core | 2.41.5 |
| pydocstyle | 6.1.1 |
| pyflakes | 2.4.0 |
| Pygments | 2.11.2 |
| PyGObject | 3.42.1 |
| PyJWT | 2.3.0 |
| pymacaroons | 0.13.0 |
| PyNaCl | 1.5.0 |
| pyparsing | 3.3.2 |
| PyQt5 | 5.15.11 |
| PyQt5-Qt5 | 5.15.18 |
| PyQt5_sip | 12.18.0 |
| pyquaternion | 0.9.9 |
| pyRFC3339 | 1.1 |
| pyserial | 3.5 |
| pytest | 6.2.5 |
| python-apt | 2.4.0+ubuntu4.1 |
| python-dateutil | 2.9.0.post0 |
| python-debian | 0.1.43+ubuntu1.1 |
| pytz | 2022.1 |
| pyxdg | 0.27 |
| PyYAML | 5.4.1 |
| referencing | 0.37.0 |
| reportlab | 3.6.8 |
| requests | 2.25.1 |
| retrying | 1.4.2 |
| roman | 3.3 |
| rosdistro-modules | 1.0.1 |
| rospkg-modules | 1.6.1 |
| rpds-py | 0.30.0 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| screen-resolution-extra | 0.0.0 |
| seaborn | 0.13.2 |
| SecretStorage | 3.3.1 |
| setuptools | 59.6.0 |
| six | 1.16.0 |
| snowballstemmer | 2.2.0 |
| soupsieve | 2.3.1 |
| stack-data | 0.6.3 |
| sympy | 1.14.0 |
| systemd-python | 234 |
| threadpoolctl | 3.6.0 |
| toml | 0.10.2 |
| torch | 2.7.1+cu118 |
| torchaudio | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| tqdm | 4.67.3 |
| traitlets | 5.14.3 |
| triton | 3.3.1 |
| typing_extensions | 4.15.0 |
| typing-inspection | 0.4.2 |
| tzdata | 2025.3 |
| ubuntu-drivers-common | 0.0.0 |
| ubuntu-pro-client | 8001 |
| ufw | 0.36.1 |
| ultralytics | 8.4.21 |
| ultralytics-thop | 2.0.18 |
| unattended-upgrades | 0.1 |
| urllib3 | 1.26.5 |
| wadllib | 1.3.6 |
| wcwidth | 0.6.0 |
| webencodings | 0.5.1 |
| Werkzeug | 3.1.6 |
| wheel | 0.37.1 |
| widgetsnbextension | 4.0.15 |
| xdg | 5 |
| xkit | 0.0.0 |
| zipp | 1.0.0 |
