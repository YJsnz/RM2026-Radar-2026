#!/bin/bash
# RM2026 Radar YSU System 启动脚本

echo "正在启动 YSU-radar ..."
# 切换到脚本所在目录
cd "$(dirname "$0")"

# ========== 环境变量配置 ==========
# 海康相机SDK环境变量
export MVCAM_COMMON_RUNENV=/opt/MVS/lib

# Python路径配置（使用当前项目内的 hik_camera，确保能找到 MvImport）
export PYTHONPATH="$(pwd)/driver/hik_camera:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"
# 禁用 ~/.local 中的用户级 Python 包，避免 CUDA 13/cuDNN 9.20 与当前环境的
# CUDA 12/cuDNN 9.1 混用后导致 cudnnGetLibConfig 符号冲突。
export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR=/tmp/ultralytics
export ULTRALYTICS_SETTINGS_DIR=/tmp/ultralytics
export XDG_CONFIG_HOME=/tmp
export MPLCONFIGDIR=/tmp/matplotlib

# 系统库路径配置
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# 添加虚拟环境的 CUDA 库路径（优先）
export LD_LIBRARY_PATH=/home/geqingyu/miniconda3/envs/rm2026-radar/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/geqingyu/miniconda3/envs/rm2026-radar/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/geqingyu/miniconda3/envs/rm2026-radar/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/geqingyu/miniconda3/envs/rm2026-radar/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/geqingyu/miniconda3/envs/rm2026-radar/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# 添加系统CUDA库路径（后备）
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Qt库配置（解决版本冲突）
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms
export QT_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins

# 强制使用系统Qt库
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libQt5Core.so.5:/usr/lib/x86_64-linux-gnu/libQt5Gui.so.5:/usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5

# ========== 环境激活 ==========
# 通过 Conda 的 shell 钩子激活虚拟环境。envs/<name>/bin/activate 并不存在。
source /home/geqingyu/miniconda3/etc/profile.d/conda.sh
conda activate rm2026-radar

# 激活ROS2 Humble环境
source /opt/ros/humble/setup.bash

# 再次添加虚拟环境的CUDA库路径，确保被正确设置
export LD_LIBRARY_PATH=/home/geqingyu/miniconda3/envs/rm2026-radar/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/geqingyu/miniconda3/envs/rm2026-radar/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/geqingyu/miniconda3/envs/rm2026-radar/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/geqingyu/miniconda3/envs/rm2026-radar/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/geqingyu/miniconda3/envs/rm2026-radar/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH



# ========== 环境验证 ==========
echo "=== RM2025 Radar Algorithm System 启动 ==="
echo "环境变量配置:"
echo "- PYTHONPATH: $PYTHONPATH"
echo "- PYTHONNOUSERSITE: $PYTHONNOUSERSITE"
echo "- LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "- MVCAM_COMMON_RUNENV: $MVCAM_COMMON_RUNENV"
echo "- QT_QPA_PLATFORM_PLUGIN_PATH: $QT_QPA_PLATFORM_PLUGIN_PATH"
echo "- YOLO_CONFIG_DIR: $YOLO_CONFIG_DIR"
echo "- ULTRALYTICS_SETTINGS_DIR: $ULTRALYTICS_SETTINGS_DIR"
echo "- XDG_CONFIG_HOME: $XDG_CONFIG_HOME"
echo "- MPLCONFIGDIR: $MPLCONFIGDIR"

cat << "EOF"
╔════════════════════════════════╗
║  ██╗   ██╗███████╗██╗   ██╗    ║
║  ╚██╗ ██╔╝██╔════╝██║   ██║    ║
║   ╚████╔╝ ███████╗██║   ██║    ║
║    ╚██╔╝  ╚════██║██║   ██║    ║
║     ██║   ███████║╚██████╔╝    ║
║     ╚═╝   ╚══════╝ ╚═════╝     ║
╚════════════════════════════════╝
        YSU Project Launcher
EOF

echo "=== 检查项目内MvImport ==="
ls -l driver/hik_camera/MvImport/ 2>/dev/null || echo "警告: MvImport目录不存在"
echo "===================="

# ========== 启动项目 ==========
echo "启动雷达算法系统..."
/home/geqingyu/miniconda3/envs/rm2026-radar/bin/python main.py --config config/params.yaml --device_config config/device.yaml
