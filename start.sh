#!/bin/bash
# RM2025 Radar Algorithm System 启动脚本

# ========== 关键修正：使用项目内的MvImport路径 ==========
# 项目内MvImport的上级目录（hik_camera目录）
export MVCAM_COMMON_RUNENV=/opt/MVS/lib  # 仍需保留，用于底层库
export LD_LIBRARY_PATH=/opt/MVS/lib64:$LD_LIBRARY_PATH  # 底层库路径
# 核心：将项目内的hik_camera目录加入Python路径
export PYTHONPATH=/home/geqingyu/RM2025-Radar-Algorithm-main/driver/hik_camera:$PYTHONPATH

# ========== 原有配置保留 ==========
cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins

# ========== 调试：验证MvImport是否存在 ==========
echo "=== 环境变量配置 ==="
echo "PYTHONPATH: $PYTHONPATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "MVCAM_COMMON_RUNENV: $MVCAM_COMMON_RUNENV"
echo "=== 检查项目内MvImport ==="
ls -l /home/geqingyu/RM2025-Radar-Algorithm-main/driver/hik_camera/MvImport/ 2>/dev/null
echo "===================="

# 启动项目
python3 main.py --config config/params.yaml --device_config config/device.yaml
