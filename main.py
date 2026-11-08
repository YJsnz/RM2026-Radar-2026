# -*- coding: utf-8 -*-
"""
Main entry point for RM2025 Radar Algorithm System
Copyright (c) 2025 香港科技大学ENTERPRIZE战队（HKUST ENTERPRIZE Team）

Licensed under the MIT License. See LICENSE file in the project root for license information.
"""

import os
from pathlib import Path
# 在导入任何模块之前设置Qt插件路径，避免opencv覆盖
system_qt_plugin_path = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
if os.path.exists(system_qt_plugin_path):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = system_qt_plugin_path
    os.environ["QT_PLUGIN_PATH"] = system_qt_plugin_path
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", "/tmp/ultralytics")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from driver.referee.referee_comm import RefereeCommManager
from utils.config import (
    load_cfg_from_cfg_file,
    merge_cfg_from_list,
)
from interface.core import launch
import argparse
import rclpy

from main_event_loop import MainEventLoop


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pytorch Referring Expression Segmentation"
    )
    parser.add_argument("--gpu", default="0,1")
    parser.add_argument(
        "--config", default="config/params.yaml", type=str, help="config file"
    )
    parser.add_argument(
        "--device_config",
        default="config/device.yaml",
        type=str,
        help="camera and lidar setting",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="override some settings in the config.",
    )

    return parser.parse_args()

def load_yaml_config(path: str):
    import yaml
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_device_config(path: str, opts=None):
    cfg = load_cfg_from_cfg_file(path)
    if opts is not None:
        cfg = merge_cfg_from_list(cfg, opts)
    return cfg


def _is_camera_index(source) -> bool:
    if isinstance(source, int):
        return True
    if not isinstance(source, str):
        return False
    stripped = source.strip()
    return stripped.isdigit()


def validate_runtime_config(yaml_config, device_config):
    required_files = {
        "car detector": yaml_config["car_detector"]["weights_path"],
        "armor detector": yaml_config["armor_detector"]["weights_path"],
        "field mesh": yaml_config["field"]["mesh_path"],
        "field keypoints": yaml_config["field"]["keypoints"],
        "demo image": yaml_config["transform"]["demo_img_path"],
        "point guesser config": yaml_config["tracker"]["point_guesser_config_path"],
    }
    warmup_image_path = yaml_config["tracker"].get("warmup_image_path")
    if warmup_image_path:
        required_files["warmup image"] = warmup_image_path

    missing_files = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if path and not Path(path).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing required runtime files:\n" + "\n".join(missing_files)
        )

    if yaml_config["debug"]["inference_video"]:
        video_source = yaml_config["debug"]["video_source"]
        if not _is_camera_index(video_source) and not Path(str(video_source)).exists():
            raise FileNotFoundError(
                f"Configured video source does not exist: {video_source}"
            )
    else:
        required_device_keys = [
            "width",
            "height",
            "exposure_time",
            "gain",
            "recording_workers_num",
            "recording_save_root_dir",
        ]
        missing_keys = [key for key in required_device_keys if key not in device_config]
        if missing_keys:
            raise KeyError(
                "Missing camera settings in device config: "
                + ", ".join(missing_keys)
            )


if __name__ == "__main__":
    args = parse_args()
    yaml_config = load_yaml_config(args.config)
    device_config = load_device_config(args.device_config, args.opts)
    validate_runtime_config(yaml_config, device_config)
    rclpy.init()

    if yaml_config["debug"]["inference_video"]:
        from driver.hik_camera.mock_hik import SimpleHikCamera
        camera = SimpleHikCamera(
            video_source=yaml_config["debug"]["video_source"]
        )
    else:
        from driver.hik_camera.hik import SimpleHikCamera
        camera = SimpleHikCamera(device_config)

    referee = RefereeCommManager(port=yaml_config["referee"]["port"], 
                                baudrate=yaml_config["referee"]["baudrate"])   
    referee.start()

    event_loop = MainEventLoop(
        camera = camera, referee=referee
    )
    camera.start_streaming()
    
    
    if not yaml_config["debug"]["inference_video"] and yaml_config["debug"]["streaming_video"]:
        camera.start_saving_threads()

    launch()
    import time
    
    ## do not terminate the main eventloop
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping...")
        referee.close()
        camera.stop_streaming()
        camera.stop_saving_images()
        camera.close()
        event_loop.stop()
        
        
