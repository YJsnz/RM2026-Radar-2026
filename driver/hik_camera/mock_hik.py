import cv2
from threading import Lock, Thread
import time
import os


class SimpleHikCamera:
    _instance = None

    def __init__(self, video_source):
        """
        Initialize the camera with the given video source.

        Args:
            video_source (int or str): The video source, can be an integer for webcam, 
                                      a string for video file, or an image file path.
        """
        self.video_source = video_source
        self.is_image = False
        self.image_path = None
        
        # Check if it's an image file
        if isinstance(video_source, str) and os.path.isfile(video_source):
            ext = os.path.splitext(video_source)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.is_image = True
                self.image_path = video_source
                print(f"Initialized camera with image source: {video_source}")
            else:
                self.capture = cv2.VideoCapture(video_source)
                print(f"Initialized camera with video source: {video_source}")
        else:
            self.capture = cv2.VideoCapture(video_source)
            print(f"Initialized camera with source: {video_source}")
        
        self.__class__._instance = self

        self.cur_image = None
        self.lock = Lock()
        self.fetch_loop = Thread(target=self.fetch_image_loop, daemon=True)
        self.fetch_loop.start()

    def fetch_image_loop(self):
        if self.is_image:
            # Load and loop the image
            img_bgr = cv2.imread(self.image_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                while True:
                    self.cur_image = img_rgb.copy()
                    time.sleep(0.033)  # ~30 FPS
            else:
                print(f"Failed to load image from: {self.image_path}")
        else:
            # Original video/camera loop
            while True:
                ret, frame = self.capture.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.cur_image = frame
                else:
                    print("Failed to capture image from camera.")
                    break
                time.sleep(0.033)

            if hasattr(self, 'capture'):
                self.capture.release()

    def get_image_latest(self, group_id: str = "", timeout=1):
        if self.cur_image is None:
            print("No image captured yet.")
            return None, 0.0
        return self.cur_image.copy(), 1.0

    def register_group(self, group_id: str):
        """
        Register a group ID for the camera.

        Args:
            group_id (str): The group ID to register.
        """
        print(f"Registered group ID: {group_id}")
    
    def start_streaming(self):
        pass
    
    def stop_streaming(self):
        pass
    
    def stop_saving_images(self):
        pass
    
    def close(self):
        if hasattr(self, 'capture') and not self.is_image:
            self.capture.release()