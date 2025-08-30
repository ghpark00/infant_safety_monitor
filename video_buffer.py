# video_buffer.py
import cv2
import threading
from collections import deque
from datetime import datetime

class VideoBuffer:
    def __init__(self, max_seconds=3, fps=30):
        self.buffer = deque(maxlen=max_seconds * fps)
        self.lock = threading.Lock()
        self.fps = fps
        self.is_recording = False

    def add_frame(self, frame):
        with self.lock:
            if self.is_recording:
                self.buffer.append(frame.copy())

    def start_recording(self):
        with self.lock:
            self.buffer.clear()
            self.is_recording = True

    def stop_and_save(self, output_path):
        with self.lock:
            self.is_recording = False
            if not self.buffer:
                return False

            height, width = self.buffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

            for frame in self.buffer:
                out.write(frame)
            out.release()
            return True