from framegrab import FrameGrabber
import numpy as np
import cv2
import logging
import time
import os

import image_utils as iu

from threading import Thread
from queue import Queue, Full, Empty

from timing import LoopManager

logger = logging.getLogger(__name__)

class ThreadedVideoWriter:
    def __init__(self, name: str, resolution: tuple, fps: int) -> None:
        """
        Records video in a separate thread to improve performance.
        """
        self.name = name
        self.resolution = resolution
        self.fps = fps
        
        directory = 'video_output'
        os.makedirs(directory, exist_ok=True)
        self.filename = os.path.join(directory, f"{name}.mp4")

        self.queue = Queue(maxsize=10)
        self.writer = cv2.VideoWriter(
            filename=self.filename,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=fps,
            frameSize=resolution,
        )
        self.run = False
        
        self.thread = Thread(target=self._run_loop, daemon=True)

        self.start()

    def add_frame(self, frame: np.ndarray) -> None:
        try:
            self.queue.put_nowait(frame)
        except Full:
            logger.error("Video recorder queue full! Dropping frame.")

    def start(self) -> None:
        self.run = True
        self.thread.start()

    def stop(self) -> None:
        self.run = False
        self.thread.join()
        self.writer.release()
        
        logger.info('Video recording completed.')

    def _run_loop(self) -> None:
        while self.run or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
            except Empty:
                continue  # No frame to write, loop again
            
            time.sleep(0.01)
            self.writer.write(frame)
        
class ThreadedFrameGrabber:
    def __init__(self, grabber: FrameGrabber, fps: int = 10) -> None:
        """
        A wrapper around framegrab.FrameGrabber that improves performance while streaming video
        """
        self._setup_camera(grabber)
        self._grabber = grabber
        self._frames: dict[str, np.ndarray] = None
        
        self._wait_time = 1 / fps
        
        self.timestamp = 0.0
        
        self._start()
        
    def grab(self) -> tuple[dict[str, np.ndarray], float]:
        return self._frames, self.timestamp
    
    def _setup_camera(self, grabber: FrameGrabber) -> None:
        """
        Enable 4K, set a reasonable frame rate, etc.
        """
        
        # MJPG enables 4K on cameras like the logitech brio
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
        grabber.capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        fps = grabber.capture.get(cv2.CAP_PROP_FPS)
        logger.info(f"Original camera frame rate for {grabber.config.name}: {fps} FPS")
        
        desired_fps = 30
        grabber.capture.set(cv2.CAP_PROP_FPS, desired_fps)
        
        new_fps = grabber.capture.get(cv2.CAP_PROP_FPS)
        logger.info(f"New camera frame rate for {grabber.config.name}: {new_fps} FPS")
    
    def _start(self) -> None:
        def thread() -> None:
            camera_loop = LoopManager('Camera Loop', self._wait_time)
            self._running = True
            while self._running:
                camera_loop.start()
                
                frame = self._grabber.grab()
                timestamp = time.perf_counter() # capture the timestamp right after grabbing the frame
                
                self._resize_in_thread(frame, timestamp)
                
                camera_loop.wait()
                
            self._grabber.release()
        
        t = Thread(target=thread)
        t.daemon = True
        t.start()
        
    def _resize_in_thread(self, frame: np.ndarray, timestamp: float) -> None:
        def thread() -> None:
            object_detection_frame = iu.resize(frame, max_width=200)
            
            self._frames = {
                'original': frame,
                'annotated': frame.copy(),
                'object_detection': object_detection_frame,
            }
            self.timestamp = timestamp
                
        t = Thread(target=thread)
        t.daemon = True
        t.start()
    
    def release(self) -> None:
        self._running = False
        
    
        