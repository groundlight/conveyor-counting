from framegrab import FrameGrabber
import numpy as np
import cv2
import logging

import image_utils as iu

from threading import Thread

from timing import LoopManager

FPS = 4
CAMERA_LOOP_TIME = 1 / FPS

logger = logging.getLogger(__name__)
    
class ThreadedFrameGrabber:
    def __init__(self, grabber: FrameGrabber, fps: int = 10) -> None:
        # self._setup_camera(grabber)
        self._grabber = grabber
        self._frames: dict[str, np.ndarray] = None
        
        self._wait_time = 1 / fps
        
        self._start()
        
    def grab(self) -> dict[str, np.ndarray]:
        return self._frames
    
    def _setup_camera(self, grabber: FrameGrabber) -> None:
        """
        Enable 4K, set a reasonable frame rate, etc.
        """
        
        # MJPG enables 4K on cameras like the logitech brio
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
        grabber.capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        fps = grabber.capture.get(cv2.CAP_PROP_FPS)
        logger.info(f"Original camera frame rate for {grabber.config['name']}: {fps} FPS")
        
        desired_fps = 30
        grabber.capture.set(cv2.CAP_PROP_FPS, desired_fps)
        
        new_fps = grabber.capture.get(cv2.CAP_PROP_FPS)
        logger.info(f"New camera frame rate for {grabber.config['name']}: {new_fps} FPS")
    
    def _start(self) -> None:
        def thread() -> None:
            camera_loop = LoopManager('Camera Loop', CAMERA_LOOP_TIME)
            self._running = True
            while self._running:
                camera_loop.start()
                
                frame = self._grabber.grab()
                self._resize_in_thread(frame)
                
                camera_loop.wait()
                
            self._grabber.release()
        
        t = Thread(target=thread)
        t.daemon = True
        t.start()
        
    def _resize_in_thread(self, frame: np.ndarray) -> None:
        def thread() -> None:
            annotated = iu.resize(frame, max_width=1280)
            object_detection = iu.resize(annotated, max_width=200)
            self._frames = {
                'original': frame,
                'annotated': annotated,
                'object_detection': object_detection,
            }
        t = Thread(target=thread)
        t.daemon = True
        t.start()
    
    def release(self) -> None:
        self._running = False
        
    
        