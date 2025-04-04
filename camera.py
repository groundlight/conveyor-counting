from framegrab import FrameGrabber
import cv2
import numpy as np
import logging

from threading import Thread

from timing import LoopManager

FPS = 5
CAMERA_LOOP_TIME = 1 / FPS

logger = logging.getLogger(__name__)
    
class ThreadedFrameGrabber:
    def __init__(self, grabber: FrameGrabber, fps: int = 10) -> None:
        self._grabber = grabber
        self._frame = None
        
        self._wait_time = 1 / fps
        
        self._start()
        
    def grab(self) -> np.ndarray:
        return self._frame
    
    def _start(self) -> None:
        def thread() -> None:
            camera_loop = LoopManager('Camera Loop', CAMERA_LOOP_TIME)
            self._running = True
            while self._running:
                camera_loop.start()
                self._frame = self._grabber.grab()
                camera_loop.wait()
                
            self._grabber.release()
        
        t = Thread(target=thread)
        t.daemon = True
        t.start()
    
    def release(self) -> None:
        self._running = False
        