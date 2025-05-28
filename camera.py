from framegrab import FrameGrabber
import cv2

def enable_4k(grabber: FrameGrabber) -> None:
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
    grabber.capture.set(cv2.CAP_PROP_FOURCC, fourcc)