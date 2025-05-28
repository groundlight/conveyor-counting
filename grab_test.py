import cv2
import framegrab
import time
from imgcat import imgcat

WIDTH = 3840
HEIGHT = 2160

config = {
    "input_type": "generic_usb",
    # "options": options,
    }
grabber = framegrab.FrameGrabber.create_grabber(config, warmup_delay=0.0)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
grabber.capture.set(cv2.CAP_PROP_FOURCC, fourcc)

# Request 4K resolution
grabber.capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
grabber.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Check actual resolution
width = grabber.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = grabber.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Actual resolution: {int(width)}x{int(height)}")

while True:
    print('-')
    input('Press enter: ')
    
    t1 = time.perf_counter()
    frame = grabber.grab()
    t2 = time.perf_counter()
    print(t2 - t1)
    print((1 / (t2 - t1)))
    print(frame.shape)
    
    imgcat(frame[:, :, ::-1])