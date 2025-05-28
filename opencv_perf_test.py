import cv2
import framegrab
import time
from imgcat import imgcat

ITERATIONS = 100
WIDTH = 1920 # 3840
HEIGHT = 1080 # 2160

print(framegrab.__version__)

print('Starting framegrab test')
options = {
        "resolution": {
            "width": WIDTH,
            "height": HEIGHT,
        },
        "crop": {
            "relative": {
                "left": 0.2,
                "right": 0.9,
            }
        },
        "num_90_deg_rotations": 2,
        "is_video": True
    }
config = {
    "input_type": "generic_usb",
    "options": options,
    }
grabber = framegrab.FrameGrabber.create_grabber(config, warmup_delay=0.0)

# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
# grabber.capture.set(cv2.CAP_PROP_FOURCC, fourcc)

# # Request 4K resolution
# grabber.capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# grabber.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Check actual resolution
width = grabber.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = grabber.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Actual resolution: {int(width)}x{int(height)}")
print(f'grabber.idx: {grabber.idx}')

t1 = time.perf_counter()
for n in range(ITERATIONS):
   frame = grabber.grab()
t2 = time.perf_counter()
print(1 / ((t2 - t1) / ITERATIONS))
print(frame.shape)
imgcat(frame[:, :, ::-1])

grabber.release()

print('Starting OpenCV test')

cap = cv2.VideoCapture(4)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# Request 4K resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # this slows things down for sure

# Check actual resolution
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Actual resolution: {int(width)}x{int(height)}")


t1 = time.perf_counter()
for n in range(ITERATIONS):
    _, frame = cap.read()
t2 = time.perf_counter()
print(1 / ((t2 - t1) / ITERATIONS)) # we are achieving a frame rate of nearly 30 fps
print(frame.shape) 
imgcat(frame[:, :, ::-1])

cap.release()


