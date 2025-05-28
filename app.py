import groundlight
import framegrab
from imgcat import imgcat

import math

import image_utils as iu
import object_utils as ou
import camera as cam

from framegrab_web_server import FrameGrabWebServer

POSITION_DETECTOR_ID = 'det_2v99nYk9fXp7vobIx6YliqnH9gP'
BINARY_DEFECT_DETECTOR_ID = 'det_2v98BBiviD4wSq5eenGnePJv2hH'


print(f'Groundlight Version: {groundlight.__version__}')
print(f'Framegrab Version: {framegrab.__version__}')

gl = groundlight.ExperimentalApi(endpoint="http://localhost:30101/")
logged_in_user = gl.whoami()

POSITION_DETECTOR = gl.get_detector(POSITION_DETECTOR_ID)
BINARY_DEFECT_DETECTOR = gl.get_detector(BINARY_DEFECT_DETECTOR_ID)

print(f'Welcome, {logged_in_user}')

options = {
        "resolution": {
            "width": 3840,
            "height": 2160,
        },
        "num_90_deg_rotations": 2,
    }
config = {
    "input_type": "generic_usb",
    "options": options}
grabber = framegrab.FrameGrabber.create_grabber(config, warmup_delay=0.0)
cam.enable_4k(grabber)

grabber.apply_options(options)

web_server = FrameGrabWebServer('Lid Detector')

while True:
    user_input = input(
        "Enter 'q' to quit. "
        "Enter 'c' to capture an image."
        "Enter any other key to capture an image and perform inference: "
        ).lower()
    
    if user_input == "q":
        print('Quitting...')
        break
    
    print('Capturing image...', end='')
    frame = grabber.grab()
    print(f'Image captured: {frame.shape}')
    
    frame_small = iu.resize(frame, max_width=640)
    imgcat(frame_small)
    print('-' * 50)
    
    if user_input == 'c':
        continue

    print(f'Scanning for lids...')
    # Scale down to a reasonable size for the object detector
    position_detector_frame = iu.resize(frame, max_width=200)
    print(f'position_detector_frame: {position_detector_frame.shape}')
    position_iq = gl.ask_ml(POSITION_DETECTOR, position_detector_frame)
    
    if not position_iq.rois:
        print('No lids detected in this frame.')
        continue
    
    for roi in position_iq.rois:
        bbox = roi.geometry
        is_fully_onscreen = ou.is_fully_onscreen(bbox)
        
        color = (0, 255, 0) if is_fully_onscreen else (255, 0, 0)

        cropped_frame = iu.crop_from_image_query(frame, bbox)
        cropped_frame_small = iu.resize(cropped_frame, max_width=640)
        
        annotated_frame = iu.draw_bbox(frame, bbox, color)
        annotated_frame_small = iu.resize(annotated_frame, max_width=640)
        
        web_server.show_image(annotated_frame_small)
        imgcat(annotated_frame_small)
        print('-' * 50)
        
        if is_fully_onscreen:
            detect_iq = gl.ask_ml(BINARY_DEFECT_DETECTOR, cropped_frame)
            
            answer = detect_iq.result.label.value
            confidence = detect_iq.result.confidence
            confidence_percentage = math.floor(confidence * 100)
            print(f'Cropped image size: {cropped_frame.shape}')
            print(f'Defect result: {answer} | confidence: {confidence_percentage}%')
            imgcat(cropped_frame_small)
            print('-' * 50)
        else:
            print('This lid is not fully onscreen. Cannot detect defects.')

print('Demo finished.')
grabber.release()