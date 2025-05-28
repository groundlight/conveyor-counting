import groundlight
import framegrab
from imgcat import imgcat

import time
import logging
import argparse

import image_utils as iu
import object_utils as ou
import camera as cam
from timing import PerfTimer, LoopManager

from framegrab_web_server import FrameGrabWebServer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level'
    )
    return parser.parse_args()

def main() -> None:
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)

    # POSITION_DETECTOR_ID = 'det_2v99nYk9fXp7vobIx6YliqnH9gP'
    POSITION_DETECTOR_ID = 'det_2vCD8IJ7iHVoP14RsOjnZiGGjkT'
    BINARY_DEFECT_DETECTOR_ID = 'det_2v98BBiviD4wSq5eenGnePJv2hH'

    logger.info(f'Groundlight Version: {groundlight.__version__}')
    logger.info(f'Framegrab Version: {framegrab.__version__}')

    gl = groundlight.ExperimentalApi(endpoint="http://localhost:30101/")
    logged_in_user = gl.whoami()

    POSITION_DETECTOR = gl.get_detector(POSITION_DETECTOR_ID)
    BINARY_DEFECT_DETECTOR = gl.get_detector(BINARY_DEFECT_DETECTOR_ID)

    MAIN_LOOP_TIME = 1 / 5

    logger.info(f'Welcome, {logged_in_user}')

    options = {
            "resolution": {
                "width": 3840,
                "height": 2160,
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
        "options": options}
    blocking_grabber = framegrab.FrameGrabber.create_grabber(config, warmup_delay=0.0)

    blocking_grabber.apply_options(options)

    grabber = cam.ThreadedFrameGrabber(blocking_grabber)

    web_server = FrameGrabWebServer('Lid Detector')
    
    # image_capture_timer = PerfTimer('Image Capture')
    resizing_timer = PerfTimer('Resizing', False)
    # display_timer = PerfTimer('Display')
    od_inference_timer = PerfTimer('Object Detection Inference', False)
    # main_loop_timer = PerfTimer('Main Loop')

    main_loop_manager = LoopManager('Main Loop', loop_time=MAIN_LOOP_TIME)
    
    logger.info('Starting main loop...')
    while True:
        # main_loop_timer.start()
        main_loop_manager.start()
        
        # image_capture_timer.start()
        frame = grabber.grab()
        # image_capture_timer.stop()
        
        if frame is None:
            time.sleep(1)
            continue
        
        resizing_timer.start()
        annotated_frame = iu.resize(frame, max_width=640)
        object_detection_frame = iu.resize(annotated_frame, max_width=200)
        resizing_timer.stop()
        
        od_inference_timer.start()
        position_iq = gl.ask_ml(POSITION_DETECTOR, object_detection_frame)
        od_inference_timer.stop()
        
        
        # display_timer.start()
        rois = [] if position_iq.rois is None else position_iq.rois
        for roi in rois:
            bbox = roi.geometry
            
            is_fully_onscreen = ou.is_fully_onscreen(bbox)
            color = (0, 0, 0) if is_fully_onscreen else (255, 255, 255)
            
            if is_fully_onscreen:
                cropped_frame = iu.crop_from_image_query(frame, bbox)
                # defect_iq = gl.ask_ml(BINARY_DEFECT_DETECTOR, cropped_frame)
                
            #     answer = defect_iq.result.label.value
            #     confidence = defect_iq.result.confidence
            #     if confidence > BINARY_DEFECT_DETECTOR.confidence_threshold:
            #         if answer == "YES":
            #             color = (0, 255, 0)
            #         elif answer == "NO":
            #             color = (0, 0, 255)
            #         else:
            #             color = (0, 255, 255)
            #     else:
            #         color = (0, 255, 255)
            
            iu.draw_bbox(annotated_frame, bbox, color)
        # if cropped_frame is not None: 
        #     web_server.show_image(cropped_frame)
        # else:
        #     web_server.show_image(annotated_frame)
            
        web_server.show_image(annotated_frame)
        # display_timer.stop()
        
        main_loop_manager.wait()
        # main_loop_timer.stop()

if __name__ == "__main__":
    main()