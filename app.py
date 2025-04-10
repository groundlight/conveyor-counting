import groundlight
import framegrab

import time
import logging
import argparse

import image_utils as iu
import object_tracking as ot
import camera as cam
from timing import PerfTimer, LoopManager
import yaml
from enum import Enum

from framegrab_web_server import FrameGrabWebServer

import cv2

class AppMode(str, Enum):
    VIDEO_ONLY = "VIDEO_ONLY"
    VIDEO_INFERENCE = "VIDEO_INFERENCE"
    SNAPSHOT_INFERENCE = "SNAPSHOT_INFERENCE"

    def get_values() -> list[str]:
        return [value.value for value in AppMode]
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level'
    )
    parser.add_argument(
        '--mode',
        default=AppMode.get_values()[0],
        choices=AppMode.get_values(),
        help='The mode of the application',
    )    
    
    return parser.parse_args()

def main() -> None:
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)

    logger.info(f'Groundlight Version: {groundlight.__version__}')
    logger.info(f'Framegrab Version: {framegrab.__version__}')
    
    yaml_path = 'config.yaml'
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if args.mode in (AppMode.VIDEO_INFERENCE, AppMode.SNAPSHOT_INFERENCE):
        gl = groundlight.ExperimentalApi(endpoint="http://localhost:30101/")
        logged_in_user = gl.whoami()
        logger.info(f'Welcome, {logged_in_user}')
            
        OBJECT_DETECTOR_IDS = config["detector_ids"]["object_detection"]
        
        object_detectors = []
        object_detector_timers = []
        for detector_id in OBJECT_DETECTOR_IDS:
            
            detector = gl.get_detector(detector_id)
            object_detectors.append(detector)
            
            timer_name = f'{detector.name} Inference'
            timer = PerfTimer(timer_name, False)
            object_detector_timers.append(timer)
    else:
        logger.info('Inference disabled. Streaming camera only.')
        
    if args.mode == AppMode.VIDEO_INFERENCE:
        object_tracker = ot.ObjectTracker(0.0, -0.7)
    
    MAIN_LOOP_TIME = 1 / 5

    # Connect to the camera and create a threaded framegrabber so we can capture frames more efficiently
    blocking_grabber = framegrab.FrameGrabber.from_yaml(yaml_path)[0]
    grabber = cam.ThreadedFrameGrabber(blocking_grabber)

    web_server = FrameGrabWebServer('Object Counter')
    
    main_loop_manager = LoopManager('Main Loop', loop_time=MAIN_LOOP_TIME)
    
    while True:
        if args.mode == AppMode.SNAPSHOT_INFERENCE:
            input('Press enter to perform inference: ')
            
        main_loop_manager.start()
        
        frames, timestamp = grabber.grab()
        
        if frames is None:
            time.sleep(1)
            continue
        
        annotated_frame = frames['annotated']
        
        if args.mode in (AppMode.VIDEO_INFERENCE, AppMode.SNAPSHOT_INFERENCE):
            object_detection_frame = frames['object_detection']
            
            detector = object_detectors[0]
            timer = object_detector_timers[0]
            timer.start()
            try:
                iq = gl.ask_ml(detector, object_detection_frame)
            except:
                logger.error(f'Encountered an error while performing inference', exc_info=True)
                
            timer.stop()
            
            rois = [] if iq.rois is None else iq.rois
            
            object_tracker.add_rois(rois, timestamp)
            object_tracker.annotate_frame(annotated_frame)
            object_tracker.purge_missing_objects()
            
            # scores = [roi.score for roi in rois]    
            # logger.info(f"Confidence: {iq.result.confidence} | {scores}")
            
        web_server.show_image(annotated_frame)
        
        main_loop_manager.wait()

if __name__ == "__main__":
    main()