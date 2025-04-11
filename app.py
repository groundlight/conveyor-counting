import groundlight
import framegrab

import time
import logging
import argparse

import object_tracking as ot
import camera as cam
from timing import PerfTimer, LoopManager
import yaml
from enum import Enum

from framegrab_web_server import FrameGrabWebServer

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
            
        counting_detector_id = config["detector_ids"]["counting"]
        counting_detector = gl.get_detector(counting_detector_id)
        counting_timer = PerfTimer("Counting", False)
        
        # Multiple choice is not currently supported
        # multiple_choice_detector_id = config["detector_ids"]["multiple_choice"]
        # multiple_choice_detector = gl.get_detector(multiple_choice_detector_id)
        # counting_timer = PerfTimer("Counting", False)
        
    else:
        logger.info('Inference disabled. Streaming camera only.')
        
    if args.mode == AppMode.VIDEO_INFERENCE:
        object_tracker = ot.ObjectTracker(0.3, 0.0)
    
    FPS = 5
    MAIN_LOOP_TIME = 1 / FPS

    # Connect to the camera and create a threaded framegrabber so we can capture frames more efficiently
    blocking_grabber = framegrab.FrameGrabber.from_yaml(yaml_path)[0]
    grabber = cam.ThreadedFrameGrabber(blocking_grabber, FPS)

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
            
            counting_timer.start()
            try:
                iq = gl.ask_ml(counting_detector, object_detection_frame)
            except KeyboardInterrupt:
                break
            except Exception:
                logger.error(f'Encountered an unexpected error while performing inference', exc_info=True)
                continue
            finally:
                counting_timer.stop()
            
            if args.mode == AppMode.VIDEO_INFERENCE:
                object_tracker.run(iq, timestamp, annotated_frame)
                
            # scores = [roi.score for roi in rois]    
            # logger.info(f"Confidence: {iq.result.confidence} | {scores}")
            
        web_server.show_image(annotated_frame)
        
        # debug_str = None if iq is None else iq.id
        # debug_str = iq.metadata["is_from_edge"]
        # debug_str = str(iq)
        main_loop_manager.wait()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")