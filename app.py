import groundlight
import framegrab

import time
import logging
import argparse

import image_utils as iu
import object_utils as ou
import camera as cam
from timing import PerfTimer, LoopManager
import yaml

from framegrab_web_server import FrameGrabWebServer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level'
    )
    parser.add_argument(
        '--inference',
        default=False,
        help='Run inference on the camera stream',
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
    
    if args.inference:
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
            
        # TODO - add this to config or autogenerate them
        COLORS = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
        ]
    else:
        logger.info('Inference disabled. Streaming camera only.')
    
    MAIN_LOOP_TIME = 1 / 3

    # Connect to the camera and create a threaded framegrabber so we can capture frames more efficiently
    blocking_grabber = framegrab.FrameGrabber.from_yaml(yaml_path)[0]
    grabber = cam.ThreadedFrameGrabber(blocking_grabber)

    web_server = FrameGrabWebServer('Object Counter')
    
    main_loop_manager = LoopManager('Main Loop', loop_time=MAIN_LOOP_TIME)
    
    logger.info('Starting main loop...')
    while True:
        main_loop_manager.start()
        
        frames = grabber.grab()
        
        if frames is None:
            time.sleep(1)
            continue
        
        annotated_frame = frames['annotated']
        
        if args.inference:
            object_detection_frame = frames['object_detection']
            
            object_detection_iqs = []
            for detector, timer  in zip(object_detectors, object_detector_timers):
                timer.start()
                iq = gl.ask_ml(detector, object_detection_frame)
                timer.stop()
                object_detection_iqs.append(iq)
                
            for iq, color in zip(object_detection_iqs, COLORS):
                rois = [] if iq.rois is None else iq.rois
                for roi in rois:
                    bbox = roi.geometry
                    
                    if ou.is_fully_onscreen(bbox):
                        iu.draw_bbox(annotated_frame, bbox, color)
            
        web_server.show_image(annotated_frame)
        
        main_loop_manager.wait()

if __name__ == "__main__":
    main()