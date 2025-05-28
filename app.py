import groundlight
import framegrab

import time
import logging
import argparse

import object_tracking as ot
import camera as cam
from timing import PerfTimer, LoopManager
import yaml
from enums import AppMode, RecordingMode
from datetime import datetime

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
        '--app-mode',
        default=AppMode.get_default(),
        choices=AppMode.get_values(),
        help='The mode of the application',
    )    
    parser.add_argument(
        '--recording-mode',
        default=RecordingMode.get_default(),
        choices=RecordingMode.get_values(),
        help='NONE: do not record a video, RAW: record without bounding boxes and other annotations, ANNOTATED: record with annotations',
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
        
    if args.app_mode in (AppMode.VIDEO_INFERENCE, AppMode.SNAPSHOT_INFERENCE):
        gl = groundlight.ExperimentalApi(endpoint="http://localhost:30101/")
        logged_in_user = gl.whoami()
        logger.info(f'Welcome, {logged_in_user}')
            
        counting_detector_id = config["detector_ids"]["counting"]
        counting_detector = gl.get_detector(counting_detector_id)
        counting_timer = PerfTimer("Counting", False)
    else:
        logger.info('Inference disabled. Streaming camera only.')
        
    if args.app_mode == AppMode.VIDEO_INFERENCE:
        object_tracker = ot.ObjectTracker(0.4, 0.0)
    
    FPS = config.get('fps')
    if FPS is None:
        raise ValueError(
            f'No fps was provided in {yaml_path}. Please provided one. Exiting.'
        )
    else:
        logger.info(f'Running at {FPS} frames per second.')
    
        
    MAIN_LOOP_TIME = 1 / FPS

    # Connect to the camera and create a threaded framegrabber so we can capture frames more efficiently
    blocking_grabber = framegrab.FrameGrabber.from_yaml(yaml_path)[0]
    grabber = cam.ThreadedFrameGrabber(blocking_grabber, FPS)

    web_server = FrameGrabWebServer('Object Counter')
    
    # Get the first frames from the camera to initialize the display and the video writer (if necessary)
    for _ in range(100):
        time.sleep(.01)
        frames, timestamp = grabber.grab()
        if frames is not None:
            annotated_frame = frames['original']
            web_server.show_image(annotated_frame)
            logger.info('Got first frames from the camera.')
            break
    else:
        logger.error('Could not get frames from the camera. Exiting.')
        exit(1)
    
    # Start the video writer, if necessary
    if args.recording_mode in (RecordingMode.RAW, RecordingMode.ANNOTATED):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.recording_mode == RecordingMode.ANNOTATED:
            video_type = 'annotated'
            name = f'{timestamp}_annotated'
        elif args.recording_mode == RecordingMode.RAW:
            video_type = 'raw'
            name = f'{timestamp}_raw'
        else:
            raise ValueError('Unexpected value for recording mode')
        
        frame = frames['original']
        resolution = (frame.shape[1], frame.shape[0])
        video_writer = cam.ThreadedVideoWriter(name, resolution, FPS)
        
        logger.info(f'Recording {video_type} video to {video_writer.filename}.')
    else:
        video_writer = None
        logger.info('Not recording video.')
        
    main_loop_manager = LoopManager('Main Loop', loop_time=MAIN_LOOP_TIME)
    
    try:
        while True:
            if args.app_mode == AppMode.SNAPSHOT_INFERENCE:
                input('Press enter to perform inference: ')
                
            main_loop_manager.start()
            
            frames, timestamp = grabber.grab()
            
            annotated_frame = frames['annotated']
            
            # Peform inference
            if args.app_mode in (AppMode.VIDEO_INFERENCE, AppMode.SNAPSHOT_INFERENCE):
                object_detection_frame = frames['object_detection']
                
                counting_timer.start()
                try:
                    iq = gl.ask_ml(counting_detector, object_detection_frame)
                except Exception:
                    logger.error(f'Encountered an unexpected error while performing inference', exc_info=True)
                    continue
                finally:
                    counting_timer.stop()
                
                if args.app_mode == AppMode.VIDEO_INFERENCE:
                    object_tracker.run(iq, timestamp, annotated_frame)
                    
            # Record       
            if args.recording_mode == RecordingMode.NONE:
                pass
            elif args.recording_mode == RecordingMode.RAW:
                original_frame = frames['original']
                video_writer.add_frame(original_frame)
            elif args.recording_mode == RecordingMode.ANNOTATED:
                video_writer.add_frame(annotated_frame)
            else:
                raise ValueError(
                    f'Unexpected value for recording mode: {args.recording_mode}'
                )
            
            # show the result       
            web_server.show_image(annotated_frame)
            
            main_loop_manager.wait()
            
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
    finally:
        if video_writer is not None:
            video_writer.stop()

if __name__ == "__main__":
    main()