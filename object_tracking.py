import math
from model import ROI
import cv2
import numpy as np
from groundlight import ImageQuery

def is_fully_onscreen(bbox) -> bool:
    
    ONSCREEN_MARGIN = 0.005
    
    if bbox.left < ONSCREEN_MARGIN:
        return False
    if bbox.right > 1.0 - ONSCREEN_MARGIN:
        return False
    if bbox.top < ONSCREEN_MARGIN:
        return False
    if bbox.bottom > 1.0 - ONSCREEN_MARGIN:
        return False
    
    return True

class TrackedObject:
    def __init__(self, 
                 roi: ROI, 
                 timestamp: float, 
                 expected_x_velocity: float = 0.0, 
                 expected_y_velocity: float = 0.0) -> None:
        
        self.idx = ObjectTracker.counter; ObjectTracker.counter += 1
        
        # Make a strong assumption that the objects will move in a constant, known direction (this works fine on a conveyor belt)
        self.EXPECTED_X_VELOCITY = expected_x_velocity
        self.EXPECTED_Y_VELOCITY = expected_y_velocity
        
        self.MAX_OBSERVATIONS = 2
        self.observations: list[tuple] = []
        
        self.first_observation = (roi, timestamp)
        
        self._needs_purging = False
        
        self.is_missing = False
        
        self.gl_class = None
        
        self.add_observation(*self.first_observation)
        
    def add_observation(self, roi: ROI, timestamp: float) -> None:
        self.observations.append((roi, timestamp))
        
        if len(self.observations) > self.MAX_OBSERVATIONS:
            self.observations.pop(0)
            
    def current_roi(self) -> ROI:
        return self.observations[-1][0]
    
    def previous_roi(self) -> ROI | None:
        if len(self.observations) < 2:
            return None
        else:
            return self.observations[-2][0]
        
    def mark_for_purging(self) -> None:
        self._needs_purging = True
        
    def needs_purging(self) -> None:
        return self._needs_purging
    
    def distance_traveled(self) -> float:
        bbox1 = self.first_observation[0].geometry
        x1 = bbox1.x
        y1 = bbox1.y

        bbox2 = self.observations[-1][0].geometry
        x2 = bbox2.x
        y2 = bbox2.y

        return math.hypot(x2 - x1, y2 - y1)
            
    def estimate_next_position(self, timestamp: float) -> tuple[float, float] | None:
        """
        timestamp: the current timestamp
        Returns the estimated (x, y) position based on the expected velocity
        and the time elapsed since the last observation.
        """
        if len(self.observations) == 0:
            return None

        previous_roi, previous_timestamp = self.observations[-1]
        previous_bbox = previous_roi.geometry

        # Time since last known observation
        dt = timestamp - previous_timestamp
        if dt < 0:
            return None  # Future timestamp? Skip

        # Estimate position using constant velocity
        estimated_x = previous_bbox.x + self.EXPECTED_X_VELOCITY * dt
        estimated_y = previous_bbox.y + self.EXPECTED_Y_VELOCITY * dt

        return (estimated_x, estimated_y)
    
    def time_since_last_seen(self, timestamp: float) -> float:
        return timestamp - self.observations[-1][1]
            
    def get_velocity(self) -> float | None:
        if len(self.observations) < 2:
            return None

        # Calculate the velocity based on the two most recent ROIs
        curr_roi, curr_timestamp = self.observations[-1]
        prev_roi, prev_timestamp = self.observations[-2]

        time_diff = curr_timestamp - prev_timestamp
        if time_diff <= 0:
            return None  # Avoid divide-by-zero or negative time diff

        displacement = math.sqrt(
            (curr_roi.geometry.x - prev_roi.geometry.x) ** 2 +
            (curr_roi.geometry.y - prev_roi.geometry.y) ** 2
        )

        return displacement / time_diff

        
class ObjectTracker:
    counter = 0 # counts the instances of unique objects the Object Tracker has seen.
    def __init__(self, expected_x_velocity: float = 0.0, expected_y_velocity: float = 0.0) -> None:
        """
        Tracks objects across frames
        """
        self.EXPECTED_X_VELOCITY = expected_x_velocity
        self.EXPECTED_Y_VELOCITY = expected_y_velocity
        
        self.MIN_DISTANCE_TRAVELED_THRESH = 0.5
        
        self.DISTANCE_MATCHING_THRESH = 0.1 # normalized screen units
        self.MAX_TIME_SINCE_LAST_SEEN = 0.5 
        
        self.tracked_objects = []
        
        self.object_count = 0
        
    def add_rois(self, rois: list[ROI], timestamp: float) -> None:
        # Initialize all the objects as missing, we'll mark them as not missing if/when we find them
        for tracked_object in self.tracked_objects:
            tracked_object.is_missing = True
        
        # Attempt to match each ROI with a previously tracked object
        for roi in rois:
            bbox = roi.geometry
            
            # If it's not fully onscreen, we can't see it well enough to estimate its position, so we'll just skip it
            if not is_fully_onscreen(bbox):
                continue
            
            for tracked_object in self.tracked_objects:
                estimated_next_pos = tracked_object.estimate_next_position(timestamp)
                
                distance = math.sqrt((estimated_next_pos[0] - bbox.x) ** 2 + (estimated_next_pos[1] - bbox.y) ** 2)
                
                if distance < self.DISTANCE_MATCHING_THRESH:
                    tracked_object.add_observation(roi, timestamp)
                    tracked_object.is_missing = False
                    break
            # If the current ROI can't be matched to any previously tracked object, create a new tracked object
            else:
                tracked_object = TrackedObject(roi, timestamp, self.EXPECTED_X_VELOCITY, self.EXPECTED_Y_VELOCITY)
                self.tracked_objects.append(tracked_object)

        # Check for objects that needs to be purged (have been missing too long)
        for tracked_object in self.tracked_objects:
            time_since_last_seen = tracked_object.time_since_last_seen(timestamp)
            if time_since_last_seen > self.MAX_TIME_SINCE_LAST_SEEN:
                tracked_object.mark_for_purging()
    
    def purge_missing_objects(self) -> None:
        """
        Remove tracked objects that have been marked for purging and update the object count.
        
        This method performs cleanup by:
        1. Removing tracked objects that have been missing for too long (marked for purging)
        2. Incrementing the total object count for objects that traveled sufficient distance
        before being purged (indicating they were legitimate objects that completed
        their journey across the tracking area)
        
        Only objects that traveled more than MIN_DISTANCE_TRAVELED_THRESH are counted
        toward the final object count, filtering out noise, false detections, or objects
        that were only briefly visible.
        
        This method should be called after add_rois() to clean up stale tracking data
        and maintain accurate object counts for conveyor belt or similar applications
        where
        """
        
        tracked_objects = []
        for tracked_object in self.tracked_objects:
            if not tracked_object.needs_purging():
                tracked_objects.append(tracked_object)
            else:
                distance_traveled = tracked_object.distance_traveled()
                
                if distance_traveled > self.MIN_DISTANCE_TRAVELED_THRESH:
                    self.object_count += 1
                
        self.tracked_objects = tracked_objects
        
    def annotate_frame(self, frame: np.ndarray) -> None:
        """
        Draw bounding boxes around currently tracked objects onto the frame.
        Assumes bbox has normalized coordinates (left, top, right, bottom in 0.0–1.0).
        """
        height, width = frame.shape[:2]
        thickness = 2
        
        # Draw a solid white rectangle under the text, make it the same size as the text, but with a little margin
        text = f'Object count: {self.object_count}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness, margin = 1.0, 2, 5
        org = (10, 30)

        size, baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = org
        cv2.rectangle(frame, (x - margin, y - size[1] - margin), (x + size[0] + margin, y + baseline + margin), (255, 255, 255), -1)
        cv2.putText(frame, text, org, font, scale, (0, 255, 0), thickness)

        for tracked_object in self.tracked_objects:
            
            current_roi = tracked_object.current_roi()
            bbox = current_roi.geometry

            # Convert normalized coords to pixel coords
            x1 = int(bbox.left * width)
            y1 = int(bbox.top * height)
            x2 = int(bbox.right * width)
            y2 = int(bbox.bottom * height)
            
            # Draw the previous bounding box
            white = (255, 255, 255)
            previous_roi = tracked_object.previous_roi()
            if previous_roi is not None:
                previous_bbox = previous_roi.geometry
                x1_prev = int(previous_bbox.left * width)
                y1_prev = int(previous_bbox.top * height)
                x2_prev = int(previous_bbox.right * width)
                y2_prev = int(previous_bbox.bottom * height)

                cv2.rectangle(frame, (x1_prev, y1_prev), (x2_prev, y2_prev), white, 1)

                cv2.line(frame, (x1_prev, y1_prev), (x1, y1), white, 1)  # Top-left
                cv2.line(frame, (x2_prev, y1_prev), (x2, y1), white, 1)  # Top-right
                cv2.line(frame, (x1_prev, y2_prev), (x1, y2), white, 1)  # Bottom-left
                cv2.line(frame, (x2_prev, y2_prev), (x2, y2), white, 1)  # Bottom-right
                
            if tracked_object.needs_purging():
                color = (0, 0, 0)
                
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.line(frame, (x1, y2), (x2, y1), color, thickness)
            elif tracked_object.is_missing:
                color = (0, 0, 0)
            else:
                color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label with ID
            velocity = tracked_object.get_velocity()
            velocity_str = "-" if velocity is None else f"{velocity:.4f}"
            label = f"ID: {tracked_object.idx} | velocity: {velocity_str}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    def run(self, iq: ImageQuery, timestamp: float, annotated_frame: np.ndarray) -> None:
        rois = [] if iq.rois is None else iq.rois
        self.add_rois(rois, timestamp)
        self.annotate_frame(annotated_frame)
        self.purge_missing_objects()