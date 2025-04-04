import numpy as np
import cv2

def crop_from_image_query(frame: np.ndarray, bbox) -> np.ndarray:
    """Crops the image using the bounding box from the first ROI in a Groundlight ImageQuery."""

    left, top, right, bottom = bbox.left, bbox.top, bbox.right, bbox.bottom

    # Get image dimensions
    height, width = frame.shape[:2]

    # Convert normalized coords to pixel indices
    x1 = int(left * width)
    y1 = int(top * height)
    x2 = int(right * width)
    y2 = int(bottom * height)

    # Return the cropped image
    return frame[y1:y2, x1:x2]


def draw_bbox(frame: np.ndarray, bbox, color: tuple) -> None:
    """Draws the bounding box from the first ROI in a Groundlight ImageQuery onto the frame."""

    left, top, right, bottom = bbox.left, bbox.top, bbox.right, bbox.bottom

    # Get image dimensions
    height, width = frame.shape[:2]

    # Convert normalized coords to pixel indices
    x1 = int(left * width)
    y1 = int(top * height)
    x2 = int(right * width)
    y2 = int(bottom * height)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)

def resize(frame: np.ndarray, max_width: int = None, max_height: int = None) -> np.ndarray:
    if max_width is None and max_height is None:
        raise ValueError('Please provide either max_height, max_width, or both.')

    h, w = frame.shape[:2]

    if max_width is None:
        # Scale by height
        scale = max_height / float(h)
        dim = (int(w * scale), max_height)
    elif max_height is None:
        # Scale by width
        scale = max_width / float(w)
        dim = (max_width, int(h * scale))
    else:
        # Scale to fit within the specified width and height
        scale_w = max_width / float(w)
        scale_h = max_height / float(h)
        scale = min(scale_w, scale_h)
        dim = (int(w * scale), int(h * scale))

    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized_frame

