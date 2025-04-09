from typing import List, Tuple

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

def iou(box1, box2) -> float:
    # Intersection over Union (IoU) between two bounding boxes
    x_left = max(box1.left, box2.left)
    y_top = max(box1.top, box2.top)
    x_right = min(box1.right, box2.right)
    y_bottom = min(box1.bottom, box2.bottom)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0  # no overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1.right - box1.left) * (box1.bottom - box1.top)
    box2_area = (box2.right - box2.left) * (box2.bottom - box2.top)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area


def dedup_bboxes(bboxes1: List[object], bboxes2: List[object], max_overlap: float = 0.75) -> Tuple[List[object], List[object], List[object]]:
    filtered_bboxes2 = []
    removed_bboxes = []

    for b2 in bboxes2:
        overlaps = [iou(b1, b2) for b1 in bboxes1]
        if all(overlap < max_overlap for overlap in overlaps):
            filtered_bboxes2.append(b2)
        else:
            removed_bboxes.append(b2)

    return bboxes1, filtered_bboxes2, removed_bboxes
