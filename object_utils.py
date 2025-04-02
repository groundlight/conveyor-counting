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