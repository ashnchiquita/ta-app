import cv2
from components.roi_selector.base import ROISelector

class FullFaceSquare(ROISelector):
    def __init__(self, target_size=(72, 72), larger_box_coef=1.0):
        self.target_size = target_size
        self.larger_box_coef = larger_box_coef
        
    def select(self, frame, face_rect):
        x, y, x_end, y_end = face_rect
        w, h = x_end - x, y_end - y
        
        # Calculate center of original face rectangle
        center_x = x + w // 2
        center_y = y + h // 2

        # Make the box square using the larger dimension
        box_size = max(w, h)
        
        # Apply larger box coefficient if specified
        if self.larger_box_coef > 1.0:
            box_size = int(box_size * self.larger_box_coef)
        
        # Calculate new square box coordinates centered on face
        half_size = box_size // 2
        new_x = center_x - half_size
        new_y = center_y - half_size
        new_x_end = new_x + box_size
        new_y_end = new_y + box_size
        
        # Ensure the box doesn't overflow frame bounds
        frame_h, frame_w, _ = frame.shape
                
        # Adjust if box goes beyond frame boundaries
        if new_x < 0:
            new_x_end -= new_x  # shift right
            new_x = 0
        elif new_x_end > frame_w:
            new_x -= (new_x_end - frame_w)  # shift left
            new_x_end = frame_w
            
        if new_y < 0:
            new_y_end -= new_y  # shift down
            new_y = 0
        elif new_y_end > frame_h:
            new_y -= (new_y_end - frame_h)  # shift up
            new_y_end = frame_h
        
        # Final bounds check
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_x_end = min(frame_w, new_x_end)
        new_y_end = min(frame_h, new_y_end)
        
        # Extract ROI
        roi = frame[new_y:new_y_end, new_x:new_x_end]
        
        # Resize to target size using OpenCV (ROI is guaranteed to be square)
        if roi.shape[:2] != self.target_size:
            roi = cv2.resize(roi, self.target_size, interpolation=cv2.INTER_AREA)

        if roi.size == 0:
            return None, None

        return roi, (new_x, new_y, new_x_end, new_y_end)
