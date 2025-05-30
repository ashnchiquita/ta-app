from components.roi_selector.base import ROISelector
import cv2

class Forehead(ROISelector):
    def select(self, frame, face_rect):
        x, y, x_end, y_end = face_rect
        face_width = x_end - x
        face_height = y_end - y
        
        # TODO: Adjust these values
        forehead_x = x + int(face_width * 0.2)
        forehead_width = int(face_width * 0.6)
        forehead_y = y + int(face_height * 0.05)
        forehead_height = int(face_height * 0.25)
        
        forehead = frame[forehead_y:forehead_y+forehead_height, forehead_x:forehead_x+forehead_width]
        
        if forehead.size > 0:
            return cv2.resize(forehead, self.target_size)
        
        return None
