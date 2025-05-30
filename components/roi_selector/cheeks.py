import cv2
import numpy as np
from components.roi_selector.base import ROISelector

class Cheeks(ROISelector):
    def select(self, frame, face_rect):
        x, y, x_end, y_end = face_rect
        face_width = x_end - x
        face_height = y_end - y

        # [TODO] Adjust these values
        left_cheek_x = x + int(face_width * 0.1)
        right_cheek_x = x + int(face_width * 0.6)
        cheek_y = y + int(face_height * 0.4)
        cheek_width = int(face_width * 0.3)
        cheek_height = int(face_height * 0.3)
        
        left_cheek = frame[cheek_y:cheek_y+cheek_height, left_cheek_x:left_cheek_x+cheek_width]
        right_cheek = frame[cheek_y:cheek_y+cheek_height, right_cheek_x:right_cheek_x+cheek_width]
        
        if left_cheek.size > 0 and right_cheek.size > 0:
            combined_cheeks = np.hstack((left_cheek, right_cheek))
            return cv2.resize(combined_cheeks, self.target_size)
        
        return None
