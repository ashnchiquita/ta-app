import numpy as np
from components.face_tracker.base import FaceTracker

class Centroid(FaceTracker):
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}  # Stores: {object_id: {'rect': (x, y, x_end, y_end), 'centroid': [c_x, c_y]}}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, rect):
        x, y, x_end, y_end = rect
        centroid = np.array([int((x + x_end) / 2), int((y + y_end) / 2)], dtype="int")
        self.objects[self.next_object_id] = {
            'rect': rect,
            'centroid': centroid
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
            
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
            
    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        if len(self.objects) == 0:
            for i in range(0, len(rects)):
                self.register(rects[i])
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            c_x = int((start_x + end_x) / 2)
            c_y = int((start_y + end_y) / 2)
            input_centroids[i] = (c_x, c_y)
        
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[obj_id]['centroid'] for obj_id in object_ids]
            
        D = np.zeros((len(object_centroids), len(input_centroids)))
        for i in range(len(object_centroids)):
            for j in range(len(input_centroids)):
                D[i, j] = np.linalg.norm(object_centroids[i] - input_centroids[j])
                        
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_rows = set()
        used_cols = set()
        
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
                    
            object_id = object_ids[row]
            
            self.objects[object_id] = {
                'rect': rects[col],
                'centroid': input_centroids[col]
            }
            self.disappeared[object_id] = 0
            
            used_rows.add(row)
            used_cols.add(col)
                
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        
        if D.shape[0] >= D.shape[1]:
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        else:
            for col in unused_cols:
                self.register(rects[col])
                            
        return self.objects
