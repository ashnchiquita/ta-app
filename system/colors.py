ANNOTATION_COLORS = [
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
    (0, 165, 255),    # Orange
    (128, 0, 128),    # Purple
    (203, 192, 255),  # Pink
    (0, 255, 127),    # Spring Green
    (147, 20, 255),   # Deep Pink
    (255, 144, 30),   # Dodger Blue
    (0, 69, 255),     # Red Orange
    (50, 205, 50),    # Lime Green
    (0, 215, 255)     # Gold
]
LEN_ANNOTATION_COLORS = len(ANNOTATION_COLORS)

def get_annotation_color(index):
    return ANNOTATION_COLORS[index % LEN_ANNOTATION_COLORS]

def get_random_annotation_color():
    import random
    return ANNOTATION_COLORS[random.randint(0, LEN_ANNOTATION_COLORS - 1)]
