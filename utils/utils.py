import cv2
from typing import List, Tuple
from utils.data_classes import Detections


def draw_bboxes_red(image, x, y):
    return cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), 2)


def get_xy_from_box(box: List[Tuple[int, int, int, int]]) -> Tuple[float, float]:
    box = box[0]
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return x, y


def x_y_to_detection(x_1, y_1, i, camera_id: int) -> Detections:
    return Detections(camera_id=camera_id, probability=0.9, timestamp=i, x=x_1, y=y_1, z=0)
