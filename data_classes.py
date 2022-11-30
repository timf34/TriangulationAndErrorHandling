from dataclasses import dataclass
from typing import NamedTuple


# TODO: will change this to accept a box, and then process the box to get x, y coordinates... maybe
@dataclass
class Detections:
    camera_id: int
    probability: float
    timestamp: float
    x: int = 0
    y: int = 0
    # z is set to 1 to work with homography for now
    z: int = 1
    x_hom: float = 0.0
    y_hom: float = 0.0


@dataclass
class ThreeDPoints:
    # Object for storing the resulting 3D points in the MultiCameraTracker object for error handling
    x: float
    y: float
    z: float
    timestamp: float


class Camera(NamedTuple):
    id: int
    homography: list
    real_world_camera_coords: tuple