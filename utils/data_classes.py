from dataclasses import dataclass
from typing import NamedTuple, Tuple


# TODO: will change this to accept a box, and then process the box to get x, y coordinates... maybe
@dataclass
class Detections:
    camera_id: int
    probability: float
    timestamp: float
    x: float = 0
    y: float = 0
    # z is set to 1 to work with homography for now
    z: float = 1
    x_hom: float = 0.0
    y_hom: float = 0.0


@dataclass
class ThreeDPoints:
    # Object for storing the resulting 3D points in the MultiCameraTracker object for error handling
    x: float
    y: float
    z: float
    timestamp: float



from dataclasses import dataclass

@dataclass
class DetectionError:
    """
    Base dataclass representing a detection error, containing the 3D coordinates
    of the erroneous detection (x, y, z) and the time of the detection (timestamp).
    """

    x: float
    y: float
    z: float
    timestamp: float

    @classmethod
    def from_three_d_points(cls, threedpoints: ThreeDPoints):
        """
        Creates a DetectionError instance from a ThreeDPoints object, preserving coordinates and timestamp.

        Args:
            threedpoints (ThreeDPoints): The ThreeDPoints object to convert.

        Returns:
            DetectionError: A DetectionError instance with the same attributes as the input ThreeDPoints object.
        """
        return cls(threedpoints.x, threedpoints.y, threedpoints.z, threedpoints.timestamp)


@dataclass
class OutOfBounds(DetectionError):
    """
    Dataclass representing an out-of-bounds detection.
    Inherits from DetectionError and retains the same structure and methods.
    """


@dataclass
class FailedCommonSense(DetectionError):
    """
    Dataclass representing a detection that failed the 'common_sense' test.
    Inherits from DetectionError and retains the same structure and methods.
    """



# TODO: add more precise typing here
class Camera(NamedTuple):
    id: int
    homography: list
    real_world_camera_coords: Tuple
    image_field_coordinates: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]  # Coordinates are in the format (x, y), and start from the top left corner and go clockwise.