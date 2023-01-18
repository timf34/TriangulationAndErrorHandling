from dataclasses import dataclass
from typing import Tuple, List, Dict

# TODO: note sure what this class is for... Its not used anywhere in the code.
@dataclass
class TriangulationConfig:
    image_folders: Tuple[str, str] = (
        r"C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\bohs-preprocessed\unpacked_jpg\jetson1_date_01_04_2022_time__19_45_01_4",
        r"C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\bohs-preprocessed\unpacked_jpg\jetson3_1_4_2022_time__19_45_01_4"
    )
    ground_truth_annotations: Tuple[str, str] = (
        r"C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\bohs-preprocessed\annotations\jetson1_date_01_04_2022_time__19_45_01_4.xml",
        r"C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\bohs-preprocessed\annotations\jetson3_1_4_2022_time__19_45_01_4.xml"
    )

    only_matching_frames: bool = True  # Only show frames where both cameras have a frame with a ball.
    only_ball_frames: bool = True  # Only show frames where the ball is in the frame.


def get_image_field_coordinates() -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    This function returns a dict containing the boundaries of the field in the image for each camera.
    Where the camera ID's are "1" and "3"
    Coordinates are in the format (x, y), and start from the top left corner and go clockwise.
    Also note that the top left corner is the origin here (i.e. 0,0)
    :return: Dict[str, Tuple[int, int, int, int]]
    """
    return {
        "1": ((0, 580), (1918, 576), (1920, 1080), (0, 1080)),
        "3": ((0, 260), (1920, 230), (1920, 980), (0, 740)),
    }
