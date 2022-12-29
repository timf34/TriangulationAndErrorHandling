from dataclasses import dataclass
from typing import Tuple, List


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

