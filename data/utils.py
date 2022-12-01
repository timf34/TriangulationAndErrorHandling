import numpy as np
import os

from collections import defaultdict
from typing import Tuple, List


class SequenceAnnotations:
    '''
    Class for storing annotations for the video sequence

    This should probably be replaced by a Python dataclass
    '''

    def __init__(self):
        # ball_pos contains list of ball positions (x,y) on each frame; multiple balls per frame are possible
        self.ball_pos = defaultdict(list)  # Dict[frame_number] = List[(x, y)]


def _load_bohs_groundtruth(xml_file_path: str) -> dict:
    """
    This function reads and laods the xml file into a dictionary which we will
    then pass to _create_bohs_annotations...

    In general, this will read the groundtruth xml file, to extract frame and x-y position

    :params xml_file_path: path to the xml file
    :returns: dictionary of ball positions
    The structure of this dictionary is as follows:
        {'BallPos': [[382, 382, 437, 782]
        ... where the elements of the list are as follows [start_frame, end_frame, x, y]
    """

    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # print("root.tage", root.tag)  # >>> annotations
    # print("atribbute", root.attrib)  # {}

    gt = {"BallPos": []}
    for child in root:
        for subchild in child:
            if "frame" in subchild.attrib:
                frame = int(subchild.attrib['frame'])
                # We convert the string first to a float, and then to an int
                x = int(float(subchild.attrib['points'].split(',')[0]))
                y = int(float(subchild.attrib['points'].split(',')[1]))

                # We put frame in twice simply to match FootAndBall dataloader
                gt["BallPos"].append([frame, frame, x, y])
    return gt


def _create_bohs_annotations(gt: dict, frame_shape: Tuple[int, int] = (1080, 1920)) -> SequenceAnnotations:
    """
    Converts are groundtruth from _load_bohs_annotations to a SequenceAnnotations object

    annoations.ball_pos -> Dict[frame_number] = List[(x, y)]

    :params groundtruth: dictionary of ball positions
    :params frame_shape: shape of the frame
    :returns: SequenceAnnotations

    """
    annotations = SequenceAnnotations()

    # print("max ball pos", max(gt["BallPos"])) # This prints [3594, 3594, 1667, 633], 3594 is max frame number

    for (start_frame, end_frame, x, y) in gt['BallPos']:
        for i in range(start_frame, end_frame + 1):
            annotations.ball_pos[i].append((x, y))

    return annotations


def read_bohs_ground_truth(annotations_path: str, xml_file_name: str) -> SequenceAnnotations:
    """
    Reads the groundtruth xml file and returns a SequenceAnnotations object

    :params annotations_path: path to the 'annotations' folder
    :params xml_file_name: name of the xml file
    """
    xml_file_path = os.path.join(annotations_path, xml_file_name)
    gt = _load_bohs_groundtruth(xml_file_path)
    return _create_bohs_annotations(gt)


def get_xy_from_box(box: np.array) -> Tuple[int, int]:
    """
        This function gets x and y from an ndarray of shape (1, 4)
    :param box:
    :return:
    """
    x1, y1, x2, y2 = box
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    return x, y