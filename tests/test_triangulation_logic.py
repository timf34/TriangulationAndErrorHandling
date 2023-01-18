import pytest
import sys
import numpy as np

from typing import List, Dict

from triangulation_logic import MultiCameraTracker
from utils.data_classes import Detections


JETSON1_REAL_WORLD = np.array([[-19.41], [-21.85], [7.78]])
JETSON3_REAL_WORLD = np.array([[0.], [86.16], [7.85]])


def initialize_tracker() -> MultiCameraTracker:
    """
    Initialize a MultiCameraTracker object with two cameras and some detections
    """
    tracker = MultiCameraTracker(use_formplane=False)
    tracker.add_camera(1, JETSON1_REAL_WORLD)
    tracker.add_camera(3, JETSON3_REAL_WORLD)
    return tracker


def test_initialize_tracker():
    tracker = initialize_tracker()
    assert tracker is not None


def test_remove_oob_detections() -> None:
    tracker = initialize_tracker()


    # Detections that are out of bounds
    oob_detections: List[Detections] = [
        Detections(camera_id=1, probability=0.9, timestamp=0, x=0, y=0, z=0),
        Detections(camera_id=1, probability=0.9, timestamp=0, x=-800, y=-800, z=0),
        Detections(camera_id=1, probability=0.9, timestamp=0, x=2000, y=10000, z=0),
        Detections(camera_id=3, probability=0.9, timestamp=0, x=0, y=0, z=0),
        Detections(camera_id=3, probability=0.9, timestamp=0, x=-800, y=-800, z=0),
        Detections(camera_id=3, probability=0.9, timestamp=0, x=2000, y=10000, z=0),
        Detections(camera_id=3, probability=0.9, timestamp=0, x=0, y=250, z=0),  # Close but not quite in bounds
    ]

    # Detections that are in bounds
    in_bounds_detections: List[Detections] = [
        Detections(camera_id=1, probability=0.9, timestamp=0, x=800, y=800, z=0),
        Detections(camera_id=1, probability=0.9, timestamp=0, x=700, y=700, z=0),
        Detections(camera_id=1, probability=0.9, timestamp=0, x=10, y=1000, z=0),
        Detections(camera_id=3, probability=0.9, timestamp=0, x=800, y=800, z=0),
        Detections(camera_id=3, probability=0.9, timestamp=0, x=700, y=700, z=0),
    ]

    _detections = oob_detections + in_bounds_detections
    _detections = tracker.remove_oob_detections(_detections)
    assert _detections == in_bounds_detections


def test_perform_homography() -> None:
    tracker = initialize_tracker()

    # Empty detections
    _detections: List[Detections] = []

    # Detections that are in bounds
    in_bounds_detections: List[Detections] = [
        Detections(camera_id=1, probability=0.9, timestamp=0, x=800, y=800, z=0),
        Detections(camera_id=1, probability=0.9, timestamp=0, x=700, y=700, z=0),
        Detections(camera_id=1, probability=0.9, timestamp=0, x=10, y=1000, z=0),
        Detections(camera_id=3, probability=0.9, timestamp=0, x=800, y=800, z=0),
        Detections(camera_id=3, probability=0.9, timestamp=0, x=700, y=700, z=0),
    ]

    pass
