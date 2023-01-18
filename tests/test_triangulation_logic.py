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
        Detections(camera_id=1, probability=0.9, timestamp=0, x=10, y=1000, z=0),  # This is actually outside of the pitch, but within our bounds
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
        Detections(camera_id=3, probability=0.9, timestamp=0, x=800, y=800, z=0),  # This is actually slightly outside of the pitch, but within our bounds
        Detections(camera_id=3, probability=0.9, timestamp=0, x=700, y=700, z=0),
    ]

    _detections = tracker.perform_homography(in_bounds_detections)

    print(_detections)

    # Check that the homography was performed correctly
    assert len(_detections) == 5
    assert len(_detections[0].x) == 1
    assert type(_detections[0].x) == np.ndarray  # Not sure if this is what we want, but it's what we have for now

    # Check that the outputs are reasonable
    assert 10 < _detections[0].x[0] < 11
    assert 46 < _detections[0].y[0] < 50
    assert _detections[2].x < 0  # This value is in bounds, but outside of the pitch! It should be negative (in this scenario)
    assert _detections[3].y < 0, "This value is just off the pitch, passed the side lines, so should be negative"

