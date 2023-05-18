import numpy as np
import unittest

from triangulation_logic import MultiCameraTracker, Detections, create_tracker_instance


class TestMultiCameraTracker(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test. You can use
        # it to set up any state that is common to all your tests.
        self.tracker = create_tracker_instance()

    def test_create_tracker_instance(self):
        tracker = create_tracker_instance()
        assert tracker is not None

    def test_camera_count(self):
        self.assertEqual(self.tracker.camera_count, 2)

    def test_remove_oob_detections(self):
        within_field_dets = [
            Detections(camera_id=1, probability=0.9, timestamp=12, x=10, y=817, z=0),
            Detections(camera_id=3, probability=0.9, timestamp=12, x=100, y=417, z=0)
        ]
        filtered_detections = self.tracker.remove_oob_detections(within_field_dets)
        self.assertEqual(len(filtered_detections), 2)

        out_of_field_dets = [
            Detections(camera_id=1, probability=0.9, timestamp=0, x=10, y=17, z=0),
            Detections(camera_id=3, probability=0.9, timestamp=12, x=10, y=17, z=0),
        ]
        filtered_detections = self.tracker.remove_oob_detections(out_of_field_dets)
        self.assertEqual(len(filtered_detections), 0)

    def test_calculate_midpoint(self):
        midpoint = self.tracker.calculate_midpoint(0., 0., 2., 2.)
        print(type(midpoint), type(midpoint[0]))
        self.assertEqual(midpoint, (1, 1))
        self.assertEqual(type(midpoint), tuple)
        self.assertEqual(type(midpoint[0]), np.float64)  # Note: I might change this later, but its this for now.

    def test_transition_smoothing(self):
        # I need to have made one valid detection prior to this to test it.
        pass

    def test_internal_height_estimation(self):
        processed_val = [Detections(camera_id=3, probability=0.9, timestamp=9, x=8.874453020053366, y=22.255735859466892, z=0, x_hom=0.0, y_hom=0.0)]
        # Note: This will only be called if our plane is not None! So we can set it manually for now.
        self.tracker.plane = np.array([[-22], [-5], [0], [414]])  # Value retrieved by stepping through the real code with debugger.
        height = self.tracker.internal_height_estimation(processed_val)
        # Assert that height is close to :[array([1.1996040147356337], dtype=object), array([77.52174233516322], dtype=object), array([77.52174233516322], dtype=object)]
        self.assertAlmostEqual(int(height[2]), 78)
        # Note: this test could be cleaned up!


if __name__ == "__main__":
    unittest.main()
