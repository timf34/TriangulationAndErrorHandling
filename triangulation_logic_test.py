import copy
import numpy as np
import unittest

from triangulation_logic import MultiCameraTracker, Detections, create_tracker_instance, ThreeDPoints


class TestMultiCameraTracker(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test. You can use
        # it to set up any state that is common to all your tests.
        self.tracker = create_tracker_instance()

    def test_create_tracker_instance(self):
        """
        Test that the create_tracker_instance function returns a MultiCameraTracker object.
        """
        tracker = create_tracker_instance()
        assert tracker is not None

    def test_camera_count(self):
        """
        Test that the camera_count property is set correctly. It is set in create_tracker_instance.
        """
        self.assertEqual(self.tracker.camera_count, 2)

    def test_remove_oob_detections(self):
        """
        Test that the remove_oob_detections function removes detections that are out of bounds.
        Image field coords are currently set in utils.config in the get_image_field_coordinates() function.
        """
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
        """
        Test that the calculate_midpoint function returns the correct midpoint.
        """
        midpoint = self.tracker.calculate_midpoint(0., 0., 2., 2.)
        self.assertEqual(midpoint, (1, 1))
        self.assertEqual(type(midpoint), tuple)
        self.assertEqual(type(midpoint[0]), np.float64)  # Note: I might change this type later, but its this for now.
        self.assertEqual(type(midpoint[1]), np.float64)

    def test_transition_smoothing(self):
        """
        Test that the transition_smoothing function smooths the transition between two points.

        1. Test how it handles receiving a single point with no previous detections. It should just return the same
        ThreeDPoint object.
        2. Then we append the first point to the tracker's three_d_points list (this is done in the real code in the funcs
        that call transition_smoothing).
        3. When we pass in a second point, and it should return a ThreeDPoint object with smoothed (i.e. halved) x and y
        """
        three_d_det = ThreeDPoints(x=50., y=25., z=0, timestamp=0)
        smoothed_det1 = self.tracker.transition_smoothing(three_d_det)

        self.tracker.three_d_points.append(copy.deepcopy(three_d_det))

        three_d_det = ThreeDPoints(x=0., y=25., z=0, timestamp=0)
        smoothed_det2 = self.tracker.transition_smoothing(three_d_det)

        self.assertEqual(type(smoothed_det1), ThreeDPoints)
        self.assertEqual(smoothed_det1.x, 50.)
        self.assertEqual(smoothed_det1.y, 25.)
        self.assertEqual(smoothed_det2.x, 25.)
        self.assertEqual(smoothed_det2.y, 25.)



    def test_two_camera_detection(self):
        raise NotImplementedError

    def test_one_camera_detection(self):
        raise NotImplementedError

    def test_multi_camera_analysis(self):
        raise NotImplementedError

    def tets_perform_homogrpahy(self):
        raise NotImplementedError

    def test_form_plan(self):
        raise NotImplementedError

    def test_common_sense(self):
        raise NotImplementedError

    def test_ball_speed(self):
        raise NotImplementedError

    def test_triangulate(self):
        raise NotImplementedError

    def test_internal_height_estimation(self):
        processed_val = [Detections(camera_id=3, probability=0.9, timestamp=9, x=8.874453020053366, y=22.255735859466892, z=0, x_hom=0.0, y_hom=0.0)]
        # Note: This will only be called if our plane is not None! So we can set it manually for now.
        self.tracker.plane = np.array([[-22], [-5], [0], [414]])  # Value retrieved by stepping through the real code with debugger.
        height = self.tracker.internal_height_estimation(processed_val)
        # Assert that height is close to :[array([1.1996040147356337], dtype=object), array([77.52174233516322], dtype=object), array([77.52174233516322], dtype=object)]
        self.assertAlmostEqual(int(height[2]), 77)
        # Note: this test could be cleaned up!


if __name__ == "__main__":
    unittest.main()
