import copy
import numpy as np
import unittest

from triangulation_logic import MultiCameraTracker, Detections, create_tracker_instance, ThreeDPoints
from utils.data_classes import OutOfBounds


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
        self.assertEqual(type(midpoint[0]), float)
        self.assertEqual(type(midpoint[1]), float)

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
        self.assertEqual(type(smoothed_det1.x), float)
        self.assertEqual(type(smoothed_det2.y), float)

    def test_two_camera_detection(self):
        """
        Test that the two_camera_detection function returns the correct ThreeDPoint object.

        Args to the two_camera_detection function is a list of Detections of length 2, and the camera ids of the two
        cameras that detected the ball.
        """
        dets = [
            Detections(camera_id=1, probability=0.9, timestamp=12, x=10, y=817, z=0),
            Detections(camera_id=3, probability=0.9, timestamp=12, x=100, y=417, z=0)
        ]
        cam_list = [1, 3]
        three_d_point = self.tracker.two_camera_detection(dets, cam_list)
        self.assertEqual(type(three_d_point), OutOfBounds)
        self.assertEqual(type(three_d_point.x), float)

    def test_one_camera_detection(self):
        """
        Test that the one_camera_detection function returns the correct ThreeDPoint object.
        """
        det = [
            Detections(camera_id=1, probability=0.9, timestamp=12, x=10, y=817, z=0)
        ]
        three_d_point = self.tracker.one_camera_detection(det)
        self.assertEqual(type(three_d_point), ThreeDPoints)
        self.assertTrue(isinstance(three_d_point.x, (float, int)))  # TODO: note this returns ints in ThreeDPoints atm

    def test_multi_camera_analysis(self):
        """
        Test that the multi_camera_analysis function returns the correct ThreeDPoint object.

        Will need to test for multiple scenarios here...
        """
        # One detection, camera 1
        det = [
            Detections(camera_id=1, probability=0.9, timestamp=12, x=10, y=817, z=0),
        ]
        three_d_point = self.tracker.multi_camera_analysis(det)
        print("1. ", three_d_point)
        self.assertEqual(type(three_d_point), ThreeDPoints)
        self.assertEqual(type(three_d_point.x), float)

        # One detection, camera 3
        det = [
            Detections(camera_id=3, probability=0.9, timestamp=12, x=10, y=817, z=0),
        ]
        three_d_point = self.tracker.multi_camera_analysis(det)
        print("2. ", three_d_point)
        self.assertEqual(type(three_d_point), OutOfBounds)

        # Two detections, camera 1 and 3
        dets = [
            Detections(camera_id=1, probability=0.9, timestamp=12, x=10, y=817, z=0),
            Detections(camera_id=3, probability=0.9, timestamp=12, x=10, y=817, z=0),
        ]
        three_d_point = self.tracker.multi_camera_analysis(dets)
        print("3. ", three_d_point)
        self.assertEqual(type(three_d_point), ThreeDPoints)

        # Again
        dets = [
            Detections(camera_id=1, probability=0.9, timestamp=12, x=100, y=817, z=0),
            Detections(camera_id=3, probability=0.9, timestamp=12, x=69, y=817, z=0),
        ]
        three_d_point = self.tracker.multi_camera_analysis(dets)
        print("4. ", three_d_point)
        self.assertEqual(type(three_d_point), ThreeDPoints)

        dets = [
            Detections(camera_id=3, probability=0.9, timestamp=9, x=891, y=284, z=0),
            Detections(camera_id=1, probability=0.9, timestamp=9, x=274, y=754, z=0)
        ]
        three_d_point = self.tracker.multi_camera_analysis(dets)
        print("5. ", three_d_point)

        dets = [
            Detections(camera_id=3, probability=0.9, timestamp=9, x=488, y=452, z=0),
            Detections(camera_id=1, probability=0.9, timestamp=9, x=1153, y=665, z=0)
        ]
        three_d_point = self.tracker.multi_camera_analysis(dets)
        print("6. ", three_d_point)
        self.assertEqual(type(three_d_point), ThreeDPoints)

        # Another one
        det = [
            Detections(camera_id=3, probability=0.9, timestamp=9, x=488, y=452, z=0)
        ]
        three_d_point = self.tracker.multi_camera_analysis(det)
        print("7. ", three_d_point)
        self.assertEqual(type(three_d_point), ThreeDPoints)
        self.assertEqual(type(three_d_point.x), float)

    def test_perform_homogrpahy(self):
        det1 = [
            Detections(camera_id=3, probability=0.9, timestamp=9, x=488, y=452, z=0)
        ]

        dets = [
            Detections(camera_id=3, probability=0.9, timestamp=9, x=488, y=452, z=0),
            Detections(camera_id=1, probability=0.9, timestamp=9, x=1153, y=665, z=0)
        ]

        result1 = self.tracker.perform_homography(det1)
        result2 = self.tracker.perform_homography(dets)

        self.assertEqual(type(result1), list)
        self.assertEqual(type(result2), list)
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 2)
        self.assertEqual(type(result1[0]), Detections)
        self.assertEqual(type(result2[0]), Detections)
        self.assertEqual(type(result1[0].x), float)
        self.assertEqual(type(result2[0].x), float)

    def test_form_plane(self):
        # TODO: this seems somewhat tricky to test, will come back to it later.
        pass

    def test_common_sense(self):
        # TODO: actual function is not implemented yet
        pass

    def test_ball_speed(self):
        # TODO: actual function is not implemented yet
        pass

    def test_triangulate(self):
        # TODO: come back to this in a little bit.
        pass

    def test_internal_height_estimation(self):
        processed_val = [Detections(camera_id=3, probability=0.9, timestamp=9, x=8.874453020053366, y=22.255735859466892, z=0, x_hom=0.0, y_hom=0.0)]
        # Note: This will only be called if our plane is not None! So we can set it manually for now.
        self.tracker.plane = np.array([[-22], [-5], [0], [414]])  # Value retrieved by stepping through the real code with debugger.
        height = self.tracker.internal_height_estimation(processed_val)
        # Assert that height is close to :[array([1.1996040147356337], dtype=object), array([77.52174233516322], dtype=object), array([77.52174233516322], dtype=object)]
        self.assertAlmostEqual(int(height[2]), 77)
        # Note: this test could be cleaned up!

    def test_filter_most_confident_dets(self):
        dets = [
            Detections(camera_id=3, probability=0.9, timestamp=9, x=488, y=452, z=0),
            Detections(camera_id=3, probability=0.94, timestamp=9, x=420, y=69, z=0),
            Detections(camera_id=1, probability=0.6, timestamp=8, x=69, y=420, z=0),
            Detections(camera_id=1, probability=0.9, timestamp=9, x=1153, y=665, z=0)
        ]
        filtered_dets = self.tracker.filter_most_confident_dets(dets)
        self.assertEqual(len(filtered_dets), 2)
        self.assertEqual(filtered_dets[0].probability, 0.94)
        self.assertEqual(filtered_dets[1].probability, 0.9)

if __name__ == "__main__":
    unittest.main()
