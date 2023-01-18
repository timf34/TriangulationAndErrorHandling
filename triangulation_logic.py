import copy
from collections import namedtuple
from utils.camera_homography import *
from utils.data_classes import Camera, Detections, ThreeDPoints
from typing import Dict, List, Union

from python_learning.homography_practice import get_new_homographies
from utils.config import get_image_field_coordinates


JETSON1_REAL_WORLD = np.array([[-19.41], [-21.85], [7.78]])
JETSON3_REAL_WORLD = np.array([[0.], [86.16], [7.85]])
MAX_SPEED = 40
# This is a constant for the ball_speed method and is an arbitrary constant that will need to be changed for once we
# are set up in real life. But for now, 100 will do as we are dealing with a frame rate of 25 FPS
MAX_DELTA_T = 75

THREE_D_POINTS_FLAG: List[float] = [999., 999., 999.]  # Flag used for when we have no detections


# Generally inspired from: https://github.com/HaziqRazali/Soccer-Tracker


class MultiCameraTracker:
    def __init__(self, use_formplane: bool = True):
        self.cameras: Dict[str, Camera] = {}
        self.camera_count: int = 0
        self.homographies: Dict = get_new_homographies()  # TODO: this needs refactoring when time to cleanup
        self.image_field_coordinates: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = get_image_field_coordinates()
        self.three_d_points: List[ThreeDPoints] = []
        self.plane: List[np.array] = None  # Looks more like Tuple(np.array, np.array, np.array, np.array) - should refactor this!
        FieldDimensions = namedtuple('FieldDimensions', 'width length')
        self.field_model: Tuple = FieldDimensions(68, 105)
        self.use_formplane: bool = use_formplane

    def add_camera(self, idx, real_world_camera_coords):
        cam = Camera(
            id=idx,
            homography=self.homographies[str(idx)],
            real_world_camera_coords=real_world_camera_coords,
            image_field_coordinates=self.image_field_coordinates[str(idx)]
        )
        self.cameras[str(idx)] = cam
        self.camera_count = len(self.cameras)

    # TODO: this is perfect for unit practical_testing.
    def remove_oob_detections(self, _detections: List[Detections]) -> List[Union[Detections, None]]:
        """
        This method removes any detections that are out of bounds of the field in the image frame.
        Args:
            _detections: list of Detections objects
        Returns: list of Detections objects
        """
        for det in _detections:
            # Get the camera ID and then get the image field coordinates for that camera
            image_field_coordinates: Tuple = self.image_field_coordinates[str(det.camera_id)]
            x1,y1, x2, y2, x3, y3, x4, y4 = image_field_coordinates

            if not min(x1, x2, x3, x4) <= det.x <= max(x1, x2, x3, x4) and  \
                min(y1, y2, y3, y4) <= det.y <= max(y1, y2, y3, y4):
                _detections.remove(det)

        return _detections

    def multi_camera_analysis(self, _detections: List[Detections]):
        """
            This method receives detections from all the cameras, and performs all the core multi camera analysis.
            It will receive the detections from arrll of the cameras, triangulate if possible, or give a best estimate
            of where the ball most likely is (note: that I will make other functions to help handle some of these tasks
            but they will all be put together here)
            Args: Detections object iterable
        Returns: 3D world position of the ball
        """
        # TODO: refactor this function. Reduce its length and complexity


        detections = self.perform_homography(copy.deepcopy(_detections))

        # Prepare for Triangulation
        # Create a list with the different camera ids
        cam_list = [det.camera_id for det in detections]

        # In case there are no detections
        three_d_pos = None

        if len(detections) == 2:

            # deleting the plane
            self.plane = None

            cam1 = self.cameras[str(cam_list[0])].real_world_camera_coords
            cam2 = self.cameras[str(cam_list[1])].real_world_camera_coords
            three_d_pos = self.triangulate(detections[0], cam1, detections[1], cam2)
            # this assumes that the detections coming through have the same timestamp
            three_d_pos = ThreeDPoints(x=three_d_pos[0], y=three_d_pos[1], z=three_d_pos[2],
                                       timestamp=detections[0].timestamp)

            if (self.field_model.width > three_d_pos.x > 0) and (self.field_model.length > three_d_pos.y > 0):
                if self.common_sense(three_d_pos):
                    self.three_d_points.append(copy.deepcopy(three_d_pos))
                else:
                    self.three_d_points.append(copy.deepcopy(THREE_D_POINTS_FLAG))
                    three_d_pos = None
            else:
                self.three_d_points.append(copy.deepcopy(THREE_D_POINTS_FLAG))
                three_d_pos = None
                print("!!!the detected ball is out of range!!!")

        if len(detections) == 1:
            if self.plane is None:
                self.plane = self.form_plane()

            if self.plane is not None:  # Check that the ball was recently detected by two cameras


                # Flag for whether to just use homography or use form plane.
                if all(np.all(arr == 0) for arr in self.plane) or not self.use_formplane:  # Check if the plane is all 0's (i.e. if the ball is still) (unncesarily complicated expression as are plane isn't just a single np.array, its 4 in a list atm)
                    three_d_estimation = ThreeDPoints(
                        x=detections[0].x,
                        y=detections[0].y,
                        z=0,
                        timestamp=detections[0].timestamp
                    )
                    print("using homogaphy")
                else:
                    three_d_estimation = self.internal_height_estimation(detections)
                    three_d_estimation = ThreeDPoints(x=three_d_estimation[0],
                                                      y=three_d_estimation[1],
                                                      z=three_d_estimation[2],
                                                      timestamp=detections[0].timestamp)
                    if len(three_d_estimation.x) > 1:
                        print(len(three_d_estimation.x))
                        print("okkkkkk")

                print("yolo ooo ya", self.field_model, three_d_estimation)

                if (self.field_model.width > three_d_estimation.x > 0) and \
                        (self.field_model.length > three_d_estimation.y > 0):
                    if self.common_sense(three_d_pos):
                        self.three_d_points.append(copy.deepcopy(three_d_estimation))
                        three_d_pos = three_d_estimation
                    else:
                        self.three_d_points.append(copy.deepcopy(THREE_D_POINTS_FLAG))
                        three_d_pos = None
                else:
                    self.three_d_points.append(copy.deepcopy(THREE_D_POINTS_FLAG))
                    three_d_pos = None
                    print("!!!the detected ball is out of range!!!")

        return three_d_pos

    def perform_homography(self, detections):

        """
        Just a convenience method however should maybe change the Detections class/ structure so that it automatically
        calculates and stores the homography information
        Args:
            *detections: Detections objects iterable, where detection.z=1.0 (we've hard coded that in here)
        Returns: list of Detections objects with homographied x and y coordinates
        """

        dets_ = []

        for det in detections:
            temp = self.homographies[str(det.camera_id)] @ np.array([[det.x], [det.y], [1.0]], dtype=object)
            temp = temp / temp[2]
            det.x, det.y = temp[0], temp[1]
            dets_.append(det)

        return dets_

    def form_plane(self):
        """
            This function forms a plane for the purpose of estimating the height of the ball in 3D when there is only 1
            detection of the ball.
        """

        # TODO: do note that we delete the plane once we have two detections (I think I had a point but have forgot)

        try:
            # Instead of relying on the ball being out of frame for 1 frame, we'll make it 10.
            last_10_points = self.three_d_points[-10:]
            try:
                last_2_points = [p for p in last_10_points if p != THREE_D_POINTS_FLAG][-2:]
            except IndexError:
                print("Not enough points to create a plane!")
                return None


            temp1 = last_2_points[-1]
            # this is probably not 'good code' but it works, we need to subtract two vectors

            a = np.array([[temp1.x], [temp1.y], [temp1.z]])
            temp2 = last_2_points[-2]
            b = np.array([[temp2.x], [temp2.y], [temp2.z]])
        except IndexError:
            print("theres no last points to form the plane")
            return

        ab = b - a

        # Normal vector of ab which is lying on the ground plane (a, b, 0)
        # However one thing to note is that I'm not sure if my homography is in the 'ground plane' or 1 metre above it
        normal = np.array([-ab[1], ab[0], 0], dtype=object)

        # Equation of the plane, ax + by + cz + constant = 0... and where the plane is vertical so z is always zero
        # constant = -(ax + by + cz) = -ax - by
        plane = np.array([normal[0], normal[1], 0, -(a[0] * normal[0] + a[1] * normal[1])], dtype=object)

        self.plane = plane

        return plane

    def internal_height_estimation(self, detections):
        # There will be just one detection if this function is called
        # This function estimates the height of the ball in the scenario where there is just one detection
        for i in detections:
            _id = i.camera_id
            c = self.cameras[str(_id)].real_world_camera_coords
            ball_coords = np.array([[i.x], [i.y], [i.z]], dtype=object)

            # projection of the camera onto xy plane, point d
            d = c
            d[-1] = 0

            # Vector from projected camera to the ball, vector DA
            # This is coming out wrong!
            da = ball_coords - c

            try:
                t = (-self.plane[3] - c[0] * self.plane[0] - c[1] * self.plane[1]) / \
                    (da[0] * self.plane[0] + da[1] * self.plane[1])
            except IndexError:  # TODO: also need to catch division by zero error
                # Print the error and return the last known position
                print("Error in internal height estimation")
                t = 0
                # TODO: need to check if this is the best way to handle this, or if we even should be getting a
                #  ZeroDivisionError here (when the ball stays in the same position)

            # Point of intersection
            intersection = [(da[0] * t + c[0]), (da[1] * t + c[1]), (da[1] * t + c[1])]

            return intersection

    def inv_triangulate(self, detections):
        # This is to locate the xy coordinates of the ball when there is just one detection
        # This was an experiment... not sure if I'll hold it
        for i in detections:
            c = self.cameras[str(i)].real_world_camera_coords
            ball_coords = detections[i]

            # Vector of line from camera to ball
            vector = ball_coords - c

            # Parametric Equation of the Line
            # p_line = np.array([[c[0], vector[0]], [c[1, vector[1]], [c[2], vector[2]]]])

            # Parametric Equation of the Line
            t = c[2] / (-1 * vector[2])

            # Coordinate at which the line intersects the XY plane
            _x = c[0] + vector[0] * t
            _y = c[1] + vector[1] * t

            return [_x, _y]

    def common_sense(self, possible_detection):
        # This method will return a boolean whether or not the proposed detection is possible.

        # Currently we are only checking for ball speed, but this should be extended

        # if self.ball_speed(possible_detection) > MAX_SPEED:
        #     return False

        return True

    def ball_speed(self, possible_detection):
        # Also note that this method can only be called when the ball has relatively successive detections so the ball
        # doesn't curve around a good bit (the ball is in a relatively straight line)

        if len(self.three_d_points) != 0:
            last_det = self.three_d_points[-1]
        else:
            return 0

        # Euclidean distance between two points
        distance = np.sqrt((float(last_det.x) - float(possible_detection.x)) ** 2 +
                           (float(last_det.y) - float(possible_detection.y)) ** 2)

        delta_t = possible_detection.timestamp - last_det.timestamp

        if delta_t > MAX_DELTA_T:
            # return 0 as if the delta_t is greater than a few seconds, theres no point in finding the ball speed as
            # the ball speed as the ball is more likely to move in a non-straight line (the ball speed wouldn't be
            # accurate).
            return 0
        return distance / delta_t if delta_t != 0 else 99999  # Speed

    @staticmethod
    def triangulate(ball_p, cam_p, ball_q, cam_q):
        """
        ballP and ballQ are inputs of the ball's coordinates. For now we will assume they have already be mapped to the real
        world coordinate system through a homography and that they have attributes
              x, y, z where z=0
              Note that z actually equals 1!!!
        although going forward, I should probably build that into this function, including recognising which camera the
        detections belong to, to apply the correct homography.
        camP and camQ are the cameras real world positions with attribute x, y and z.
        This function uses mid-point triangulation ->
        https://en.wikipedia.org/wiki/Triangulation_(computer_vision)#Mid-point_method
        """
        ball_p = np.array([[ball_p.x], [ball_p.y], [ball_p.z]], dtype=object)
        ball_q = np.array([[ball_q.x], [ball_q.y], [ball_q.z]], dtype=object)

        l1 = ball_p - cam_p  # direction vectors
        l2 = ball_q - cam_q

        r1 = np.vdot(l1, l1)  # this is the squared norm/ magnitude of L1
        r2 = np.vdot(l2, l2)  # same for L2

        l1_l2 = np.vdot(l1, l2)  # dot product of L1 and L2

        balls_l1 = np.vdot((ball_q - ball_p), l1)  # dot product of direction vector between the balls, and L1
        balls_l2 = np.vdot(l2, (ball_q - ball_p))  # same but for L2

        s = (((l1_l2 * balls_l2) + (balls_l1 * r2)) / ((r1 * r2) + (l1_l2 ** 2)))
        t = (((l1_l2 * balls_l1) - (balls_l2 * r1)) / ((r1 * r2) + (l1_l2 ** 2)))

        s = abs(s)
        t = abs(t)

        shortest_point1 = ((1 - s) * ball_p) + s * cam_p
        shortest_point2 = ((1 - t) * ball_q) + t * cam_q

        midpoint = (shortest_point1 + shortest_point2) / 2

        # Convert ndarray: (3, 1) to list of ints
        intersection = [midpoint[0], midpoint[1], midpoint[2]]

        return intersection


def get_test_cases() -> List[Dict[str, Union[List[Detections], str]]]:
    """
    This function will return a list of detections that will be used to test the ball tracker
    :return: List of Detections
    """
    cases = [
        {
            "detections": [Detections(camera_id=3, probability=0.9, timestamp=9, x=1017, y=298, z=0),
                           Detections(camera_id=1, probability=0.9, timestamp=9, x=508, y=764, z=0)
                           ],
            "expected": "40, 10 ish"
        },
        {
            "detections": [Detections(camera_id=3, probability=0.9, timestamp=9, x=891, y=284, z=0),
                           Detections(camera_id=1, probability=0.9, timestamp=9, x=274, y=754, z=0)
                           ],
            "expected": "50, 10 ish"
        },
        {
            "detections": [Detections(camera_id=3, probability=0.9, timestamp=9, x=488, y=452, z=0),
                           Detections(camera_id=1, probability=0.9, timestamp=9, x=1153, y=665, z=0)
                           ],
            "expected": "25, 50 ish"
        },

    ]

    return cases


if __name__ == '__main__':
    yolo = MultiCameraTracker()
    # adding cameras to the tracker object
    yolo.add_camera(1, JETSON1_REAL_WORLD)
    yolo.add_camera(3, JETSON3_REAL_WORLD)

    cases = get_test_cases()

    for case in cases:
        result = yolo.multi_camera_analysis(case["detections"])

        # Temp code for cleaning up result from array(['40.0']) to 40
        # result.x = result.x.tolist()[0]
        # result.y = result.y.tolist()[0]
        # result.z = result.z.tolist()[0]

        print(f"Expected: {case['expected']}, Result: {result}")

    # d1 = Detections(camera_id=1, probability=0.9, timestamp=12, x=1062, y=817, z=0)
    # d2 = Detections(camera_id=3, probability=0.9, timestamp=12, x=1408, y=310, z=0)
    # x = yolo.multi_camera_analysis([d1, d2])
    # print("result from 2 detections: ", x)
    #
    # # to form the plane
    # d1 = Detections(camera_id=1, probability=0.9, timestamp=12, x=1062, y=817, z=0)
    # d2 = Detections(camera_id=3, probability=0.9, timestamp=12, x=1402, y=310, z=0)
    # x = yolo.multi_camera_analysis([d1, d2])
    #
    # x = yolo.multi_camera_analysis([d1])
    # print("result from 1 detection: ", x)
    #
    # x = yolo.multi_camera_analysis([d2])
    # print("result from 1 detection: ", x)
    #
    # x = yolo.multi_camera_analysis([])
    # print("result from 0 detections: ", x)
