import json
from collections import namedtuple
from camera_constants import *
from data_classes import *
import numpy as np

GT_PATH = r'C:\Users\timf3\PycharmProjects\task2_3\gt_val_detections.json'
PITCH_JPG = r'C:\Users\timf3\Pictures\fieldmodel.jpg'
RED = (255, 0, 0)
RADIUS = 1
THICKNESS = 1
CAMERA_5_CALIBRATION = np.array([[9.31], [-33.97], [59.5]])
CAMERA_6_CALIBRATION = np.array([[27.84], [102.09], [58.83]])
MAX_SPEED = 40
# This is a constant for the ball_speed method and is an arbitrary constant that will need to be changed for once we
# are set up in real life. But for now, 100 will do as we are dealing with a frame rate of 25 FPS
MAX_DELTA_T = 75


class MultiCameraTracker:
    def __init__(self, gt_path):
        # This is a dict for all the camera objects, where the key is the camera id... and yes, the id will be stored
        # twice. Once as a kay, and a again as the first value in the tuple.
        self.cameras = {}
        # This counts the number of camera objects and is updated every time a camera is added
        self.camera_count = None
        # This is a 2nd/ newest homography attribute, which works with the fact that we are going to be adding our
        # cameras to this class using the .add_camera method now instead. And also that we will be storing the
        # information for the homography matrices in the camera constants file.
        self.homographies = {}
        # Just a list of all the 3d points detected... primarily for when theres one detection &/ more error handling
        self.three_d_points = []
        # This is for the height estimation when theres just one detection
        self.plane = None
        # This is just a tuple with the dimensions of the field model in metres (ie (68, 105))
        FieldDimensions = namedtuple('FieldDimensions', 'width length')
        self.field_model = FieldDimensions(68, 105)
        # Path to the json file for working with while developing
        self.gt_path = gt_path
        with open(self.gt_path) as file:
            ball_data = json.load(file)
        # Ball data: dict of lists with key value pairs {'camera_id': [(frame, x, y),...]}
        assert isinstance(ball_data, object)
        self.ball_data = ball_data

    def add_camera(self, idx, real_world_camera_coords):
        self.homographies[str(idx)] = homography_idx(str(idx))
        cam = Camera(id=idx, homography=self.homographies[str(idx)], real_world_camera_coords=real_world_camera_coords)
        self.cameras[str(idx)] = cam
        self.camera_count = len(self.cameras)

    def multi_camera_analysis(self, *detections):
        """
            This method receives detections from all the cameras, and performs all the core multi camera analysis.
            It will receive the detections from arrll of the cameras, triangulate if possible, or give a best estimate
            of where the ball most likely is (note: that I will make other functions to help handle some of these tasks
            but they will all be put together here)
            Args: Detections object iterable
        Returns: 3D world position of the ball
        """
        # print("detections", detections)

        detections = self.perform_homography(detections)

        # Prepare for Triangulation
        # Create a list with the different camera ids
        cam_list = [det.camera_id for det in detections]

        # In case there are no detections
        three_d_pos = None

        # TODO: need to extend this to handle more than two detections
        if len(detections) is 2:

            # deleting the plane
            self.plane = None

            cam1 = self.cameras[str(cam_list[0])].real_world_camera_coords
            cam2 = self.cameras[str(cam_list[1])].real_world_camera_coords
            three_d_pos = triangulate(detections[0], cam1, detections[1], cam2)
            # this assumes that the detections coming through have the same timestamp
            three_d_pos = ThreeDPoints(x=three_d_pos[0], y=three_d_pos[1], z=three_d_pos[2],
                                       timestamp=detections[0].timestamp)

            if (self.field_model.width > three_d_pos.x > 0) and (self.field_model.length > three_d_pos.y > 0):
                if self.common_sense(three_d_pos):
                    self.three_d_points.append(three_d_pos)
                else:
                    three_d_pos = None
            else:
                three_d_pos = None
                print("!!!the detected ball is out of range!!!")

        if len(detections) is 1:
            if self.plane is None:
                self.plane = self.form_plane()

            if self.plane is not None:
                three_d_estimation = self.internal_height_estimation(detections)
                three_d_estimation = ThreeDPoints(x=three_d_estimation[0],
                                                  y=three_d_estimation[1],
                                                  z=three_d_estimation[2],
                                                  timestamp=detections[0].timestamp)

                if (self.field_model.width > three_d_estimation.x > 0) and \
                        (self.field_model.length > three_d_estimation.y > 0):
                    if self.common_sense(three_d_pos):
                        self.three_d_points.append(three_d_estimation)
                        three_d_pos = three_d_estimation
                    else:
                        three_d_pos = None
                else:
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
        # homo_dict = {i: (self.homographies[i] @ detections[i]) for i in detections}
        # homo_dict = {i: (homo_dict[i] / homo_dict[i][2]) for i in homo_dict}

        dets_ = []

        for det in detections:
            temp = self.homographies[str(det.camera_id)] @ np.array([[det.x], [det.y], [1.0]])
            temp = temp / temp[2]
            det.x, det.y = temp[0], temp[1]
            dets_.append(det)

        return dets_

    def form_plane(self):
        """
            This function forms a plane for the purpose of estimating the height of the ball in 3D when there is only 1
            detection of the ball.
        """

        try:
            temp1 = self.three_d_points[-1]
            # this is probably not 'good code' but it works, we need to subtract two vectors
            a = np.array([[temp1.x], [temp1.y], [temp1.z]])
            temp2 = self.three_d_points[-2]
            b = np.array([[temp2.x], [temp2.y], [temp2.z]])
        except IndexError:
            print("theres no last points to form the plane")
            return

        ab = b - a

        # Normal vector of ab which is lying on the ground plane (a, b, 0)
        # However one thing to note is that I'm not sure if my homography is in the 'ground plane' or 1 metre above it
        normal = np.array([-ab[1], ab[0], 0])

        # Equation of the plane, ax + by + cz + constant = 0... and where the plane is vertical so z is always zero
        plane = np.array([normal[0], normal[1], 0, -(a[0] * normal[0] + a[1] * normal[1])])

        self.plane = plane

        return plane

    def internal_height_estimation(self, detections):
        # There will be just one detection if this function is called
        # This function estimates the height of the ball in the scenario where there is just one detection
        for i in detections:
            _id = i.camera_id
            c = self.cameras[str(_id)].real_world_camera_coords
            ball_coords = np.array([[i.x], [i.y], [i.z]])

            # projection of the camera onto xy plane, point d
            d = c
            d[-1] = 0

            # Vector from projected camera to the ball, vector DA
            da = ball_coords - c

            # Parametric equation of line da
            # xyz = aT + camera_constants
            # I can probably make a numpy array out of this, or just make a cleaner comprehension... using zip() perhaps
            # line_da = [[da[0], c[0]], [da[1], c[1]], [da[2], c[2]]]

            # TODO: check the shape of the plane
            t = (-self.plane[3] - c[0] * self.plane[0] - c[1] * self.plane[1]) / \
                (da[0] * self.plane[0] + da[1] * self.plane[1])

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

        if self.ball_speed(possible_detection) > MAX_SPEED:
            return False

        return True

    def ball_speed(self, possible_detection):
        # Also note that this method can only be called when the ball has relatively successive detections so the ball
        # doesn't curve around a good bit (the ball is in a relatively straight line)

        if len(self.three_d_points) is not 0:
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
        if delta_t != 0:
            speed = distance/delta_t
            return speed
        else:
            # Just a large number/ speed instead of returning false
            return 99999


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
    https://en.wikipedia.org/wiki/Triangulation_(computer_vision)#Mid-point_method """

    ball_p = np.array([[ball_p.x], [ball_p.y], [ball_p.z]])
    ball_q = np.array([[ball_q.x], [ball_q.y], [ball_q.z]])

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

    midpoint = (shortest_point1 + shortest_point2) / 2  # where midpoint is a 3x1 vector

    return midpoint


if __name__ == '__main__':
    # TODO: at this stage, our detections are wrapped in three numpy arrays! Look into getting rid of them!

    yolo = MultiCameraTracker(GT_PATH)
    # adding cameras to the tracker object
    yolo.add_camera(5, CAMERA_5_CALIBRATION)
    yolo.add_camera(6, CAMERA_6_CALIBRATION)

    d1 = Detections(camera_id=5, probability=0.9, timestamp=12, x=1000, y=500, z=0)
    d2 = Detections(camera_id=6, probability=0.9, timestamp=12, x=900, y=550, z=0)

    d3 = Detections(camera_id=5, probability=0.9, timestamp=13, x=100, y=550, z=0)
    d4 = Detections(camera_id=6, probability=0.9, timestamp=13, x=950, y=540, z=0)

    x = yolo.multi_camera_analysis(d1, d2)
    y = yolo.multi_camera_analysis(d3, d4)

    print("resulting coordinates 1: ", x)
    print("\n")
    print("resulting coordinates 2: ", y)