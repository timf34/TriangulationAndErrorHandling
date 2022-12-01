import numpy as np
import cv2

from typing import NamedTuple, Tuple, List


class CameraJetson1:
    def __init__(self):
        # Note: this definitely is not he cleanest way to do this, but it works for now
        #  What would be better is to have a config file that stores all of this information as a dictionary maybe
        # Structure is (x_pixel_coords, y_pixel_coords)
        self.corner1: Tuple[int, int] = (807, 1005)
        self.corner2: Tuple[int, int] = (9, 646)
        self.bixBox1: Tuple[int, int] = (475, 886)
        self.bixBox2: Tuple[int, int] = (1062, 817)
        self.bixBox3: Tuple[int, int] = (406, 665)
        self.bigBox4: Tuple[int, int] = None
        self.box1: Tuple[int, int] = (295, 784)
        self.box2: Tuple[int, int] = (471, 772)
        self.box3: Tuple[int, int] = (262, 698)
        self.box4: Tuple[int, int] = (57, 673)
        self.goalPost1: Tuple[int, int] = (229, 748)  # They can be floats!
        self.goalPost2: Tuple[int, int] = (161, 716)
        self.semiCircle1: Tuple[int, int] = (775, 751)
        self.semiCircle2: Tuple[int, int] = (618, 771)
        self.boxParallel1: Tuple[int, int] = (1416, 899)
        self.boxParallel2: Tuple[int, int] = None
        self.circle1: Tuple[int, int] = (1487, 899)
        self.circle2: Tuple[int, int] = (1172, 669)
        self.halfway1: Tuple[int, int] = None

        # Real world camera coordinates
        self.real_world_x: float = 19.41
        self.real_world_y: float = 21.85
        self.real_world_z: float = 7.78


class CameraJetson3:
    def __init__(self):
        # Note: this definitely is not he cleanest way to do this, but it works for now
        #  What would be better is to have a config file that stores all of this information as a dictionary maybe
        self.corner1: Tuple[int, int] = (1801, 298)
        self.corner2: Tuple[int, int] = (1800, 827)
        self.bixBox1: Tuple[int, int] = None
        self.bixBox2: Tuple[int, int] = (1408, 310)
        self.bixBox3: Tuple[int, int] =(983, 534)
        self.bigBox4: Tuple[int, int] = (1809, 595)
        self.box1: Tuple[int, int] = (642, 709)
        self.box2: Tuple[int, int] = (1655, 355)
        self.box3: Tuple[int, int] = (1598, 456)
        self.box4: Tuple[int, int] = (1809, 471)
        self.goalPost1: Tuple[int, int] = (1806, 387)
        self.goalPost2: Tuple[int, int] = (1810, 427)
        self.semiCircle1: Tuple[int, int] = (1329, 347)
        self.semiCircle2: Tuple[int, int] = (1188, 419)
        self.boxParallel1: Tuple[int, int] = None
        self.boxParallel2: Tuple[int, int] = None
        self.circle1: Tuple[int, int] = (531, 323)
        self.circle2: Tuple[int, int] = (202, 398)
        self.halfway1: Tuple[int, int] = None

        # Real world camera coordinates
        self.real_world_x: float = 0.
        self.real_world_y: float = 86.16
        self.real_world_z: float = 7.85


class RealWorldPitchCoords:
    def __init__(self):
        self.corner1: Tuple[int, int] = (0, 0)
        self.corner2: Tuple[int, int]= (0, 64)
        self.bixBox1: Tuple[int, int] =(0, 10)
        self.bixBox2: Tuple[int, int] = (16, 10)
        self.bixBox3: Tuple[int, int] =(16, 54)
        self.bigBox4: Tuple[int, int] = None
        self.box1: Tuple[int, int] = (0, 23)
        self.box2: Tuple[int, int] = (5, 23)
        self.box3: Tuple[int, int] = (5, 41)
        self.box4: Tuple[int, int] = (0, 41)
        self.goalPost1:Tuple[int, float] = (0, 29.5)  # They can be floats!
        self.goalPost2:Tuple[int, float] = (0, 34.5)
        self.semiCircle1:Tuple[int, int] = (16, 24)
        self.semiCircle2: Tuple[int, int] = (16, 40)
        self.boxParallel1:Tuple[int, int] = (16, 0)
        self.boxParallel2: Tuple[int, int] = None
        self.circle1: Tuple[int, int] = (51, 24)
        self.circle2: Tuple[int, int] = (51, 40)
        self.halfway1: Tuple[int, int] = None


def compute_homography():
    jetson1 = CameraJetson1()
    jetson2 = CameraJetson2()
    jetsons = [jetson1, jetson2]
    real_world = RealWorldPitchCoords()

    # Create an empty np array to store the pixel coordinates
    j1_arr = np.array([])
    j2_arr = np.array([])

    world_points = np.array([])
    for key in jetson1.__dict__.keys():
        if key in real_world.__dict__.keys() and real_world.__dict__[key] is not None and jetson1.__dict__[key] is not None:
            print(jetson1.__dict__[key])
            j1_arr = np.append(j1_arr, jetson1.__dict__[key], axis=0)  # TODO: make a python learning script where I add tuples to a numpy array... make sure I can add them as distinct points
            world_points = np.append(world_points, real_world.__dict__[key])

    # TODO: look to np_array_appending for how to do this using .reshape(-1, 2)

    print(j1_arr)
    print(world_points)


    print(real_world.__dict__)



########### Legacy Code ############

def legacy_compute_homographies():
    """
    Note that currently this function is super messy and so I have stored it in another python file - going forward,
    I should probably just make a proper config file which has the actual matrices but this will do for now.
    Returns: Dictionary of all of the homography matrices that we are interested in
    """
    homography = {}
    # camera 5
    image_pts5 = np.array([[1690, 800], [1525, 390], [1450, 170], [565, 175], [280, 795], [895, 705], [935, 200],
                           [205, 200],
                           [290, 100], [125, 305], [395, 300], [270, 525], [1105, 895], [1625, 125], [1070, 130],
                           [625, 400],
                           [943, 97], [877, 1030], [1104, 1033], [1779, 1035], [1430, 95]])

    world_pts5 = np.array([[27.01278, 24.034], [27.8374, 34.5678], [28.2874, 58.0262], [7.73866, 57.9556],
                           [7.58353, 9.54519],
                           [16.3643, 13.7746], [16.3566, 54.0618], [0.0786684, 54.0799], [0.105432, 67.9509],
                           [0.0919561, 43.1069], [5.35106, 43.0971], [5.35488, 24.779], [19.4079, 4.83892],
                           [32.9241, 63.4719], [19.5252, 63.5554], [10.837, 33.9567], [16.356, 67.96], [16.356, 0],
                           [19.4079, 0], [27.018, 0], [28.274, 68]])

    # camera 6
    image_pts6 = np.array([[525, 170], [405, 405], [195, 790], [1620, 785], [1370, 175], [1720, 205], [1000, 205],
                           [1025, 705],
                           [1650, 520], [1535, 300], [1795, 305], [875, 140], [335, 135], [800, 910], [1300, 400]])
    # TODO: I'll work with the skew lines for now, but come back and try fix this up -> add more constraints
    world_pts6 = np.array([[27.01278, 24.034], [27.8374, 34.5678], [28.2874, 58.0262], [7.73866, 57.9556],
                           [7.58353, 9.54519],
                           [-.01735, 13.77], [16.3643, 13.7746], [16.3566, 54.0618], [5.35106, 43.0971],
                           [5.35488, 24.779],
                           [0.007198, 24.751], [19.4079, 4.83892], [32.8564, 4.60142], [19.5252, 63.5554],
                           [10.837, 33.9567]])

    # homographies from pixel coords to real world
    h5, status5 = cv2.findHomography(image_pts5, world_pts5)
    h6, status6 = cv2.findHomography(image_pts6, world_pts6)

    homography['5'] = h5
    homography['6'] = h6

    return homography


def homography_idx(camera_id):
    # This returns the homography matrix, given a camera number (where the number is a string!)
    homography_dict = legacy_compute_homographies()

    if camera_id in homography_dict:
        return homography_dict[camera_id]
    else:
        print(str(camera_id) + ' does not have a key in homography_dict')
        raise KeyError


def main():
    compute_homography()


if __name__ == '__main__':
    main()
