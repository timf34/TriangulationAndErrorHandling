import numpy as np
import cv2


def compute_homographies():
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
    homography_dict = compute_homographies()

    if camera_id in homography_dict:
        return homography_dict[camera_id]
    else:
        print(str(camera_id) + ' does not have a key in homography_dict')
        raise KeyError