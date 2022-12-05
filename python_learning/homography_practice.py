# Import the necessary libraries
import numpy as np

from camera_homography import get_coords_as_array


image_coords, real_world_coords = get_coords_as_array()

# Add a zero to the end of each real world coordinate
real_world_coords = np.hstack((real_world_coords, np.zeros((real_world_coords.shape[0], 1))))


# Define the arrays of image and real world coordinates
# image_coords = [[x1, y1], [x2, y2], [x3, y3], ...]
# real_world_coords = [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], ...]

# Create a list of point correspondences
point_correspondences = []
for image_point, real_world_point in zip(image_coords, real_world_coords):
    point_correspondences.append((image_point, real_world_point))

# Compute the intrinsic and extrinsic parameters of the camera using DLT
A = []
for image_point, real_world_point in point_correspondences:
    x, y = image_point
    X, Y, Z = real_world_point
    A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
    A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
A = np.array(A)
U, S, V = np.linalg.svd(A)
P = V[-1].reshape(3, 4)

# Use the intrinsic and extrinsic parameters to map points in the image to corresponding points in the real world
homography_matrix = P[:, :3]
for image_point in image_coords:
    # Convert the image coordinates to homogeneous coordinates
    x, y = image_point
    point = np.array([x, y, 1])

    # Multiply the point by the homography matrix to map it to the real world
    real_world_point = homography_matrix.dot(point)

    # # Convert the real world coordinates back to Cartesian coordinates
    # x, y
    print("point: ", point)
    print("real world point: ", real_world_point)

    real_world_point = real_world_point / real_world_point[2]
    print("real world point: ", real_world_point)
