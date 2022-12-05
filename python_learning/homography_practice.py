# Import the necessary libraries
import numpy as np

from camera_homography import get_coords_as_array

class HomographyMethods:
    def __init__(self):
        self.image_coords, self.real_world_coords = get_coords_as_array()

        # Add a zero to the end of each real world coordinate
        self.real_world_coords_xyz = np.hstack((self.real_world_coords, np.zeros((self.real_world_coords.shape[0], 1))))

        # Add a zero to the end of each image coordinate
        self.image_coords_xyz = np.hstack((self.image_coords, np.zeros((self.image_coords.shape[0], 1))))

    def h_from_points(self):
        """ Find homography H, such that fp is mapped to tp
            using the linear DLT method. Points are conditioned
            automatically.

            This is the method for getting the homography matrix from the saved PDF:
                `Excellent - homography - Programming Computer Vision with Python Tools and algorithms for analyzing images by Jan Erik Solem (z-lib.org).pdf
        """

        fp = self.image_coords_xyz.T
        tp = self.real_world_coords_xyz.T

        if fp.shape != tp.shape:
            raise RuntimeError('number of points do not match')

        # condition points (important for numerical reasons)
        # -- from points --
        m = np.mean(fp[:2], axis=1)
        maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
        C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
        C1[0][2] = -m[0] / maxstd
        C1[1][2] = -m[1] / maxstd
        fp = np.dot(C1, fp)

        # -- to points --
        m = np.mean(tp[:2], axis=1)
        C2 = C1.copy()  # must use same scaling for both point sets
        C2[0][2] = -m[0] / maxstd
        C2[1][2] = -m[1] / maxstd
        tp = np.dot(C2, tp)

        # create matrix for linear method, 2 rows for each correspondence pair
        nbr_correspondences = fp.shape[1]
        A = np.zeros((2 * nbr_correspondences, 9))
        for i in range(nbr_correspondences):
            A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
                        tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
            A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                            tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

        U, S, V = np.linalg.svd(A)
        H = V[8].reshape((3, 3))

        # decondition
        H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

        # normalize
        H = H / H[2, 2]

        # Convert the image coordinates to real world coordinates
        real_world_coords = np.dot(H, self.image_coords_xyz.T)

        # Divide by the last element to get the real world coordinates
        real_world_coords = real_world_coords / real_world_coords[2]

        print(real_world_coords)


def main():
    hom = HomographyMethods()
    hom.H_from_points()


if __name__ == '__main__':
    main()
