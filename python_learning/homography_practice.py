# Import the necessary libraries
import numpy as np

from utils.camera_homography import get_coords_as_array, get_all_coords_as_arrays

np.set_printoptions(suppress=True)


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
        # real_world_coords = np.dot(H, self.image_coords_xyz.T)

        # Divide by the last element to get the real world coordinates
        # real_world_coords = real_world_coords / real_world_coords[2]

        # Convert a single sample from image coordinates to real world coordinates
        samp_image_coords = self.image_coords_xyz[0]
        samp_real_world_coords = np.dot(H, samp_image_coords.T)

        # Divide by the last element to get the real world coordinates
        samp_real_world_coords = samp_real_world_coords / samp_real_world_coords[2]

        print("Textbook method...")
        print(f"Homography: \n {H}")
        print(f"Real world coords: \n {samp_real_world_coords} from the image coordinates: {samp_image_coords}")
        print(f"The ground truth real world coordinates are: {self.real_world_coords[0]}")

        print("Trying another... \n")
        # Convert a single sample from image coordinates to real world coordinates
        samp_image_coords = self.image_coords_xyz[1]
        samp_real_world_coords = np.dot(H, samp_image_coords.T)

        # Divide by the last element to get the real world coordinates
        samp_real_world_coords = samp_real_world_coords / samp_real_world_coords[2]
        print(f"Real world coords: \n {samp_real_world_coords} from the image coordinates: {samp_image_coords}")
        print(f"The ground truth real world coordinates are: {self.real_world_coords[1]}")

    def gpt_homography(self):

        image_coords = self.image_coords
        real_world_coords = self.real_world_coords_xyz

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
            A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
            A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        homography_matrix = V[-1].reshape((3, 4))[:, :3]

        # Use the intrinsic and extrinsic parameters to map points in the image to corresponding points in the real world

        print("\n---------------------------------\n")
        print("ChatGPT method...")
        print(f"Homography: \n {homography_matrix}")
        for count, image_point in enumerate(image_coords[:2]):
            # Convert the image coordinates to homogeneous coordinates
            x, y = image_point
            point = np.array([x, y, 1])

            # Multiply the point by the homography matrix to map it to the real world
            real_world_point = homography_matrix.dot(point)

            # # Convert the real world coordinates back to Cartesian coordinates
            real_world_point = real_world_point / real_world_point[2]

            print(f"Real world coords: \n {real_world_point} from the image coordinates: {image_point}")
            print(f"The ground truth real world coordinates are: {self.real_world_coords[count]}")

    @staticmethod
    def using_four_points():

        # Using points from Box1-4
        four_image_coords = np.array([[295., 784.], [471., 772.], [262., 698.], [57., 673.]])
        four_real_coords_metres = np.array([[0., 23.], [5., 23.], [5., 41.], [0., 41.]])

        # Multiplying the real world coordinates by 1920/102 and 1080/64 to get the image coordinates
        four_real_coords_pixels = np.array([[0., 388.125], [94.11, 388.125], [94.11, 691.875], [0., 691.875]])

        # Convert the image coordinates to homogeneous coordinates
        four_image_coords = np.hstack((four_image_coords, np.ones((four_image_coords.shape[0], 1))))

        # Convert the real world coordinates to homogeneous coordinates (i.e. add a 1 to the end of each point)
        four_real_coords = np.hstack((four_real_coords_metres, np.ones((four_real_coords_metres.shape[0], 1))))

        # Compute the homography matrix using the four corresponding points and the DLT algorithm
        A = []
        for fp, tp in zip(four_image_coords, four_real_coords):
            A.append([0, 0, 0, -fp[0], -fp[1], -1, tp[1] * fp[0], tp[1] * fp[1], tp[1]])
            A.append([fp[0], fp[1], 1, 0, 0, 0, -tp[0] * fp[0], -tp[0] * fp[1], -tp[0]])
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        H = V[-1].reshape((3, 3))

        # Normalize the homography matrix
        H = H / H[2, 2]

        print("\n---------------------------------\n")
        # Convert the image coordinates to real world coordinates
        real_world_coords = np.dot(H, four_image_coords.T)

        # Divide by the last element to get the real world coordinates
        real_world_coords = real_world_coords / real_world_coords[2]

        print("\n\nUsing four points...")
        print(f"Homography: \n {H}")
        # Round the real world coordinates to 2 decimal places
        real_world_coords = np.round(real_world_coords, 2)
        print(f"Real world coords: \n {real_world_coords.T}")
        print(f"The ground truth real world coordinates are: \n{four_real_coords}")

    def extending_using_four_points(self):
        """
        This function will extend `def using_four_points()` to use arbitrary amount of corresponding points to
        calculate the homography matrix and convert image coordinates to real world coordinates
        """
        image_coords = self.image_coords
        # Subtract the first element of each element in image_coords from 1920 to get the mirror image x coord
        # TODO: Note this is only for Jetson3!
        # image_coords = np.array([[1920 - x, y] for x, y in image_coords])

        real_world_coords = self.real_world_coords

        # Convert the image coordinates to homogeneous coordinates
        image_coords = np.hstack((image_coords, np.ones((image_coords.shape[0], 1))))

        # Convert the real world coordinates to homogeneous coordinates (i.e. add a 1 to the end of each point)
        real_world_coords = np.hstack((real_world_coords, np.ones((real_world_coords.shape[0], 1))))

        # Compute the homography matrix using the n corresponding points and the DLT algorithm
        A = []
        for fp, tp in zip(image_coords[:], real_world_coords[:]):
            A.append([0, 0, 0, -fp[0], -fp[1], -1, tp[1] * fp[0], tp[1] * fp[1], tp[1]])
            A.append([fp[0], fp[1], 1, 0, 0, 0, -tp[0] * fp[0], -tp[0] * fp[1], -tp[0]])
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        H = V[-1].reshape((3, 3))

        # Normalize the homography matrix
        H = H / H[2, 2]

        print("\n---------------------------------\n")
        # Convert the image coordinates to real world coordinates
        real_world_coords = np.dot(H, image_coords.T)

        # Divide by the last element to get the real world coordinates
        real_world_coords = real_world_coords / real_world_coords[2]

        print("\n\nExtending using four points...")
        print(f"Homography: \n {H}")
        # Round the real world coordinates to 2 decimal places
        real_world_coords = np.round(real_world_coords, 1)
        print(f"Image coords: \n {image_coords.astype(int)}")
        print(f"Real world coords: \n {real_world_coords.T}")
        print(f"The ground truth real world coordinates are: \n{self.real_world_coords}")

        # # Test one more example for Jetson3
        test_samp = np.array([[1317.], [683.], [1.]])
        # test_samp = np.dot(H, test_samp)
        # test_samp = test_samp / test_samp[2]
        # print(f"Test sample: \n {test_samp}") # Looks all good! 7m 3m 1

        # Test one more example for Jetson1
        # test_samp = np.array([[366.], [355.], [1.]])
        test_samp = np.dot(H, test_samp)
        test_samp = test_samp / test_samp[2]
        print(f"Test sample: \n {test_samp}")


def get_new_homographies():
    image_coords1, image_coords2, real_world_coords1, real_world_coords2 = get_all_coords_as_arrays()

    homographies = {}

    # Subtract the first element of each element in image_coords from 1920 to get the mirror image x coord
    # image_coords1 = np.array([[1920 - x, y] for x, y in image_coords1])  # This was the error!!!! Only for jetson3!
    image_coords2 = np.array([[1920 - x, y] for x, y in image_coords2])

    # Convert the image coordinates to homogeneous coordinates
    image_coords1 = np.hstack((image_coords1, np.ones((image_coords1.shape[0], 1))))

    # Convert the real world coordinates to homogeneous coordinates (i.e. add a 1 to the end of each point)
    real_world_coords1 = np.hstack((real_world_coords1, np.ones((real_world_coords1.shape[0], 1))))

    # Compute the homography matrix using the n corresponding points and the DLT algorithm
    A = []
    for fp, tp in zip(image_coords1[:], real_world_coords1[:]):
        A.append([0, 0, 0, -fp[0], -fp[1], -1, tp[1] * fp[0], tp[1] * fp[1], tp[1]])
        A.append([fp[0], fp[1], 1, 0, 0, 0, -tp[0] * fp[0], -tp[0] * fp[1], -tp[0]])
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))

    # Normalize the homography matrix
    H1 = H / H[2, 2]
    homographies["1"] = H1

    # Convert the image coordinates to homogeneous coordinates
    image_coords2 = np.hstack((image_coords2, np.ones((image_coords2.shape[0], 1))))

    # Convert the real world coordinates to homogeneous coordinates (i.e. add a 1 to the end of each point)
    real_world_coords2 = np.hstack((real_world_coords2, np.ones((real_world_coords2.shape[0], 1))))

    # Compute the homography matrix using the n corresponding points and the DLT algorithm
    A = []
    for fp, tp in zip(image_coords2[:], real_world_coords2[:]):
        A.append([0, 0, 0, -fp[0], -fp[1], -1, tp[1] * fp[0], tp[1] * fp[1], tp[1]])
        A.append([fp[0], fp[1], 1, 0, 0, 0, -tp[0] * fp[0], -tp[0] * fp[1], -tp[0]])
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))

    # Normalize the homography matrix
    H2 = H / H[2, 2]
    homographies["3"] = H2

    return homographies



def main():
    hom = HomographyMethods()
    # hom.h_from_points()
    # hom.gpt_homography()
    # hom.using_four_points()
    hom.extending_using_four_points()

    h = get_new_homographies()


if __name__ == '__main__':
    main()
