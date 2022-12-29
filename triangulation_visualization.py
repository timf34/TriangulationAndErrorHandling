import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Generator

from data.bohs_dataset import create_triangulation_dataset
from utils.data_classes import Detections
from triangulation_logic import MultiCameraTracker

JETSON1_REAL_WORLD = np.array([[-19.41], [-21.85], [7.78]])
JETSON3_REAL_WORLD = np.array([[0.], [86.16], [7.85]])


def draw_bboxes_red(image, x, y):
    return cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), 2)


def get_xy_from_box(box: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    box = box[0]
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return x, y


class TriangulationVisualization:
    def __init__(self, small_dataset=False):
        self.dataset = create_triangulation_dataset(small_dataset=small_dataset)
        self.pitch_image = np.array(Image.open("images/pitch.jpg"))
        self.pitch_width, self.pitch_height = self.pitch_image.shape[1], self.pitch_image.shape[0]

    @staticmethod
    def plot_images(image_1, image_2, image_3):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5), tight_layout=True, gridspec_kw={'hspace': .1})
        # Add images to the figure without showing them
        ax1.imshow(image_1)
        ax2.imshow(image_2)
        ax3.imshow(image_3)
        plt.show(block=False)

    def draw_point(self, x, y):
        """
        This function allows us to draw a point on self.pitch_image
        :param x: x coordinate of point
        :param y: y coordinate of point
        :return: image with point drawn on it
        """
        self.pitch_image = np.array(Image.open("images/pitch.jpg"))  # Clear the image
        assert 0 <= x <= self.pitch_width and 0 <= y <= self.pitch_height, f"x and y must be between " \
                                                                           f"{self.pitch_width} and {self.pitch_height}"
        return cv2.circle(self.pitch_image, (x, y), 15, (255, 0, 0), -1)

    @staticmethod
    def x_y_to_detection(x_1, y_1, i, camera_id: int) -> Tuple[Detections, Detections]:
        d1 = Detections(camera_id=camera_id, probability=0.9, timestamp=i, x=x_1, y=y_1, z=0)
        return d1

    @staticmethod
    def create_plot():
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5), tight_layout=True,
                                            gridspec_kw={'hspace': .1})
        return fig, ax1, ax2, ax3

    def get_triangulated_images(self) -> Generator:
        tracker = MultiCameraTracker()
        tracker.add_camera(1, JETSON1_REAL_WORLD)
        tracker.add_camera(3, JETSON3_REAL_WORLD)

        # Where the number is the camera id
        for i, (image_3, image_1, box_3, box_1, label_3, label_1, image_path_3, image_path_1) in enumerate(self.dataset):

            det = None
            dets = []

            if box_3.size != 0:
                x_3, y_3 = get_xy_from_box(box_3)
                image_3 = cv2.putText(image_3, f"x: {x_3}, y: {y_3}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                image_3 = draw_bboxes_red(image_3, x_3, y_3)
                x_3 = 1920 - x_3
                dets.append(self.x_y_to_detection(x_3, y_3, i, camera_id=3))

                # cv2.imshow("Image 1", image_1)

            if box_1.size != 0:
                x_1, y_1 = get_xy_from_box(box_1)
                # note: mirroring for Jetson3 to bring the origins a bit closer together in the diff plances (in my mind at least, haven't tested to see if it works better yet)
                image_1 = draw_bboxes_red(image_1, x_1, y_1)
                image_1 = cv2.putText(image_1, f"x: {x_1}, y: {y_1}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                dets.append(self.x_y_to_detection(x_1, y_1, i, camera_id=1))
                # cv2.imshow("Image 2", image_2)

            if len(dets) == 0:
                # det = tracker.multi_camera_analysis(dets)
                det = None
            else:
                det = tracker.multi_camera_analysis(dets)

            if det is not None:
                det.x = det.x * (1920 / 102)
                det.y = det.y * (1218 / 64)
                pitch_image = self.draw_point(int(det.x), int(det.y))
                pitch_image = cv2.putText(pitch_image, f"x: {det.x}, y: {det.y}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                pitch_image = self.draw_point(0, 0)

            # Print a progress message
            if i % 500 == 0:
                print(f"Processed {i*2} images")

            yield image_3, image_1, pitch_image

    def run(self, video_name: str, show_images: bool = False) -> None:
        fig, ax1, ax2, ax3 = self.create_plot()

        # Create a cv2 VideoWriter object
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 60, (1280, 2160))

        # Loop through self.get_triangulated_images(); update the plot; write the frame to the video
        for i in self.get_triangulated_images():
            img1, img2, pitch_image = i

            # Convert all images to RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            pitch_image = cv2.cvtColor(pitch_image, cv2.COLOR_BGR2RGB)

            # Resize images
            img1 = cv2.resize(img1, (1280, 720))
            img2 = cv2.resize(img2, (1280, 720))
            pitch_image = cv2.resize(pitch_image, (1280, 720))

            # Stack the images together
            stacked_image = np.vstack((img1, img2))
            stacked_image = np.vstack((stacked_image, pitch_image))

            if show_images:
                cv2.imshow("Stacked image", stacked_image)
                cv2.waitKey(0)

            out.write(stacked_image)

        # Release the VideoWriter object
        out.release()


def main():
    triangulation = TriangulationVisualization(small_dataset=False)
    triangulation.run("v3-full-triangulation.avi", show_images=False)


if __name__ == '__main__':
    main()
