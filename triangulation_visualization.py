import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

from typing import Tuple, List, Generator

from data.bohs_dataset import create_triangulation_dataset
from data_classes import Detections
from triangulation_logic import MultiCameraTracker

JETSON1_REAL_WORLD = np.array([[-19.41], [-21.85], [7.78]])
JETSON3_REAL_WORLD = np.array([[0.], [86.16], [7.85]])


def get_xy_from_box(box: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    box = box[0]
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return x, y


class TriangulationVisualization:
    def __init__(self):
        self.dataset = create_triangulation_dataset()
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

        # ax1.imshow(image_1)
        # ax2.imshow(image_2)
        # ax3.imshow(image_3)
        # plt.show(block=False)

    def draw_point(self, x, y):
        """
        This function allows us to draw a point on self.pitch_image
        :param x: x coordinate of point
        :param y: y coordinate of point
        :return: image with point drawn on it
        """
        # Clear the image
        self.pitch_image = np.array(Image.open("images/pitch.jpg"))
        assert 0 <= x <= self.pitch_width and 0 <= y <= self.pitch_height, f"x and y must be between {self.pitch_width} and {self.pitch_height}"
        return cv2.circle(self.pitch_image, (x, y), 15, (255, 0, 0), -1)

    @staticmethod
    def x_y_to_detection(x_1, y_1, x_3, y_3, i) -> Tuple[Detections, Detections]:
        d1 = Detections(camera_id=1, probability=0.9, timestamp=i, x=x_1, y=y_1, z=0)
        d2 = Detections(camera_id=3, probability=0.9, timestamp=i, x=x_3, y=y_3, z=0)
        return d1, d2

    @staticmethod
    def create_plot():
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5), tight_layout=True,
                                            gridspec_kw={'hspace': .1})
        # fig.canvas.draw()
        return fig, ax1, ax2, ax3

    def get_triangulated_images(self) -> Generator:
        tracker = MultiCameraTracker()
        tracker.add_camera(1, JETSON1_REAL_WORLD)
        tracker.add_camera(3, JETSON3_REAL_WORLD)

        for i, (image_1, image_2, box_1, box_2, label_1, label_2, image_path_1, image_path_2) in enumerate(
                self.dataset):

            # Get x and y coordinates of the box
            x_1, y_1 = get_xy_from_box(box_2)
            x_3, y_3 = get_xy_from_box(box_1)

            # TODO: mirroring for Jetson3 to bring the origins a bit closer together in the diff plances (in my mind at least, haven't tested to see if it works better yet)
            x_3 = 1920 - x_3

            d1, d2 = self.x_y_to_detection(x_1, y_1, x_3, y_3, i)
            det = tracker.multi_camera_analysis(d1, d2)

            if det is not None:
                det.x = det.x * (1920 / 102)
                det.y = det.y * (1218 / 64)
                pitch_image = self.draw_point(int(det.x), int(det.y))
            else:
                pitch_image = self.draw_point(0, 0)

            yield image_1, image_2, pitch_image

    def run(self):
        fig, ax1, ax2, ax3 = self.create_plot()

        def update_plot(i) -> None:
            # Unpack i
            image_1, image_2, pitch_image = i
            fig.canvas.draw()
            ax1.imshow(image_1)
            ax2.imshow(image_2)
            ax3.imshow(pitch_image)

        # Create a cv2 VideoWriter object
        out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'XVID'), 60, (1280, 2160))

        # Loop through self.get_triangulated_images(); update the plot; write the frame to the video
        for count, i in enumerate(self.get_triangulated_images()):
            img1, img2, pitch_image = i

            # Convert all images to RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            pitch_image = cv2.cvtColor(pitch_image, cv2.COLOR_BGR2RGB)

            # Resize images
            img1 = cv2.resize(img1, (1280, 720))
            img2 = cv2.resize(img2, (1280, 720))
            pitch_image = cv2.resize(pitch_image, (1280, 720))

            # Get the image size
            # height, width, layers = img1.shape
            # print(height, width, layers)

            # Stack the images together
            stacked_image = np.vstack((img1, img2))
            stacked_image = np.vstack((stacked_image, pitch_image))
            # Show images
            # cv2.imshow("Stacked image", stacked_image)
            # cv2.waitKey(0)

            # Write the frame to the video
            out.write(stacked_image)

            # if count == 3:
            #     break

        # Release the VideoWriter object
        out.release()





        # ani = FuncAnimation(fig, update_plot, frames=_frames)
        # my_writer = PillowWriter(fps=20, codec='libx264', bitrate=2)
        # ani.save("lets_go.gif", writer='pillow', fps=20, codec='libx264', bitrate=2)
        # plt.show()

    def og_run(self):
        tracker = MultiCameraTracker()
        tracker.add_camera(1, JETSON1_REAL_WORLD)
        tracker.add_camera(3, JETSON3_REAL_WORLD)

        for i, (image_1, image_2, box_1, box_2, label_1, label_2, image_path_1, image_path_2) in enumerate(
                self.dataset):

            # Get x and y coordinates of the box
            x_1, y_1 = get_xy_from_box(box_2)
            x_3, y_3 = get_xy_from_box(box_1)

            # TODO: mirroring for Jetson3 to bring the origins a bit closer together in the diff plances (in my mind at least, haven't tested to see if it works better yet)
            x_3 = 1920 - x_3

            d1, d2 = self.x_y_to_detection(x_1, y_1, x_3, y_3, i)
            det = tracker.multi_camera_analysis(d1, d2)

            if det is not None:
                det.x = det.x * (1920 / 102)
                det.y = det.y * (1218 / 64)
                pitch_image = self.draw_point(int(det.x), int(det.y))
            else:
                pitch_image = self.draw_point(0, 0)

            # Creating the plot
            if i == 0:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5), tight_layout=True,
                                                    gridspec_kw={'hspace': .1})
                fig.canvas.draw()
                # Add images to the figure without showing them
                i1 = ax1.imshow(image_1)
                i2 = ax2.imshow(image_2)
                i3 = ax3.imshow(pitch_image)
                plt.show(block=False)
            else:
                # Updating the plot...
                i1.set_data(image_1)
                i2.set_data(image_2)
                i3.set_data(pitch_image)
                plt.show(block=False)
                plt.pause(0.000001)

        # Close the figure
        plt.close()


def main():
    triangulation = TriangulationVisualization()
    triangulation.run()


if __name__ == '__main__':
    main()
