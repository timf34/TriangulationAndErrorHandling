import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from typing import Tuple, List

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
        assert 0 <= x <= self.pitch_width and 0 <= y <= self.pitch_height, f"x and y must be between {self.pitch_width} and {self.pitch_height}"
        return cv2.circle(self.pitch_image, (x, y), 15, (255, 0, 0), -1)

    def run(self):
        tracker = MultiCameraTracker()
        tracker.add_camera(1, JETSON1_REAL_WORLD)
        tracker.add_camera(3, JETSON3_REAL_WORLD)

        # Create a video write object to save the matplotlib figure to
        video_writer = cv2.VideoWriter('triangulation_visualization.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))

        # We will now open the video write object and write the frames to it


        for i, (image_1, image_2, box_1, box_2, label_1, label_2, image_path_1, image_path_2) in enumerate(self.dataset):


            # Get x and y coordinates of the box
            x_1, y_1 = get_xy_from_box(box_2)
            x_3, y_3 = get_xy_from_box(box_1)

            # TODO: mirroring for Jetson3 to bring the origins a bit closer together in the diff plances (in my mind at least, haven't tested to see if it works better yet)
            x_3 = 1920 - x_3

            d1 = Detections(camera_id=1, probability=0.9, timestamp=i, x=x_1, y=y_1, z=0)
            d2 = Detections(camera_id=3, probability=0.9, timestamp=i, x=x_3, y=y_3, z=0)
            det = tracker.multi_camera_analysis(d1, d2)


            if det is not None:
                det.x = det.x * (1920 / 102)
                det.y = det.y * (1218 / 64)
                pitch_image = self.draw_point(int(det.x), int(det.y))
            else:
                pitch_image = self.draw_point(0, 0)

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
                i1.set_data(image_1)
                i2.set_data(image_2)
                i3.set_data(pitch_image)
                plt.show(block=False)
                plt.pause(0.000001)
                # Save the figure to the video writer
                # video_writer.write(cv2.cvtColor(np.array(fig.canvas.renderer._renderer), cv2.COLOR_RGB2BGR))



            # self.plot_images(image_1, image_2, pitch_image)

            # print(box_1)

            # Write our plt figure to the video writer
            # video_writer.write(cv2.cvtColor(pitch_image, cv2.COLOR_RGB2BGR))

            # Clear the figure so we can plot the next frame
            # plt.clf()
            # plt.close()

            # Wait for user to press a key before showing next plot
            # plt.waitforbuttonpress(0)
            # plt.close()

            # if i == 10:
            #     break

        # Close the video writer
        video_writer.release()

        # Close the figure
        plt.close()




def main():
    triangulation = TriangulationVisualization()





    triangulation.run()


if __name__ == '__main__':
    main()