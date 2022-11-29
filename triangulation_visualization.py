import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List

from data.bohs_dataset import TriangulationBohsDataset, create_triangulation_dataset


class TriangulationVisualization:
    def __init__(self):
        self.dataset = create_triangulation_dataset()
        self.pitch_image = np.array(Image.open("images/pitch.jpg"))
        self.pitch_width, self.pitch_height = self.pitch_image.shape[1], self.pitch_image.shape[0]

    @staticmethod
    def plot_images(image_1, image_2, image_3):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5), tight_layout=True, gridspec_kw={'hspace': .1})
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
        assert 0 <= x <= self.pitch_width and 0 <= y <= self.pitch_height, f"x and y must be between {self.pitch_width} and {self.pitch_height}"
        return cv2.circle(self.pitch_image, (x, y), 15, (255, 0, 0), -1)

    def run(self):
        for i, (image_1, image_2, box_1, box_2, label_1, label_2, image_path_1, image_path_2) in enumerate(self.dataset):

            pitch_image = self.draw_point(500, 500)
            # self.plot_images(image_1, image_2, pitch_image)
            print(box_1)
            # Wait for user to press a key before showing next plot
            plt.waitforbuttonpress(0)
            plt.close()

            if i == 10:
                break


def main():
    triangulation = TriangulationVisualization()
    triangulation.run()


if __name__ == '__main__':
    main()