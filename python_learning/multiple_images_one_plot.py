"""
    The goal of this file is to include two images in one plot. This is to compare the ground truth.

    For now, we will just generate two images of different colours.
"""

import matplotlib.pyplot as plt
import numpy as np


def create_image(colour: str, size: int):
    if colour == 'red':
        # Create red image with 3 channels
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        return image
    elif colour == 'green':
        # Create green image with three channels
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:, :, 1] = 255
        return image
    else:
        raise ValueError("Colour must be either 'red' or 'green'")


def plot_images(image1, image2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image1)
    ax2.imshow(image2)
    plt.show()


def main():
    image1 = create_image('red', 100)
    image2 = create_image('green', 100)
    plot_images(image1, image2)


if __name__ == "__main__":
    main()
