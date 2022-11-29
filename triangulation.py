import matplotlib.pyplot as plt


from data.bohs_dataset import TriangulationBohsDataset, create_triangulation_dataset
from data.visualizing_data import TriangulationVisualization


class Triangulation:
    def __init__(self):
        self.dataset = create_triangulation_dataset()

    @staticmethod
    def plot_images(image_1, image_2):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image_1)
        ax2.imshow(image_2)
        plt.show(block=False)

    def run(self):
        for i, (image_1, image_2, box_1, box_2, label_1, label_2, image_path_1, image_path_2) in enumerate(self.dataset):
            self.plot_images(image_1, image_2)
            # Wait for user to press a key before showing next plot
            plt.waitforbuttonpress(0)
            plt.close()

            if i == 10:
                break


def main():
    triangulation = Triangulation()
    triangulation.run()


if __name__ == '__main__':
    main()