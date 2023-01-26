import cv2
from copy import deepcopy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Generator, Union

from data.bohs_dataset import create_triangulation_dataset
from utils.data_classes import Detections
from utils.timer import Timer
from triangulation_logic import MultiCameraTracker

JETSON1_REAL_WORLD = np.array([[-19.41], [-21.85], [7.78]])
JETSON3_REAL_WORLD = np.array([[0.], [86.16], [7.85]])


def draw_bboxes_red(image, x, y):
    return cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), 2)


def get_xy_from_box(box: List[Tuple[int, int, int, int]]) -> Tuple[float, float]:
    box = box[0]
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return x, y


def x_y_to_detection(x_1, y_1, i, camera_id: int) -> Detections:
    return Detections(camera_id=camera_id, probability=0.9, timestamp=i, x=x_1, y=y_1, z=0)


class TriangulationVisualization:
    def __init__(self,
                 small_dataset=False,
                 use_formplane: bool = False,
                 draw_text: bool = False,
                 visualize_homography: bool = False,
                 ):
        self.dataset = create_triangulation_dataset(small_dataset=small_dataset)
        self.pitch_image: np.array = np.array(Image.open("images/pitch.jpg"))
        self.pitch_width: int = self.pitch_image.shape[1]
        self.pitch_height: int = self.pitch_image.shape[0]
        self.timer: Timer = Timer()
        self.use_formplane = use_formplane

        self.tracker = MultiCameraTracker(use_formplane=self.use_formplane)
        self.tracker.add_camera(1, JETSON1_REAL_WORLD)
        self.tracker.add_camera(3, JETSON3_REAL_WORLD)
        self.draw_text: bool = draw_text
        self.visualize_homography: bool = visualize_homography

    @staticmethod
    def plot_images(image_1, image_2, image_3):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5), tight_layout=True, gridspec_kw={'hspace': .1})
        # Add images to the figure without showing them
        ax1.imshow(image_1)
        ax2.imshow(image_2)
        ax3.imshow(image_3)
        plt.show(block=False)

    @staticmethod
    def convert_det_to_pixels(det: Detections) -> Tuple[float, float]:
        x = det.x * (1920 / 102)
        y = det.y * (1218 / 64)
        return x, y

    def visualize_individual_cam_homography(self, tracker: MultiCameraTracker, single_cam_det: Detections,
                                            camera_id: int) -> None:
        cam_hom = tracker.perform_homography([deepcopy(single_cam_det)])[0]
        cam_hom.x, cam_hom.y = self.convert_det_to_pixels(cam_hom)
        self.pitch_image = self.draw_point(int(cam_hom.x), int(cam_hom.y), camera_id=camera_id)

    def draw_point(self, x, y, camera_id: int = None) -> np.array:
        """
        This function allows us to draw a point on self.pitch_image. Note we have to call cv2.circle before cv2.putText
        because cv2.putText draws on top of the image.
        :param camera_id:
        :param x: x coordinate of point
        :param y: y coordinate of point
        :return: image with point drawn on it
        """
        if x <= 0 or x >= self.pitch_width or y <= 0 or y >= self.pitch_height:
            # print(f"x and y must be between {self.pitch_width} and {self.pitch_height} - but got {x} and {y}")
            x = 0
            y = 0

        if camera_id == 1:
            color = (0, 255, 0)
            if not self.draw_text:
                return cv2.circle(self.pitch_image, (x, y), 20, color, -3)
            cv2.circle(self.pitch_image, (x, y), 20, color, -3)
            return cv2.putText(self.pitch_image, "1", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        elif camera_id == 3:
            color = (0, 0, 255)
            if not self.draw_text:
                return cv2.circle(self.pitch_image, (x, y), 20, color, -3)
            cv2.circle(self.pitch_image, (x, y), 20, color, -3)
            return cv2.putText(self.pitch_image, "3", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            color = (255, 0, 0)
            return cv2.circle(self.pitch_image, (x, y), 25, color, -3)

    @staticmethod
    def create_plot():
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5), tight_layout=True,
                                            gridspec_kw={'hspace': .1})
        return fig, ax1, ax2, ax3

    def process_camera_data(self,
                            image: np.ndarray,
                            box: Tuple[int, int, int, int],
                            dets: List,
                            jetson_number: int,
                            index: int
                            ) -> np.ndarray:
        """

        :return:
        """
        if box.size != 0:
            x, y = get_xy_from_box(box)
            if self.draw_text:
                image = cv2.putText(image, f"x: {x}, y: {y}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2, cv2.LINE_AA)
            image = draw_bboxes_red(image, x, y)
            if jetson_number == 3:
                x = 1920 - x  # note: mirroring for Jetson3 to bring the origins a bit closer together in the diff plances (in my mind at least, haven't tested to see if it works better yet)
            cam_det = x_y_to_detection(x, y, index, camera_id=jetson_number)

            dets.append(cam_det)

            if self.visualize_homography:
                self.visualize_individual_cam_homography(self.tracker, cam_det, camera_id=jetson_number)
        return image

    def get_triangulated_images(self, short_video: bool = False) -> Generator:
        for i, (image_3, image_1, box_3, box_1, label_3, label_1, image_path_3, image_path_1) in enumerate(
                self.dataset):

            self.pitch_image = np.array(Image.open("images/pitch.jpg"))  # Clear the image

            dets = []

            image_3 = self.process_camera_data(image_3, box_3, dets, jetson_number=3, index=i)
            image_1 = self.process_camera_data(image_1, box_1, dets, jetson_number=1, index=i)

            det = self.tracker.multi_camera_analysis(dets) if dets else None

            if det is not None:
                det.x, det.y = self.convert_det_to_pixels(det)
                self.pitch_image = self.draw_point(int(det.x), int(det.y))
                if self.draw_text:
                    self.pitch_image = cv2.putText(self.pitch_image, f"x: {det.x}, y: {det.y}", (10, 50),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                self.pitch_image = self.draw_point(0, 0)

            # Print a progress message
            if i % 500 == 0:
                print(f"Processed {i * 2} images")

            if short_video and i == 600:  # For testing; break after 500 images
                print("breaking after 200 images")
                break

            yield image_3, image_1, self.pitch_image

    def run(self,
            video_name: str,
            show_images: bool = False,
            save_video: bool = True,
            short_video: bool = False,
            ) -> None:
        # Create a cv2 VideoWriter object
        if save_video:
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 60, (1280, 2160))

        self.timer.start()

        # Loop through self.get_triangulated_images(); update the plot; write the frame to the video
        for i in self.get_triangulated_images(short_video):
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
            if save_video:
                out.write(stacked_image)

        self.timer.stop()
        print(f"Time taken: {self.timer.get_elapsed_time()}")

        # Release the VideoWriter object
        if save_video:
            out.release()


def main():
    triangulation = TriangulationVisualization(small_dataset=False, use_formplane=False, visualize_homography=True,
                                               draw_text=True)
    # triangulation.run("14_22_time_20_40_14_25__v1__16_1_23.avi.avi", show_images=False, save_video=True)
    triangulation.run("test_4_with_smoothing.avi", show_images=False, save_video=True, short_video=False)


if __name__ == '__main__':
    main()
