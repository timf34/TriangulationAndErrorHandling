import cv2
from copy import deepcopy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Generator

from data.bohs_dataset import create_triangulation_dataset
from utils.data_classes import Detections
from utils.timer import Timer
from utils.utils import x_y_to_detection, get_xy_from_box, draw_bboxes_red
from triangulation_logic import MultiCameraTracker


class TriangulationVisualization:
    """
    This class is used to visualize the triangulation logic. It is not used in the actual prod code.

    What we need is:
    - TriangulationBohsDataset object (dataset with dets from both cameras)
        - This dataset contains the images from the cameras also!
    - Video from both cameras
    """
    JETSON1_REAL_WORLD = np.array([[-19.41], [-21.85], [7.78]])
    JETSON3_REAL_WORLD = np.array([[0.], [86.16], [7.85]])

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
        self.use_formplane: bool = use_formplane

        self.tracker = MultiCameraTracker(use_formplane=self.use_formplane)  # TODO: this should be passed in
        self.tracker.add_camera(1, self.JETSON1_REAL_WORLD)
        self.tracker.add_camera(3, self.JETSON3_REAL_WORLD)
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

    def draw_point(self, x: int, y: int, camera_id: int = None) -> np.array:
        """
        This function allows us to draw a point on self.pitch_image. Note we have to call cv2.circle before cv2.putText
        because cv2.putText draws on top of the image.
        :param camera_id:
        :param x: x coordinate of point
        :param y: y coordinate of point
        :return: image with point drawn on it
        """
        # Limit x and y to be within pitch boundaries.
        if x <= 0 or x >= self.pitch_width or y <= 0 or y >= self.pitch_height:
            x = 0
            y = 0

        # Define color mappings for different camera ids.
        color_mapping = {1: ((0, 255, 0), (0, 0, 0)),
                         3: ((0, 0, 255), (255, 255, 255)),
                         None: ((255, 0, 0), None)}

        # Set color based on camera id.
        color, text_color = color_mapping.get(camera_id, ((255, 0, 0), None))

        # Draw a circle at the given point.
        cv2.circle(self.pitch_image, (x, y), 20, color, -3)

        # If draw_text is True and camera_id is provided, draw a text at the given point.
        if self.draw_text and camera_id in {1, 3}:
            cv2.putText(
                self.pitch_image,
                str(camera_id),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                2,
            )

        # Return the updated image.
        return self.pitch_image

    @staticmethod
    def create_plot():
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5), tight_layout=True,
                                            gridspec_kw={'hspace': .1})
        return fig, ax1, ax2, ax3

    def process_camera_data(self,
                            image: np.ndarray,
                            box: np.ndarray,
                            dets: List,
                            jetson_number: int,
                            index: int
                            ) -> np.ndarray:
        """
        Process data from camera. This includes drawing bounding boxes on the image, transforming
        x, y coordinates based on specific camera properties, appending the detection to a list,
        and visualizing homography if applicable.

        :param image: The image to process.
        :param box: A numpy array representing the box. Size is 4 when full, 0 when empty.
        :param dets: A list of detections.
        :param jetson_number: An identifier for the camera.
        :param index: The index of the detection.
        :return: The processed image.
        """

        # Process box if it's not empty
        if box.size != 0:
            x, y = get_xy_from_box(box)

            # Draw text on the image if needed
            if self.draw_text:
                text = f"x: {x}, y: {y}"
                color = (255, 0, 0)
                image = cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Draw bounding boxes on the image
            image = draw_bboxes_red(image, x, y)

            # Adjust x for Jetson3
            if jetson_number == 3:
                # note: mirroring for Jetson3 to bring the origins a bit closer together
                # in the diff planes (in my mind at least, haven't tested to see if it works better yet)
                x = 1920 - x

                # Transform x, y to detection and append to list
            cam_det = x_y_to_detection(x, y, index, camera_id=jetson_number)
            dets.append(cam_det)

            # Visualize homography if needed
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

    def process_and_save_frame(
            self,
            img1,
            img2,
            pitch_image,
            video_writer=None,
            show_images: bool = False
    ) -> None:
        # Convert all images to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        pitch_image = cv2.cvtColor(pitch_image, cv2.COLOR_BGR2RGB)

        # Resize images
        img1 = cv2.resize(img1, (1280, 720))
        img2 = cv2.resize(img2, (1280, 720))
        pitch_image = cv2.resize(pitch_image, (1280, 720))

        # Stack the images together
        stacked_image = np.vstack((img1, img2, pitch_image))

        # Show image if required
        if show_images:
            cv2.imshow("Stacked image", stacked_image)
            cv2.waitKey(1)

        # Save video frame if required
        if video_writer:
            video_writer.write(stacked_image)

    def run(self,
            video_name: str,
            show_images: bool = False,
            save_video: bool = True,
            short_video: bool = False,
            ) -> None:
        # Create a cv2 VideoWriter object
        video_writer = None
        if save_video:
            video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 60, (1280, 2160))

        self.timer.start()

        # Loop through self.get_triangulated_images(); update the plot; write the frame to the video
        for i in self.get_triangulated_images(short_video):
            img1, img2, pitch_image = i
            self.process_and_save_frame(img1, img2, pitch_image, video_writer=video_writer, show_images=show_images)

        self.timer.stop()
        print(f"Time taken: {self.timer.get_elapsed_time()}")

        # Release the VideoWriter object
        if save_video:
            video_writer.release()


def main():
    triangulation = TriangulationVisualization(small_dataset=False, use_formplane=False, visualize_homography=False,
                                               draw_text=False)
    # triangulation.run("14_22_time_20_40_14_25__v1__16_1_23.avi.avi", show_images=False, save_video=True)
    triangulation.run("test_5_with_smoothing_no_text.avi", show_images=False, save_video=True, short_video=True)


if __name__ == '__main__':
    main()
