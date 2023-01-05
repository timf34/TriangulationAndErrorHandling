import argparse
import sys

from data import bohs_dataset
from utils import timer as timer
from triangulation_logic import MultiCameraTracker
from triangulation_visualization import JETSON1_REAL_WORLD, JETSON3_REAL_WORLD, get_xy_from_box, x_y_to_detection


from typing import List, Tuple, Union, Optional, Generator


class CameraNodeScript:
    def __init__(self, cameras: Optional[List[str]], camera_id: str):
        self.dataset = bohs_dataset.create_triangulation_dataset(small_dataset=False, cameras=cameras, single_camera=True)
        self.tracker = MultiCameraTracker()
        self.tracker.add_camera(1, JETSON1_REAL_WORLD)
        self.tracker.add_camera(3, JETSON3_REAL_WORLD)
        self.camera_id: str = camera_id

    def get_triangulated_data(self) -> Generator:
        for i, (image, box, label, image_path, image_path) in enumerate(
                self.dataset):

            if box.size != 0:
                x_3, y_3 = get_xy_from_box(box)
                x_3 = 1920 - x_3  # note: mirroring for Jetson3 to bring the origins a bit closer together in the diff plances (in my mind at least, haven't tested to see if it works better yet)
                cam_det = self.x_y_to_detection(x_3, y_3, i, camera_id=3)

                payload = {
                    "camera": self.camera_id,
                    "detection": cam_det,
                    "image_path": image_path,
                }

            else:
                cam_det = None

            # Print a progress message
            if i % 500 == 0:
                print(f"Processed {i * 2} images")

            yield payload

    def run(self) -> None:
        pass


def main():
    parser = argparse.ArgumentParser(description="AWS IoT Core MQTT Client")
    parser.add_argument("-c", "--cameras", action="store", default="0", dest="cameras", help="The camera path as a string")
    parser.add_argument("-n", "--camera_id", action="store", default="0", dest="camera_id", help="The camera ID (required)")
    args = parser.parse_args()

    # Convert the string to a list
    cameras = args.cameras.split(",")
    camera_node_script = CameraNodeScript(cameras=cameras, camera_id=args.camera_id)
    # camera_node_script.run()


if __name__ == "__main__":
    main()
