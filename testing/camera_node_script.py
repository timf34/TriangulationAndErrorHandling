import argparse
import os
import sys

# TODO: add code to be able to execute this from within a sub-directory

from time import sleep

from data import bohs_dataset
from iot.IOTClient import IOTClient
from iot.IOTContext import IOTContext, IOTCredentials
from iot.config import CAMERA_TOPIC
from utils import timer as timer
from triangulation_logic import MultiCameraTracker
from triangulation_visualization import JETSON1_REAL_WORLD, JETSON3_REAL_WORLD, get_xy_from_box, x_y_to_detection


from typing import List, Tuple, Union, Optional, Generator


class CameraNodeScript:
    def __init__(self, cameras: Optional[List[str]], camera_id: str, iot_manager: IOTClient):
        self.dataset = bohs_dataset.create_triangulation_dataset(small_dataset=False, cameras=cameras, single_camera=True)
        self.tracker = MultiCameraTracker()
        self.tracker.add_camera(1, JETSON1_REAL_WORLD)
        self.tracker.add_camera(3, JETSON3_REAL_WORLD)
        self.camera_id: str = camera_id
        self.iot_manager: IOTClient = iot_manager

    def get_triangulated_data(self) -> Generator:
        for i, (image, box, label, image_path) in enumerate(
                self.dataset):

            if box.size != 0:
                x_3, y_3 = get_xy_from_box(box)
                x_3 = 1920 - x_3  # note: mirroring for Jetson3 to bring the origins a bit closer together in the diff plances (in my mind at least, haven't tested to see if it works better yet)
                cam_det = x_y_to_detection(x_3, y_3, i, camera_id=3)

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
        for payload in self.get_triangulated_data():
            self.iot_manager.publish(payload=payload)
            sleep(1)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="AWS IoT Core MQTT Client")
    parser.add_argument("-p", "--cameras", action="store", default="jetson1_date_01_04_2022_time__20_40_14_25", dest="cameras", help="The camera path as a string")
    parser.add_argument("-n", "--camera_id", action="store", default="0", dest="camera_id", help="The camera ID (required)")
    parser.add_argument("-c", "--cert_path", action="store",
                        default="./certificates/tims/camera_send_messages/3da7dc68bfa5d09b723ebb9068a96d54550c1555969088ec7398103e772196d2-certificate.pem.crt",
                        dest="cert_path", help="Cert ending in .pem.crt")
    parser.add_argument("-k", "--priv_key_path", action="store",
                        default="./certificates/tims/camera_send_messages/3da7dc68bfa5d09b723ebb9068a96d54550c1555969088ec7398103e772196d2-private.pem.key",
                        dest="priv_key_path", help="Private key ending in .pem.key")
    parser.add_argument("-r", "--root_ca_path", action="store",
                        default="./certificates/tims/camera_send_messages/root.pem",
                        dest="root_ca_path", help="Root CA ending in .pem (usually: AmazonRootCA1.pem")
    parser.add_argument("-u", "--client_id", action="store", default="user5", dest="client_id", help="Username")
    args = parser.parse_args()

    cwd = os.getcwd()

    iot_context = IOTContext()

    iot_credentials = IOTCredentials(
        cert_path=os.path.join(cwd, args.cert_path),
        client_id=args.client_id,
        endpoint="a13d7wu4wem7v1-ats.iot.eu-west-1.amazonaws.com",
        region="eu-west-1",
        priv_key_path=os.path.join(cwd, args.priv_key_path),
        ca_path=os.path.join(cwd, args.root_ca_path),
    )

    iot_manager = IOTClient(iot_context, iot_credentials, publish_topic=CAMERA_TOPIC)
    connect_future = iot_manager.connect()
    connect_future.result()
    print("Connected!")


    # Convert the string to a list
    cameras = args.cameras.split(",")
    camera_node_script = CameraNodeScript(cameras=cameras, camera_id=args.camera_id, iot_manager=iot_manager)
    camera_node_script.run()


if __name__ == "__main__":
    main()
