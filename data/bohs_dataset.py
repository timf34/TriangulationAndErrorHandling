import cv2
import os
import torch

import numpy as np

from copy import deepcopy
from PIL import Image
from typing import List, Tuple

from data.utils import read_bohs_ground_truth

BALL_BBOX_SIZE = 20
BALL_LABEL = 1

# I could have probably used inheritance here to some extent at least... IDK


class TriangulationBohsDataset(torch.utils.data.Dataset):
    """
    A class for loading the bohs dataset.
    """

    def __init__(self,
                 only_ball_frames: bool = False,
                 whole_dataset: bool = True,
                 dataset_size: int = 1,
                 image_extension: str = '.jpg',
                 image_name_length: int = 7,
                 only_matching_frames: bool = True,
                 small_dataset: bool = False,
                 start_frame: int = 600,
                 end_frame: int = 1000,
                 ):
        """
        Initializes the dataset.
        :param image_folder_path: Path to 'bohs-preprocessed' folder
        :param only_ball_frames: Whether to only use ball frames.
        :param whole_dataset: Whether to use the whole dataset.
        :param dataset_size: The size of the dataset to use if not using whole_dataset.
        :param transform: The transform to apply to the dataset.
        """
        print("Whole dataset: ", whole_dataset)
        self.dataset_path: str = r"C:\Users\timf3\OneDrive - Trinity College Dublin\Documents\Documents\datasets\Datasets\Bohs\bohs-preprocessed"
        self.only_ball_frames = only_ball_frames
        self.whole_dataset = whole_dataset
        self.dataset_size = dataset_size
        self.small_dataset = small_dataset
        self.cameras: List[str] = [
            # "jetson3_1_4_2022_time__19_45_01_4",
            # "jetson1_date_01_04_2022_time__19_45_01_4",
            "jetson3_1_4_2022_time__20_40_14_25",
            "jetson1_date_01_04_2022_time__20_40_14_25"
        ]
        self.image_name_length = image_name_length
        self.image_extension: str = image_extension

        self.only_matching_frames: bool = only_matching_frames  # Only show frames where both cameras have a frame with a ball.

        self.gt_annotations: dict = {}
        self.image_list: list = []

        # The folder paths we will be using.
        self.image_folder_path = os.path.join(self.dataset_path, 'unpacked_jpg')
        self.annotations_folder_path = os.path.join(self.dataset_path, 'annotations')

        # TODO: I need to write a file/ page on what the different data structures that I have are.

        for camera_id in self.cameras:
            # Ensure annotations file path exists
            annotations_file_path = os.path.join(self.annotations_folder_path, camera_id + '.xml')
            assert os.path.exists(annotations_file_path), f"Annotations file path {annotations_file_path} does not exist."

            # Read ground truth data for the sequence
            self.gt_annotations[camera_id] = read_bohs_ground_truth(annotations_path=self.annotations_folder_path,
                                                                    xml_file_name=f'{camera_id}.xml')

            # TODO: also note that we are only using images that include the ball currently
            # TODO: I need to adjust this! Make it so we include all frames... or at least, with the only_ball_frames flag, we can choose to only include frames with the ball.

            if self.only_ball_frames:
                annotated_frames: List[int] = list(set(self.gt_annotations[camera_id].ball_pos))
            else:
                annotated_frames: List[int] = list(set(self.gt_annotations[camera_id].ball_pos))
                annotated_frames = [i for i in range(1, max(annotated_frames) + 1)]

            images_path = os.path.join(self.image_folder_path, camera_id)

            for e in annotated_frames:
                e = str(e)
                e = e.zfill(self.image_name_length)
                file_path = os.path.join(images_path, f'frame_{e}{self.image_extension}')
                if os.path.exists(file_path):
                    self.image_list.append((file_path, camera_id, e))
                else:
                    print("doesn't exist", file_path)
                    print("check whether its frame_000001.png or just 000001.png")


        self.n_images = len(self.image_list)
        print(f"Total number of Bohs Images: {self.n_images}")
        self.ball_images_ndx = set(self.get_elems_with_ball())
        self.no_ball_images_ndx = set([ndx for ndx in range(self.n_images) if ndx not in self.ball_images_ndx])
        print(f'BOHS: {format(len(self.ball_images_ndx))} frames with the ball')
        print(f'BOHS: {(len(self.no_ball_images_ndx))} frames without the ball')

        print("whole dataset", whole_dataset)

        # We now want to filter image_list to only include frames where both cameras have the ball
        if self.only_matching_frames:
            # self.image_list = self.get_matching_frames()
            self.get_matching_frames()
            if self.small_dataset:
                self.image_list = self.image_list[start_frame:end_frame]

    def __len__(self):
        return self.n_images

    def __getitem__(self, ndx):
        # Returns transferred image as a normalized tensor
        image_path_1, image_path_2, camera_id_1, camera_id_2, image_ndx_1, image_ndx_2 = self.image_list[ndx]

        image_1 = Image.open(image_path_1)
        image_2 = Image.open(image_path_2)

        try:
            box_1, label_1 = self.get_annotations(camera_id_1, image_ndx_1)
            box_2, label_2 = self.get_annotations(camera_id_2, image_ndx_2)
        except:
            box_1 = [[]]
            box_2 = [[]]
            label_1 = []
            label_2 = []

        # Convert PIL image to numpy array
        image_1 = np.array(image_1)
        image_2 = np.array(image_2)

        # Write an exception to catch if box_1 is [[]]
        if box_1 != [[]]:
            image_1 = self.draw_bboxes(image_1, box_1)
        if box_2 != [[]]:
            image_2 = self.draw_bboxes(image_2, box_2)

        return image_1, image_2, box_1, box_2, label_1, label_2, image_path_1, image_path_2

    def get_annotations(self, camera_id, image_ndx):
        # Prepare annotations as list of boxes (xmin, ymin, xmax, ymax) in pixel coordinates
        # and torch int64 tensor of corresponding labels
        boxes = []
        labels = []
        # Add annotations for the ball position: positions of the ball centre

        # Convert image_ndx to int -> 'frame_0000001' -> 1
        image_ndx = int(image_ndx[-7:])


        ball_pos = self.gt_annotations[camera_id].ball_pos[int(image_ndx)]
        if ball_pos != [[]]:
            for (x, y) in ball_pos:
                x1 = x - BALL_BBOX_SIZE // 2
                x2 = x1 + BALL_BBOX_SIZE
                y1 = y - BALL_BBOX_SIZE // 2
                y2 = y1 + BALL_BBOX_SIZE
                boxes.append((x1, y1, x2, y2))
                labels.append(BALL_LABEL)

        return np.array(boxes, dtype=np.float64), np.array(labels, dtype=np.int64)

    def get_elems_with_ball(self):
        # Get indexes of images with ball ground truth
        ball_images_ndx = []
        for ndx, (_, camera_id, image_ndx) in enumerate(self.image_list):
            ball_pos = self.gt_annotations[camera_id].ball_pos[int(image_ndx)]
            if len(ball_pos) > 0 and ball_pos != [[]]:  # With the collate function, empty lists are converted to [[]]
                ball_images_ndx.append(ndx)

        return ball_images_ndx

    def get_matching_frames(self) -> List[Tuple[str, str, str, str, str, str]]:
        """
            This function will rewrite self.image_list to contain only frames where both cameras have a ball.
            And where each item of the list will be as follows...
                (image_path_1, image_path_2, camera_id_1, camera_id_2, image_ndx_1, image_ndx_2)
        :return image_list: A list of tuples containing the image paths for both cameras, and the camera ids and image ndx
        """
        matching_frames = []

        if self.only_ball_frames:

            x = self.gt_annotations[self.cameras[0]].ball_pos

            for frame in self.gt_annotations[self.cameras[0]].ball_pos.keys():
                if frame in self.gt_annotations[self.cameras[1]].ball_pos.keys():
                    if self.gt_annotations[self.cameras[0]].ball_pos[frame] != [[]] and self.gt_annotations[self.cameras[1]].ball_pos[frame] != [[]]:  # If both cameras have the ball
                        matching_frames.append(frame)

            image_list: List[str, str, str, str, str, str] = []

            for frame in matching_frames:

                frame = str(frame).zfill(self.image_name_length)
                image_path_1 = os.path.join(self.image_folder_path, self.cameras[0], f'frame_{frame}{self.image_extension}')
                image_path_2 = os.path.join(self.image_folder_path, self.cameras[1], f'frame_{frame}{self.image_extension}')
                image_list.append((image_path_1, image_path_2, self.cameras[0], self.cameras[1], frame, frame))

        else:
            # This is for if we don't have only ball frames. However we still need to ensure that the frames are matching.
            # And that image_list values have 6 values.

            # Plan: I'll iterate through image list until I find the index of the first frame from the 2nd camera_id
            i = 0
            second_camera_index = 0
            second_camera_id = self.cameras[1]
            reached_second_camera_id: bool = False
            while reached_second_camera_id is False:
                if self.image_list[i][1] == second_camera_id:
                    reached_second_camera_id = True
                    second_camera_index = i
                else:
                    i += 1

            # No we will will create two image lists for each camera id
            image_list_1 = deepcopy(self.image_list[:second_camera_index])
            image_list_2 = deepcopy(self.image_list[second_camera_index:])

            # Now we will iterate through both lists and add the matching frames to the self.image_list
            # Get the length of the shorter list
            shorter_list_length = min(len(image_list_1), len(image_list_2))

            image_list = []

            # Note that each image_list has all frames. So we can just iterate through the shorter list. But do note that it doesn't neccessarily start from the start. This is very hacky but will just get it done for now.
            for i in range(shorter_list_length):
                image_list.append((image_list_1[i][0], image_list_2[i][0], self.cameras[0], self.cameras[1], f"frame_{image_list_1[i][2]}", f"frame_{image_list_2[i][2]}"))



            pass


        # Update all our variables to reflect the new image list
        self.image_list = image_list
        self.n_images = len(image_list)
        return image_list

    @staticmethod
    def draw_bboxes(image, boxes):
        """
        Draw bounding boxes on the image
        :param image: image to draw bounding boxes on, numpy array
        :param boxes:...
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        for box in boxes:
            x1, y1, x2, y2 = box
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (255, 0, 0)
            radius = 30
            cv2.circle(image, (x, y), radius, color, 2)
        return image

    @staticmethod
    def create_new_folder(folder_name) -> None:
        """
            This function checks if a folder exists, and if not, creates it.
        """
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
            print(f"Folder {folder_name} was created")
        else:
            print(f"Folder {folder_name} already exists")


def create_triangulation_dataset(
        only_ball_frames: bool = False,
        dataset_size: int = 2,
        image_extension: str = '.jpg',
        cameras: List[int] = None,
        image_name_length: int = 7,
        small_dataset: bool = False,
        start_frame: int = 600,
        end_frame: int = 1000,
):

    if cameras is None:
        cameras = [1]

    return TriangulationBohsDataset(
        only_ball_frames=only_ball_frames,
        dataset_size=dataset_size,
        image_extension=image_extension,
        image_name_length=image_name_length,
        small_dataset=small_dataset,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def main():
    dataset = create_triangulation_dataset()

    for i in range(len(dataset)):
        image_1, image_2, box_1, box_2, label_1, label_2, image_path_1, image_path_2 = dataset[i]
        print(image_path_1, image_path_2)
        print(box_1, box_2)
        print(label_1, label_2)
        print()
        if i == 50:
            break


if __name__ == '__main__':
    main()
