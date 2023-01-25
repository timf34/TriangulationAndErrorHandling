from typing import Tuple, Dict


def get_image_field_coordinates() -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    This function returns a dict containing the boundaries of the field in the image for each camera.
    Where the camera ID's are "1" and "3"
    Coordinates are in the format (x, y), and start from the top left corner and go clockwise.
    Also note that the top left corner is the origin here (i.e. 0,0)
    :return: Dict[str, Tuple[int, int, int, int]]
    """
    return {
        "1": ((0, 580), (1918, 576), (1920, 1080), (0, 1080)),
        "3": ((0, 260), (1920, 230), (1920, 980), (0, 740)),
    }
