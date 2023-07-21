from typing import Dict, List, Tuple

from monai.config import KeysCollection
from monai.transforms import MapTransform, Transform
from monai.utils import ensure_tuple_rep


class ParseLabelMeDetectionLabel(Transform):
    """
    A transform that parses the JSON from labelme to get bounding boxes
    and their corresponding labels.

    Args:
        spatial_size: the spatial size (width, height) of the image that the bounding boxes should normalized to
    """

    def __init__(self, spatial_size=None):
        self.spatial_size = spatial_size  # W, H

    def __call__(self, json_obj: Dict) -> Tuple[List[List[float]], List[str]]:
        """
        Parses the JSON object and returns a tuple containing a list of bounding boxes and a list of labels

        Args:
            json_obj: a JSON object in labelme format

        Returns:
            A tuple containing:
            - boxes: a list of lists where each inner list contains the coordinates of a bounding box
                in the format [x_min, y_min, x_max, y_max], normalized with self.spatial_size
            - labels: a list of strings containing the corresponding labels for each bounding box
        """
        boxes = []
        labels = []

        size = json_obj["imageWidth"], json_obj["imageHeight"]  # [W, H]

        shapes = json_obj["shapes"]
        for shape in shapes:
            label = shape.get("label")

            points = shape["points"]
            x_min, y_min, x_max, y_max = (
                points[0][0],
                points[0][1],
                points[1][0],
                points[1][1],
            )
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min

            if self.spatial_size is not None:
                x_min = x_min / size[0] * self.spatial_size[0]
                y_min = y_min / size[1] * self.spatial_size[1]
                x_max = x_max / size[0] * self.spatial_size[0]
                y_max = y_max / size[1] * self.spatial_size[1]

            box = [x_min, y_min, x_max, y_max]
            boxes.append(box)
            labels.append(label)

        return boxes, labels


class ParseLabelMeDetectionLabeld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        box_keys: KeysCollection,
        label_keys: KeysCollection,
        spatial_size=None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

        self.box_keys = ensure_tuple_rep(box_keys, len(self.keys))
        self.label_keys = ensure_tuple_rep(label_keys, len(self.keys))

        if not len(self.keys) == len(self.label_keys) == len(self.box_keys):
            raise ValueError(
                "Please make sure len(self.keys)==len(label_keys)==len(box_keys)!"
            )

        self.t = ParseLabelMeDetectionLabel(spatial_size=spatial_size)

    def __call__(self, data):
        """
        Args:
            data: A list of labels, each label is a list bounding boxes
        """
        d = dict(data)
        for key, box_key, label_key in self.key_iterator(
            d, self.box_keys, self.label_keys
        ):
            d[box_key], d[label_key] = self.t(d[key])
        return d
