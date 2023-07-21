from typing import Any, Dict, List

from manafaln.data import RatioSampler


class CACSampler(RatioSampler):
    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        key: str = "cac_score",
        ratio: float = 1.0,
        threshold=400,
    ):
        self.threshold = threshold
        super().__init__(data_list, key, ratio)

    def get_labels(self, data_list: List[Dict[str, Any]]) -> List[bool]:
        labels = [bool(data[self.key] > self.threshold) for data in data_list]
        return labels
