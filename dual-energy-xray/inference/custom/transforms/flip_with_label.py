from monai.transforms import MapTransform, RandomizableTransform, Flip

class RandFlipWithLabeld(RandomizableTransform, MapTransform):
    def __init__(
        self,
        image_key: str,
        label_key: str,
        spatial_axis: int = 0,
    ):
        RandomizableTransform.__init__(self, 0.5)
        self.image_key = image_key
        self.label_key = label_key
        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self,data):
        d = dict(data)
        self.randomize(None)
        if self._do_transform:
            d[self.image_key] = self.flipper(d[self.image_key])
            d[self.label_key] = [1]
        else:
            d[self.label_key] = [0]
        return d
