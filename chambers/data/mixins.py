
class ImageLabelMixin:
    def map_image(self, func, *args, **kwargs):
        def fn(images, labels):
            return func(images, *args, **kwargs), labels

        self.map(fn)