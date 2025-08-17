import cv2
import numpy

class Preprocessor:
    """
    Image preprocessor for Hailo face detection models. Assume color space is RGB.
    """

    def __init__(self, input_shape_nhwc):
        self.fill_color = (0, 0, 0)
        self.input_shape_nhwc = input_shape_nhwc

    def _apply_pad_method(self, image):
        """Applies pad method to the image.
        Returns a tuple containing modified image and coordinate conversion function
        """
        resample_method = cv2.INTER_LINEAR

        _, h, w, _ = self.input_shape_nhwc

        dx = dy = 0  # offset of left top corner of original image in resized image
        image_obj = image
        iw, ih = self._get_image_size(image)
        coord_conv = lambda x, y: (
            iw * min(1, max(0, x / w)),
            ih * min(1, max(0, y / h)),
        )
        if iw != w or ih != h:
            scale = min(w / iw, h / ih)
            nw = min(round(iw * scale), w)
            nh = min(round(ih * scale), h)
            gain = min(nh / ih, nw / iw)
            # resize preserving aspect ratio
            scaled_image = self._resize(image_obj, nw, nh, resample_method)

            if ih != iw:
                image_obj, dx, dy = self._pad(
                    scaled_image, w, h, self.fill_color
                )
            else:
                image_obj = scaled_image
                
            coord_conv = lambda x, y: (
                min(max(0, (x - dx) / gain), iw),
                min(max(0, (y - dy) / gain), ih),
            )
        return image_obj, coord_conv

    def _get_image_size(self, img):
        """Helper method for getting the size of an image
            - `img`: the image to get the size of
        Returns tuple of width, height
        """
        return img.shape[1], img.shape[0]

    def _resize(self, img, width, height, resample_method):
        """Helper method for resizing an image
            - `img`: the image to resize
            - `width`: desired image width
            - `height`: desired image height
            - `resample_method`: resampling method
        Returns resized image
        """
        return cv2.resize(img, (width, height), interpolation=resample_method)

    def _pad(self, img, width, height, fill_color):
        """Helper method for adding padding to an image
            - `img`: the image add padding to
            - `width`: desired image width
            - `height`: desired image height
            - `fill_color`: color tuple for what color to pad with
        Returns tuple with new image, left pixel coordinate of original image inside new image
        and top pixel coordiante of original image inside new image
        """
        iw, ih = self._get_image_size(img)
        left = (width - iw) // 2
        top = (height - ih) // 2

        image_obj = numpy.zeros((height, width, 3), img.dtype)
        if self.fill_color != (0, 0, 0):
            image_obj[:] = fill_color

        image_obj[top : top + ih, left : left + iw, :] = img
        return image_obj, left, top

    def forward(self, image):
        """Implementation for 'opencv'"""
        assert len(image.shape) == 3
        image_input = image.copy()

        #
        # resize + pad/crop
        #
        image_obj, coord_conv = self._apply_pad_method(image_input)

        # convert to model data type
        image_obj = image_obj.astype(numpy.uint8)

        return image_obj, coord_conv
