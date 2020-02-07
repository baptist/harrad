import random
import torch
import numbers
import numpy as np

from .affine_transforms import Zoom,Rotate
from .utils import th_random_choice
import cv2

def _blend(img1, img2, alpha):
    """
    Weighted sum of two images
    Arguments
    ---------
    img1 : torch tensor
    img2 : torch tensor
    alpha : float between 0 and 1
        how much weight to put on img1 and 1-alpha weight
        to put on img2
    """
    return img1.mul(alpha).add(1 - alpha, img2)


class Grayscale(object):

    def __init__(self, keep_channels=False):
        """
        Convert RGB image to grayscale
        Arguments
        ---------
        keep_channels : boolean
            If true, will keep all 3 channels and they will be the same
            If false, will just return 1 grayscale channel
        """
        self.keep_channels = keep_channels
        if keep_channels:
            self.channels = 3
        else:
            self.channels = 1

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input_dst = _input[0]*0.299 + _input[1]*0.587 + _input[2]*0.114
            _input_gs = _input_dst.repeat(self.channels,1,1)
            outputs.append(_input_gs)
        return outputs if idx > 1 else outputs[0]


class RandomGrayscale(object):

    def __init__(self, p=0.5):
        """
        Randomly convert RGB image(s) to Grayscale w/ some probability,
        NOTE: Always retains the 3 channels if image is grayscaled
        p : a float
            probability that image will be grayscaled
        """
        self.p = p

    def __call__(self, *inputs):
        pval = random.random()
        if pval < self.p:
            outputs = Grayscale(keep_channels=True)(*inputs)
        else:
            outputs = inputs
        return outputs


class Gamma(object):

    def __init__(self, value):
        """
        Performs Gamma Correction on the input image. Also known as
        Power Law Transform. This function transforms the input image
        pixelwise according
        to the equation Out = In**gamma after scaling each
        pixel to the range 0 to 1.
        Arguments
        ---------
        value : float
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.value = value

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = torch.pow(_input, self.value)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]

class RandomGamma(object):

    def __init__(self, min_val, max_val):
        """
        Performs Gamma Correction on the input image with some
        randomly selected gamma value between min_val and max_val.
        Also known as Power Law Transform. This function transforms
        the input image pixelwise according to the equation
        Out = In**gamma after scaling each pixel to the range 0 to 1.
        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        NOTE:
        for values:
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Gamma(value)(*inputs)
        return outputs

class RandomChoiceGamma(object):

    def __init__(self, values, p=None):
        """
        Performs Gamma Correction on the input image with some
        gamma value selected in the list of given values.
        Also known as Power Law Transform. This function transforms
        the input image pixelwise according to the equation
        Out = In**gamma after scaling each pixel to the range 0 to 1.
        Arguments
        ---------
        values : list of floats
            gamma values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.
        NOTE:
        for values:
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = th_random_choice(self.values, p=self.p)
        outputs = Gamma(value)(*inputs)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

class Brightness(object):
    def __init__(self, value):
        """
        Alter the Brightness of an image
        Arguments
        ---------
        value : brightness factor
            =-1 = completely black
            <0 = darker
            0 = no change
            >0 = brighter
            =1 = completely white
        """
        self.value = max(min(value,1.0),-1.0)

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = torch.clamp(_input.float().add(self.value).type(_input.type()), 0, 1)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]

class RandomBrightness(object):

    def __init__(self, min_val, max_val):
        """
        Alter the Brightness of an image with a value randomly selected
        between `min_val` and `max_val`
        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Brightness(value)(*inputs)
        return outputs

class RandomChoiceBrightness(object):

    def __init__(self, values, p=None):
        """
        Alter the Brightness of an image with a value randomly selected
        from the list of given values with given probabilities
        Arguments
        ---------
        values : list of floats
            brightness values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.
        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = th_random_choice(self.values, p=self.p)
        outputs = Brightness(value)(*inputs)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

class Saturation(object):

    def __init__(self, value):
        """
        Alter the Saturation of image
        Arguments
        ---------
        value : float
            =-1 : gray
            <0 : colors are more muted
            =0 : image stays the same
            >0 : colors are more pure
            =1 : most saturated
        """
        self.value = max(min(value,1.0),-1.0)

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _in_gs = Grayscale(keep_channels=True)(_input)
            alpha = 1.0 + self.value
            _in = torch.clamp(_blend(_input, _in_gs, alpha), 0, 1)
            outputs.append(_in)
        return outputs if idx > 1 else outputs[0]

class RandomSaturation(object):

    def __init__(self, min_val, max_val):
        """
        Alter the Saturation of an image with a value randomly selected
        between `min_val` and `max_val`
        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Saturation(value)(*inputs)
        return outputs

class RandomChoiceSaturation(object):

    def __init__(self, values, p=None):
        """
        Alter the Saturation of an image with a value randomly selected
        from the list of given values with given probabilities
        Arguments
        ---------
        values : list of floats
            saturation values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.
        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = th_random_choice(self.values, p=self.p)
        outputs = Saturation(value)(*inputs)
        return outputs

# ----------------------------------------------------
# ----------------------------------------------------

class Contrast(object):
    """
    """
    def __init__(self, value):
        """
        Adjust Contrast of image.
        Contrast is adjusted independently for each channel of each image.
        For each channel, this Op computes the mean of the image pixels
        in the channel and then adjusts each component x of each pixel to
        (x - mean) * contrast_factor + mean.
        Arguments
        ---------
        value : float
            smaller value: less contrast
            ZERO: channel means
            larger positive value: greater contrast
            larger negative value: greater inverse contrast
        """
        self.value = value

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            channel_means = _input.mean().mean(2)
            channel_means = channel_means.expand_as(_input)
            _input = torch.clamp((_input - channel_means) * self.value + channel_means,0,1)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]

class RandomContrast(object):

    def __init__(self, min_val, max_val):
        """
        Alter the Contrast of an image with a value randomly selected
        between `min_val` and `max_val`
        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Contrast(value)(*inputs)
        return outputs

class RandomChoiceContrast(object):

    def __init__(self, values, p=None):
        """
        Alter the Contrast of an image with a value randomly selected
        from the list of given values with given probabilities
        Arguments
        ---------
        values : list of floats
            contrast values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.
        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = th_random_choice(self.values, p=None)
        outputs = Contrast(value)(*inputs)
        return outputs



def video_transform(images, target,
                    size=(100, 100),
                    resize_to=120,
                    roi=None,
                    maintain_aspect_ratio=True,
                    apply_focus=False,
                    apply_resize=False,
                    apply_randomcrop=False,
                    apply_centercrop=False,
                    apply_hflip=False,
                    apply_vflip=False,
                    apply_brightness=False,
                    apply_saturation=False,
                    apply_rotate=False,
                    apply_zoom=False,
                    apply_contrast=False,
                    apply_xshift=False,
                    apply_yshift=False,
                    apply_wb=False):

    if not apply_xshift:
        apply_xshift = 0

    if not apply_yshift:
        apply_yshift = 0

    if images.ndim >= 3:

        if apply_focus:
            x1, y1, x2, y2 = map(int, roi)
            cropsize = max(abs(x1 - x2), abs(y1 - y2))
            marginx, marginy = abs(abs(x1 - x2) - cropsize) // 2, abs(abs(y1 - y2) - cropsize) // 2

            images = images[:, max(0, y1 - marginy + apply_yshift):min(y2 + marginy + apply_yshift, images.shape[1]), max(0, x1 - marginx + apply_xshift):min(x2 + marginx + apply_xshift, images.shape[2])]

            # cx, cy, w, h = roi
            #
            # if h != images.shape[1] or w != images.shape[2]:
            #     images = images[:, max(0, cy - 2*h): min(cy + 2*h, images.shape[1]),
            #                        max(0, cx - 2*w): min(cx + 2*w, images.shape[2])]



        if apply_resize:
            # Resize to smallest or largest size
            op = min if apply_randomcrop or apply_centercrop else max
            newsize = resize_to#op(size)
            h, w = int(images.shape[1] / float(op(images.shape[1:3])) * newsize), \
                   int(images.shape[2] / float(op(images.shape[1:3])) * newsize)
            images = np.array([cv2.resize(img, (w, h)) for img in images])


        if images.ndim < 4:
            images = images[:, :, :, np.newaxis]

        h, w = images.shape[1:3]

        # th, tw = size if type(size) is tuple else ((max(size, h / min(w, h) * size),
        #                                             max(size, w / min(w, h) * size))
        #                                            if maintain_aspect_ratio
        #                                            else (size, size))

        th, tw = size if type(size) is tuple else (size, size)

        if apply_randomcrop:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

        if apply_centercrop:
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))


        if (not apply_centercrop and not apply_randomcrop):# or h < th or w < tw:
            tmp = np.zeros((images.shape[0], th, tw, images.shape[3]))
            tmp[:, (th-h)//2:(th-h)//2 + h, (tw-w)//2:(tw-w)//2 + w] = images
            images = tmp
        elif images.shape[1:3] != size:
            images = images[:, y1:y1 + th, x1: x1 + tw]

        if apply_hflip and random.random() >= .5 and target not in ['swiping_left', 'swiping_right']:
            images = images[:, :, ::-1]

        if apply_vflip and random.random() >= .5:
            images = images[:, ::-1]

        images = images / 255.

        if apply_contrast:
            contrast = Contrast(random.uniform(0.0, 2.0))

        if apply_brightness:
            brightness = Brightness(random.uniform(-0.12, 0.12))

        if apply_saturation:
            saturation = Saturation(random.uniform(-0.33, 0.33))

        if apply_rotate:
            rotate = Rotate(random.randint(-10, 10))

        if apply_zoom:
            zoom = Zoom(random.uniform(0.8, 1.0) if not type(apply_zoom) is float  else apply_zoom)

        augmented = []
        for i in range(len(images)):

            image = torch.from_numpy(images[i].transpose(2, 0, 1))

            if apply_saturation and image.shape[2] == 3:
                image = saturation(image)

            if apply_brightness and image.shape[2] == 3:
                image = brightness(image)

            if apply_contrast and image.shape[2] == 3:
                image = contrast(image)

            if apply_rotate:
                image = rotate(image)

            if apply_zoom:
                image = zoom(image)

            if apply_wb:
                # image = do_wb(image)
                image = noisy('poisson', image)
                image = noisy('speckle', image)
                image *= .17
            # augmented.append(image)
            augmented.append(image.numpy())

        # import matplotlib.pylab as plt
        # from matplotlib.patches import Rectangle
        # f, axes = plt.subplots(2, 10)
        # for i, j in enumerate(range(0, images.shape[0], images.shape[0] // 10)):
        #     if i >= 10:
        #         break
        #     # rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False)
        #     axes[0, i].imshow(images[j])
        #     axes[1, i].imshow((augmented[j]).transpose(1,2,0))
        #     # axes[i].add_artist(rect)
        # plt.show()


        return torch.from_numpy(np.asarray(augmented, dtype=np.float32))

def do_wb(image):

    image = np.transpose(image, [1,2,0])

    XYZ_to_sRGB_matrix = np.zeros((3, 3))
    XYZ_to_sRGB_matrix[0] = [3.2406, -1.5372, -0.4986]
    XYZ_to_sRGB_matrix[1] = [-0.9689, 1.8758, 0.0415]
    XYZ_to_sRGB_matrix[2] = [0.0557, -0.2040, 1.0570]



    bradford = np.array([[0.8951000, 0.2664000, -0.1614000],
                         [-0.7502000, 1.7135000, 0.0367000],
                         [0.0389000, -0.0685000, 1.0296000]])

    target_xy_10000K = np.array([0.2824, 0.2898])
    target_xy_2000K = np.array([0.53, 0.412])
    target_xy_3000K = np.array([0.4388, 0.4095])

    target_X = target_xy_10000K[0]/target_xy_10000K[1]
    target_Y = 1
    target_Z = (1 - target_xy_10000K[0] - target_xy_10000K[1])/target_xy_10000K[1]


    D65_XYZ = np.array([0.95, 1.0, 1.088])
    D65_XYZ = np.array([0.95, 1.0, 1.088])
    srcIllumBrad = np.dot(bradford, D65_XYZ)

    dstIllumBrad = np.dot(bradford, np.array([target_X, target_Y, target_Z]))
    bradford_gains = np.divide(dstIllumBrad, srcIllumBrad)
    bradford_gains_diag = np.diag(bradford_gains)


    fx = XYZ_to_sRGB_matrix @ np.linalg.inv(bradford) @ bradford_gains_diag @ bradford @ np.linalg.inv(XYZ_to_sRGB_matrix)

    image = image ** 2.2 # gamma compressed to linear
    image = np.expand_dims(image, 3)
    image = fx @ image
    image = np.clip(np.squeeze(image), 0.0, 1.0)
    image = image **0.454545

    image = np.transpose(image, [2,0,1])

    return image


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        gauss = np.random.normal(0, .1 ** 0.5, (row, col, ch))
        noisy = image + torch.from_numpy(gauss)
        return np.clip(noisy, 0, 1)
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * row*col*ch * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * row*col*ch * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        noisy = image + image * torch.from_numpy(np.random.randn(row, col, ch)*.5)
        return np.clip(noisy, 0, 1.0)
