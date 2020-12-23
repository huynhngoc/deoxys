# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import numpy as np
from deoxys_image import normalize, apply_affine_transform, apply_flip
from deoxys_image import ImageAugmentation
from deoxys.keras.preprocessing import ImageDataGenerator
from ..utils import Singleton


class BasePreprocessor:
    def __init__(self, **kwargs):
        """
        Wrapper for creating a trained/fitted Preprocessor instance
        """
        pass

    def transform(self, x, y):
        return x, y


class DummyPreprocessor:
    def __init__(self):
        pass

    def transform(self, x, y):
        return x, y


class SingleChannelPreprocessor(BasePreprocessor):
    def transform(self, x, y):
        x_shape = tuple(list(x.shape) + [1])
        y_shape = tuple(list(y.shape) + [1])

        return x.reshape(x_shape), y.reshape(y_shape)


class WindowingPreprocessor(BasePreprocessor):
    """Used to set the dynamic range of an image.
    """

    def __init__(self, window_center, window_width, channel):
        self.window_center, self.window_width = window_center, window_width
        self.channel = channel

    def perform_windowing(self, image):
        image = image - self.window_center
        image[image < -self.window_width / 2] = -self.window_width / 2
        image[image > self.window_width / 2] = self.window_width / 2
        return image

    def transform(self, images, targets):
        images = images.copy()
        images[..., self.channel] = self.perform_windowing(
            images[..., self.channel])
        return images, targets


class HounsfieldWindowingPreprocessor(WindowingPreprocessor):
    def __init__(self, window_center, window_width, channel,
                 houndsfield_offset=1024):
        super().__init__(
            window_center+houndsfield_offset, window_width, channel)


class ImageNormalizerPreprocessor(BasePreprocessor):
    def __init__(self, vmin=None, vmax=None):
        """
        Normalize all channels to the range of the close interval [0, 1]

        Parameters
        ----------
        vmin : int, optional
            [description], by default 0
        vmax : int, optional
            [description], by default 255
        """
        self.vmin = vmin
        self.vmax = vmax

    def transform(self, images, targets):
        transformed_images = normalize(images, self.vmin, self.vmax)

        return transformed_images, targets


class ChannelRemoval(BasePreprocessor):
    def __init__(self, channel=1):
        self.channel = channel

    def transform(self, images, targets):
        return np.delete(images, self.channel, axis=-1), targets


class ChannelSelector(BasePreprocessor):
    def __init__(self, channel=0):
        if '__iter__' not in dir(channel):
            self.channel = [channel]
        else:
            self.channel = channel

    def transform(self, images, targets):
        remove_channel = [c for c in np.arange(
            images.shape[-1]) if c not in self.channel]
        return np.delete(images, remove_channel, axis=-1), targets


class UnetPaddingPreprocessor(BasePreprocessor):
    def __init__(self, depth=4, mode='auto'):
        self.depth = depth
        self.mode = mode

    def transform(self, images, targets):
        image_shape = images.shape
        target_shape = targets.shape

        shape = image_shape[1:-1]

        divide_factor = 2 ** self.depth

        if len(shape) == 2:
            height, width = shape
            if height % divide_factor != 0:
                new_height = (height // divide_factor + 1) * divide_factor
            else:
                new_height = height

            if width % divide_factor != 0:
                new_width = (width // divide_factor + 1) * divide_factor
            else:
                new_width = width

            new_images = np.zeros(
                (image_shape[0],
                 new_height, new_width,
                 image_shape[-1]))
            new_targets = np.zeros(
                (target_shape[0],
                    new_height, new_width,
                    target_shape[-1]))

            min_h = (new_height - height) // 2
            min_w = (new_width - width) // 2

            new_images[:, min_h: min_h+height,
                       min_w: min_w+width, :] = images

            new_targets[:, min_h: min_h+height,
                        min_w: min_w+width, :] = targets

            return new_images, new_targets

        if len(shape) == 3:
            height, width, z = shape

            if height % divide_factor != 0:
                new_height = (height // divide_factor + 1) * divide_factor
            else:
                new_height = height

            if width % divide_factor != 0:
                new_width = (width // divide_factor + 1) * divide_factor
            else:
                new_width = width

            if z % divide_factor != 0:
                new_z = (z // divide_factor + 1) * divide_factor
            else:
                new_z = z

            if self.mode == 'edge':
                pass
            else:  # default - pad with zeros
                new_images = np.zeros(
                    (image_shape[0],
                     new_height, new_width, new_z,
                     image_shape[-1]))
                new_targets = np.zeros(
                    (target_shape[0],
                     new_height, new_width, new_z,
                     target_shape[-1]))

            min_h = (new_height - height) // 2
            min_w = (new_width - width) // 2
            min_z = (new_z - z) // 2

            new_images[:, min_h: min_h+height,
                       min_w: min_w+width, min_z:min_z+z, :] = images

            new_targets[:, min_h: min_h+height,
                        min_w: min_w+width, min_z:min_z+z, :] = targets

            return new_images, new_targets

        raise RuntimeError('Does not support 4D tensors')


class ImageAffineTransformPreprocessor(BasePreprocessor):
    def __init__(self, rotation_degree=0, rotation_axis=2, zoom_factor=1,
                 shift=None, flip_axis=None,
                 fill_mode='constant', cval=0):

        self.affine_transform = rotation_degree > 0 or \
            zoom_factor != 1 or shift is not None

        self.rotation_degree = rotation_degree
        self.rotation_axis = rotation_axis
        self.zoom_factor = zoom_factor
        self.shift = shift
        self.flip_axis = flip_axis
        self.fill_mode = fill_mode
        self.cval = cval

    def transform(self, images, targets):
        transformed_images = images.copy()
        transformed_targets = targets.copy()

        # loop through
        for i in range(len(images)):
            # apply affine transform if possible
            if self.affine_transform:
                # After affine transform, the pixel intensity may change
                # the image should clip back to original range
                reduced_ax = tuple(
                    range(len(transformed_images[i].shape) - 1))
                vmin = transformed_images[i].min(axis=reduced_ax)
                vmax = transformed_images[i].max(axis=reduced_ax)

                transformed_images[i] = apply_affine_transform(
                    transformed_images[i],
                    mode=self.fill_mode, cval=self.cval,
                    theta=self.rotation_degree,
                    rotation_axis=self.rotation_axis,
                    zoom_factor=self.zoom_factor,
                    shift=self.shift).clip(vmin, vmax)

                transformed_targets[i] = apply_affine_transform(
                    transformed_targets[i],
                    mode=self.fill_mode, cval=self.cval,
                    theta=self.rotation_degree,
                    rotation_axis=self.rotation_axis,
                    zoom_factor=self.zoom_factor,
                    shift=self.shift)

                # round the target label back to integer
                transformed_targets[i] = np.rint(
                    transformed_targets[i])

            # flip image
            if self.flip_axis is not None:
                transformed_images[i] = apply_flip(
                    transformed_images[i], self.flip_axis)

                transformed_targets[i] = apply_flip(
                    transformed_targets[i], self.flip_axis)

        return transformed_images, transformed_targets


class ImageAugmentation2D(BasePreprocessor):
    r"""
        Apply transformation in 2d image (and mask label) for augmentation.
        Check `ImageAugmentation3D` for augmentation on 3d images

        Parameters
        ----------
        rotation_range : int, optional
            range of the angle rotation, in degree, by default 0 (no rotation)
        rotation_axis : int, optional
            the axis of one image to apply rotation,
            by default 0
        rotation_chance : float, optional
            probability to apply rotation transformation to an image,
            by default 0.2
        zoom_range : float, list, tuple optional
            the range of zooming, zooming in when the number is less than 1,
            and zoom out when the number if larger than 1.
            If a `float`, then it is the range between that number and 1,
            by default 1 (no zooming)
        zoom_chance : float, optional
            probability to apply zoom transformation to an image,
            by default 0.2
        shift_range : tuple or list, optional
            the range of translation in each axis, by default None (no shifts)
        shift_chance : float, optional
            probability to apply translation transformation to an image,
            by default 0.1
        flip_axis : int, tuple, list, optional
            flip by one or more axis (in the single image) with a probability
            of 0.5, by default None (no flipping)
        brightness_range : int, tuple, list, optional
            range of the brightness portion,
            based on the max intensity value of each channel.
            For example, when the max intensity value of one channel is 1.0,
            and the brightness is chaned by 1.2, then every pixel in that
            channel will increase the intensity value by 0.2.

            .. math:: 0.2 = 1.0 \cdot (1.2 - 1)

            By default 1 (no changes in brightness)
        brightness_channel : int, tuple, list, optional
            the channel(s) to apply changes in brightness,
            by default None (apply to all channels)
        brightness_chance : float, optional
            probability to apply brightness change transform to an image,
            by default 0.1
        contrast_range : int, tuple, list, optional
            range of the contrast portion,
            (the history range is scaled up or down).
            By default 1 (no changes in contrast)
        contrast_channel : int, tuple, list, optional
            the channel(s) to apply changes in contrast,
            by default None (apply to all channels)
        contrast_chance : float, optional
            probability to apply contrast change transform to an image,
            by default 0.1
        noise_variance : int, tuple, list, optional
            range of the noise variance
            when adding Gaussian noise to the image,
            by default 0 (no adding noise)
        noise_channel : int, tuple, list, optional
            the channel(s) to apply Gaussian noise,
            by default None (apply to all channels)
        noise_chance : float, optional
            probability to apply gaussian noise to an image,
            by default 0.1
        blur_range : int, tuple, list, optional
            range of the blur sigma
            when applying the Gaussian filter to the image,
            by default 0 (no blur)
        blur_channel :int, tuple, list, optional
            the channel(s) to apply Gaussian blur,
            by default None (apply to all channels)
        blur_chance : float, optional
            probability to apply gaussian blur to an image,
            by default 0.1
        fill_mode : str, optional
            the fill mode in affine transformation
            (rotation, zooming, shifting / translation),
            one of {'reflect', 'constant', 'nearest', 'mirror', 'wrap'},
            by default 'constant'
        cval : int, optional
            When rotation, or zooming, or shifting is applied to the image,
            `cval` is the value to fill past edges of input
            if `fill_mode` is 'constant'.
            By default 0
        """

    RANK = 3

    def __init__(self, rotation_range=0, rotation_axis=0, rotation_chance=0.2,
                 zoom_range=1, zoom_chance=0.2,
                 shift_range=None, shift_chance=0.1,
                 flip_axis=None,
                 brightness_range=1, brightness_channel=None,
                 brightness_chance=0.1,
                 contrast_range=1, contrast_channel=None,
                 contrast_chance=0.1,
                 noise_variance=0, noise_channel=None,
                 noise_chance=0.1,
                 blur_range=0, blur_channel=None, blur_chance=0.1,
                 fill_mode='constant', cval=0):
        self.augmentation_obj = ImageAugmentation(
            self.RANK,
            rotation_range, rotation_axis, rotation_chance,
            zoom_range, zoom_chance,
            shift_range, shift_chance,
            flip_axis,
            brightness_range, brightness_channel,
            brightness_chance,
            contrast_range, contrast_channel,
            contrast_chance,
            noise_variance, noise_channel,
            noise_chance,
            blur_range, blur_channel, blur_chance,
            fill_mode, cval
        )

    def transform(self, images, targets):
        """
        Apply augmentation to a batch of images

        Parameters
        ----------
        images : np.array
            the image batch
        targets : np.array, optional
            the target batch, by default None

        Returns
        -------
        np.array
            the transformed images batch (and target)
        """
        return self.augmentation_obj.transform(images, targets)


class ImageAugmentation3D(ImageAugmentation2D):
    RANK = 4


class KerasImagePreprocessorX(BasePreprocessor):
    def __init__(self,
                 shuffle=False,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False, zca_epsilon=1e-06,
                 rotation_range=0, width_shift_range=0.0,
                 height_shift_range=0.0,
                 brightness_range=None, shear_range=0.0, zoom_range=0.0,
                 channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
                 horizontal_flip=False, vertical_flip=False, rescale=None,
                 scale_down=None,
                 preprocessing_function=None, data_format='channels_last',
                 interpolation_order=1, dtype='float32'):

        self.shuffle = shuffle

        if scale_down and (rescale is None):
            rescale = 1 / scale_down

        self.preprocessor = ImageDataGenerator(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening, zca_epsilon=zca_epsilon,
            rotation_range=rotation_range, width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range, shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range, fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format, dtype=dtype)

    def transform(self, x, y):
        return next(self.preprocessor.flow(x, batch_size=x.shape[0],
                                           shuffle=self.shuffle)), y


class KerasImagePreprocessorY(BasePreprocessor):
    def __init__(self,
                 shuffle=True,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False, zca_epsilon=1e-06,
                 rotation_range=0, width_shift_range=0.0,
                 height_shift_range=0.0,
                 brightness_range=None, shear_range=0.0, zoom_range=0.0,
                 channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
                 horizontal_flip=False, vertical_flip=False, rescale=None,
                 scale_down=None,
                 preprocessing_function=None, data_format='channels_last',
                 dtype='float32'):
        self.shuffle = shuffle

        if scale_down and (rescale is None):
            rescale = 1 / scale_down

        self.preprocessor = ImageDataGenerator(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening, zca_epsilon=zca_epsilon,
            rotation_range=rotation_range, width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range, shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range, fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            dtype=dtype)

    def transform(self, x, y):
        return x, next(self.preprocessor.flow(y, batch_size=x.shape[0],
                                              shuffle=self.shuffle))


class Preprocessors(metaclass=Singleton):
    """
    A singleton that contains all the registered customized preprocessors
    """

    def __init__(self):
        self._preprocessors = {
            'WindowingPreprocessor': WindowingPreprocessor,
            'HounsfieldWindowingPreprocessor': HounsfieldWindowingPreprocessor,
            'ImageNormalizerPreprocessor': ImageNormalizerPreprocessor,
            'UnetPaddingPreprocessor': UnetPaddingPreprocessor,
            'ChannelSelector': ChannelSelector,
            'ChannelRemoval': ChannelRemoval,
            'ImageAffineTransformPreprocessor':
                ImageAffineTransformPreprocessor,
            'ImageAugmentation2D': ImageAugmentation2D,
            'ImageAugmentation3D': ImageAugmentation3D,
            'SingleChannelPreprocessor': SingleChannelPreprocessor,
            'KerasImagePreprocessorX': KerasImagePreprocessorX,
            'KerasImagePreprocessorY': KerasImagePreprocessorY
        }

    def register(self, key, preprocessor):
        if not issubclass(preprocessor, BasePreprocessor):
            raise ValueError(
                "The customized preprocessor has to be a subclass"
                + " of deoxys.data.BasePreprocessor"
            )

        if key in self._preprocessors:
            raise KeyError(
                "Duplicated key, please use another key for this preprocessor"
            )
        else:
            self._preprocessors[key] = preprocessor

    def unregister(self, key):
        if key in self._preprocessors:
            del self._preprocessors[key]

    @property
    def preprocessors(self):
        return self._preprocessors


def register_preprocessor(key, preprocessor):
    """
    Register the customized preprocessor.
    If the key name is already registered, it will raise a KeyError exception

    Parameters
    ----------
    key : str
        The unique key-name of the preprocessor
    preprocessor : deoxys.data.BasePreprocessor
        The customized preprocessor class
    """
    Preprocessors().register(key, preprocessor)


def unregister_preprocessor(key):
    """
    Remove the registered preprocessor with the key-name

    Parameters
    ----------
    key : str
        The key-name of the preprocessor to be removed
    """
    Preprocessors().unregister(key)


def _deserialize(config, custom_objects={}):
    predefined_obj = {
        'DummyPreprocessor': DummyPreprocessor
    }

    predefined_obj.update(custom_objects)

    return predefined_obj[config['class_name']](**config['config'])


def preprocessor_from_config(config):
    if 'class_name' not in config:
        raise ValueError('class_name is needed to define preprocessor')

    if 'config' not in config:
        # auto add empty config for preprocessor with only class_name
        config['config'] = {}
    return _deserialize(config, custom_objects=Preprocessors().preprocessors)
