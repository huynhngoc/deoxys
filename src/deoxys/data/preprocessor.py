# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


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
