# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pytest
import numpy as np
from deoxys.data import Preprocessors, BasePreprocessor, \
    SingleChannelPreprocessor, WindowingPreprocessor, \
    KerasImagePreprocessorX, KerasImagePreprocessorY, \
    HounsfieldWindowingPreprocessor, ImageNormalizerPreprocessor, \
    preprocessor_from_config
from deoxys.customize import register_preprocessor, \
    unregister_preprocessor, custom_preprocessor
from deoxys.utils import Singleton


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


@pytest.fixture
def preprocessor_class():
    class TestPreprocessor(BasePreprocessor):
        pass

    yield TestPreprocessor


def test_is_singleton():
    preprocessors_instance = Preprocessors()
    another_instance = Preprocessors()

    assert preprocessors_instance is another_instance


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_preprocessor("TestPreprocessor", object)


def test_register_preprocessor_success(preprocessor_class):
    register_preprocessor("TestPreprocessor", preprocessor_class)

    assert Preprocessors(
    )._preprocessors["TestPreprocessor"] is preprocessor_class


def test_register_duplicate_preprocessor(preprocessor_class):
    register_preprocessor("TestPreprocessor", preprocessor_class)

    with pytest.raises(KeyError):
        register_preprocessor("TestPreprocessor", preprocessor_class)


def test_unregister_preprocessor(preprocessor_class):
    register_preprocessor("TestPreprocessor", preprocessor_class)

    assert Preprocessors(
    )._preprocessors["TestPreprocessor"] is preprocessor_class
    unregister_preprocessor("TestPreprocessor")

    assert "TestPreprocessor" not in Preprocessors()._preprocessors


def test_decorator():
    @custom_preprocessor
    class TestPreprocessor2(BasePreprocessor):
        pass

    assert Preprocessors(
    ).preprocessors["TestPreprocessor2"] is TestPreprocessor2


def test_preprocessor_from_config():
    pp = preprocessor_from_config({
        'class_name': 'SingleChannelPreprocessor'
    })

    assert callable(pp.transform)


def test_single_channel_preprocessor():
    scp = SingleChannelPreprocessor()

    x, y = np.arange(100).reshape(4, 5, 5), np.arange(100).reshape(4, 5, 5)
    expected_x = np.arange(100).reshape(4, 5, 5, 1)
    expected_y = np.arange(100).reshape(4, 5, 5, 1)

    output_x, output_y = scp.transform(x, y)
    assert x.shape == (4, 5, 5), 'X remains unchanged'
    assert y.shape == (4, 5, 5), 'Y remains unchanged'
    assert expected_x.shape == output_x.shape
    assert expected_y.shape == output_y.shape
    assert np.all(expected_x == output_x)
    assert np.all(expected_y == output_y)


def test_windowing_preprocessor():
    wp = WindowingPreprocessor(50, 80, 0)

    y = np.arange(200).reshape(4, 5, 5, 2)
    x_first_channel = np.arange(100).reshape(4, 5, 5)
    x_second_channel = np.arange(100).reshape(4, 5, 5)
    x = np.stack([x_first_channel, x_second_channel], axis=-1)

    expected_y = np.arange(200).reshape(4, 5, 5, 2)
    expected_x_first_channel = np.concatenate(
        ([-40] * 10,
         np.arange(-40, 40),
         [40] * 10)
    ).reshape(4, 5, 5)
    expected_x_second_channel = np.arange(100).reshape(4, 5, 5)
    expected_x = np.stack(
        [expected_x_first_channel, expected_x_second_channel], axis=-1)

    ouput_x, ouput_y = wp.transform(x, y)

    assert ouput_x.shape == expected_x.shape
    assert ouput_y.shape == expected_y.shape
    assert np.all(ouput_x == expected_x)
    assert np.all(ouput_y == expected_y)


def test_hounsfield_windowing_preprocessor():
    wp = HounsfieldWindowingPreprocessor(30, 80, 0, 20)

    y = np.arange(200).reshape(4, 5, 5, 2)
    x_first_channel = np.arange(100).reshape(4, 5, 5)
    x_second_channel = np.arange(100).reshape(4, 5, 5)
    x = np.stack([x_first_channel, x_second_channel], axis=-1)

    expected_y = np.arange(200).reshape(4, 5, 5, 2)
    expected_x_first_channel = np.concatenate(
        ([-40] * 10,
         np.arange(-40, 40),
         [40] * 10)
    ).reshape(4, 5, 5)
    expected_x_second_channel = np.arange(100).reshape(4, 5, 5)
    expected_x = np.stack(
        [expected_x_first_channel, expected_x_second_channel], axis=-1)

    ouput_x, ouput_y = wp.transform(x, y)

    assert ouput_x.shape == expected_x.shape
    assert ouput_y.shape == expected_y.shape
    assert np.all(ouput_x == expected_x)
    assert np.all(ouput_y == expected_y)


def test_image_normalizer_preprocessor():
    base_data = np.array([np.arange(30) for _ in range(5)])
    targets = np.array([np.arange(30) for _ in range(5)])

    normalized_image = np.zeros(30)
    normalized_image[20:] = 1
    normalized_image[10:20] = np.arange(10) / 10

    normalize_data = np.array([normalized_image for _ in range(5)])

    images = np.zeros((5, 5, 6, 2))
    images[..., 0] = base_data.reshape(5, 5, 6)
    images[..., 1] = base_data.reshape(5, 5, 6)

    expected_x = np.zeros((5, 5, 6, 2))
    expected_x[..., 0] = normalize_data.reshape(5, 5, 6)
    expected_x[..., 1] = normalize_data.reshape(5, 5, 6)

    expected_y = np.array([np.arange(30) for _ in range(5)])

    output_x, output_y = ImageNormalizerPreprocessor(
        vmin=10, vmax=20).transform(images, targets)

    assert output_x.shape == expected_x.shape
    assert output_y.shape == expected_y.shape
    assert np.all(output_x == expected_x)
    assert np.all(output_y == expected_y)

    output_x, output_y = ImageNormalizerPreprocessor().transform(
        images, targets)

    assert output_x.shape == expected_x.shape
    assert output_y.shape == expected_y.shape
    assert np.allclose(output_x, images/29)
    assert np.all(output_y == expected_y)


def test_image_normalizer_preprocessor_per_channel():
    base_data = np.array([np.arange(30) for _ in range(5)])
    targets = np.array([np.arange(30) for _ in range(5)])

    normalized_image = np.zeros(30)
    normalized_image[20:] = 1
    normalized_image[10:20] = np.arange(10) / 10

    normalize_data = np.array([normalized_image for _ in range(5)])

    images = np.zeros((5, 5, 6, 3))
    images[..., 0] = base_data.reshape(5, 5, 6)
    images[..., 1] = base_data.reshape(5, 5, 6)
    images[..., 2] = base_data.reshape(5, 5, 6)

    expected_x = np.zeros((5, 5, 6, 3))
    expected_x[..., 0] = normalize_data.reshape(5, 5, 6)
    expected_x[..., 1] = normalize_data.reshape(5, 5, 6)
    expected_x[..., 2] = base_data.reshape(5, 5, 6) / 29

    expected_y = np.array([np.arange(30) for _ in range(5)])

    output_x, output_y = ImageNormalizerPreprocessor(
        [10, 10], [20, 20]).transform(images, targets)

    assert output_x.shape == expected_x.shape
    assert output_y.shape == expected_y.shape
    assert np.allclose(output_x, expected_x)
    assert np.all(output_y == expected_y)

    output_x, output_y = ImageNormalizerPreprocessor(
        [10, 20], [20, 30]).transform(images, targets)

    normalized_image = np.zeros(30)
    normalized_image[20:] = np.arange(10) / 10
    normalize_data = np.array([normalized_image for _ in range(5)])
    expected_x[..., 1] = normalize_data.reshape(5, 5, 6)

    assert output_x.shape == expected_x.shape
    assert output_y.shape == expected_y.shape
    assert np.allclose(output_x, expected_x)
    assert np.all(output_y == expected_y)


def test_keras_image_preprocessor():
    kpx = KerasImagePreprocessorX(scale_down=2, shuffle=False)
    kpy = KerasImagePreprocessorY(scale_down=4, shuffle=False)

    x, y = np.arange(100).reshape(
        4, 5, 5, 1), np.arange(100).reshape(4, 5, 5, 1)

    expected_x = np.arange(100).reshape(4, 5, 5, 1) / 2
    expected_y = np.arange(100).reshape(4, 5, 5, 1) / 4

    output_x, output_y = kpx.transform(x, y)
    output_x, output_y = kpy.transform(output_x, output_y)

    assert expected_x.shape == output_x.shape
    assert expected_y.shape == output_y.shape
    assert np.allclose(expected_x, output_x)
    assert np.all(expected_y == output_y)
