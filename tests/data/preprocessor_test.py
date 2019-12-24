# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
import numpy as np
from deoxys.data import Preprocessors, BasePreprocessor, \
    SingleChannelPreprocessor, WindowingPreprocessor, \
    KerasImagePreprocessorX, KerasImagePreprocessorY, \
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
