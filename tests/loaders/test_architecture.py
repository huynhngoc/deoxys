# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pytest
from deoxys.keras.models import Model
from deoxys.keras.layers import Input, Dense, Flatten, Dropout, \
    BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from deoxys.loaders.architecture import BaseModelLoader, ModelLoaderFactory, \
    SequentialModelLoader, UnetModelLoader, DenseModelLoader, \
    load_architecture
from deoxys.customize import register_architecture, custom_architecture
from deoxys.utils import read_file, load_json_config


@pytest.fixture(autouse=True)
def clear_singleton():
    ModelLoaderFactory._loaders = {
        'Sequential': SequentialModelLoader,
        'Unet': UnetModelLoader,
        'Dense': DenseModelLoader
    }  # clear singleton


@pytest.fixture
def modelloader_class():
    class TestModelLoader(BaseModelLoader):
        pass

    yield TestModelLoader


def check_same_models(actual_model, expected_model):
    actual_config = actual_model.get_config()
    expected_config = expected_model.get_config()
    name_map = [(actual_model.name, expected_model.name)]
    for (actual_layer, expected_layer) in zip(
            actual_model.layers, expected_model.layers):
        name_map.append((actual_layer.name, expected_layer.name))

    actual_config_str = str(actual_config)

    for actual_name, expected_name in name_map:
        actual_config_str = actual_config_str.replace(
            "'" + actual_name + "'", "'" + expected_name + "'")

    return actual_config_str == str(expected_config)


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_architecture("TestModelLoader", object)


def test_register_modelloader_success(modelloader_class):
    register_architecture("TestModelLoader", modelloader_class)

    assert ModelLoaderFactory(
    )._loaders["TestModelLoader"] is modelloader_class


def test_register_duplicate_modelloader(modelloader_class):
    register_architecture("TestModelLoader", modelloader_class)

    with pytest.raises(KeyError):
        register_architecture("TestModelLoader", modelloader_class)


def test_decorator():
    @custom_architecture
    class TestModelLoader2(BaseModelLoader):
        pass

    assert ModelLoaderFactory._loaders["TestModelLoader2"] is TestModelLoader2


def test_load_sequential_model():
    architecture = load_json_config(
        read_file('tests/json/sequential_architecture.json'))

    input_params = {'shape': [32, 32]}

    model = load_architecture(architecture, input_params)

    input_layer = Input(shape=(32, 32))
    flatten = Flatten()(input_layer)
    dense = Dense(units=128, activation='relu')(flatten)
    dropout = Dropout(rate=0.2)(dense)
    output_layer = Dense(units=10, activation='softmax')(dropout)

    expected_model = Model(inputs=input_layer, outputs=output_layer)

    assert check_same_models(model, expected_model)


def test_load_unet_model():
    architecture = load_json_config(
        read_file('tests/json/unet_architecture.json')
    )

    input_params = {'shape': [128, 128, 3]}

    model = load_architecture(architecture, input_params)

    def conv_layers(filters, pre_layer):
        conv = BatchNormalization()(
            Conv2D(filters,
                   kernel_size=3,
                   activation='relu',
                   padding='same')(pre_layer))

        return BatchNormalization()(
            Conv2D(filters,
                   kernel_size=3,
                   activation='relu',
                   padding='same')(conv))

    input_layer = Input(shape=[128, 128, 3])

    conv_1 = conv_layers(4, input_layer)
    max_pool_1 = MaxPooling2D()(conv_1)

    conv_2 = conv_layers(8, max_pool_1)
    max_pool_2 = MaxPooling2D()(conv_2)

    conv_3 = conv_layers(16, max_pool_2)
    max_pool_3 = MaxPooling2D()(conv_3)

    conv_4 = conv_layers(32, max_pool_3)
    max_pool_4 = MaxPooling2D()(conv_4)

    conv_5 = conv_layers(64, max_pool_4)
    max_pool_5 = MaxPooling2D()(conv_5)

    conv_t_kwargs = {"kernel_size": 3,
                     "strides": [
                         2,
                         2
                     ],
                     "padding": "same"}

    conv_6 = conv_layers(128, max_pool_5)
    conv_trans_1 = Conv2DTranspose(32, **conv_t_kwargs)(conv_6)

    upconv_1 = conv_layers(64, concatenate([conv_trans_1, conv_5]))
    conv_trans_2 = Conv2DTranspose(16, **conv_t_kwargs)(upconv_1)

    upconv_2 = conv_layers(32, concatenate([conv_trans_2, conv_4]))
    conv_trans_3 = Conv2DTranspose(8, **conv_t_kwargs)(upconv_2)

    upconv_3 = conv_layers(16, concatenate([conv_trans_3, conv_3]))
    conv_trans_4 = Conv2DTranspose(4, **conv_t_kwargs)(upconv_3)

    upconv_4 = conv_layers(8, concatenate([conv_trans_4, conv_2]))
    conv_trans_5 = Conv2DTranspose(2, **conv_t_kwargs)(upconv_4)

    upconv_5 = conv_layers(4, concatenate([conv_trans_5, conv_1]))
    output = Conv2D(1, kernel_size=1, activation='sigmoid')(upconv_5)

    expected_model = Model(inputs=input_layer, outputs=output)

    assert check_same_models(model, expected_model)
