# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pytest
import os
import numpy as np
from deoxys.keras.models import Model
from deoxys.keras.layers import Input, Dense, Flatten, Dropout, \
    BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, \
    Conv3D, Conv3DTranspose, MaxPooling3D
from deoxys.loaders.architecture import load_architecture
from deoxys.automation import generate_unet_architecture, \
    generate_vnet_architecture, generate_unet_architecture_json, \
    generate_vnet_architecture_json, generate_densenet_2d_architecture, \
    generate_densenet_3d_architecture, generate_densenet_2d_json, \
    generate_densenet_3d_json, generate_resnet_architecture, \
    generate_voxresnet_architecture, generate_voxresnet_json, \
    generate_resnet_json
from deoxys.utils import read_file, load_json_config


FILE_NAME = 'test_architecture.json'


@pytest.fixture(autouse=True)
def clear_files():
    yield
    if os.path.isfile(FILE_NAME):
        os.remove(FILE_NAME)


def check_same_models(actual_model, expected_model):
    actual_config = actual_model.get_config()
    expected_config = expected_model.get_config()
    name_map = [(actual_model.name, expected_model.name)]
    for (actual_layer, expected_layer) in zip(
            actual_model.layers, expected_model.layers):
        name_map.append((actual_layer.name, expected_layer.name))

    actual_config_str = str(actual_config)

    # lock the activation function in the conv layer
    actual_config_str = actual_config_str.replace(
        "'activation':", "'activation_fn':")

    for actual_name, expected_name in name_map:
        actual_config_str = actual_config_str.replace(
            "'" + actual_name + "'", "'" + expected_name + "'")

    # release back to original state
    actual_config_str = actual_config_str.replace(
        "'activation_fn':", "'activation':")

    return actual_config_str == str(expected_config)


def test_generate_unet_model():
    architecture = generate_unet_architecture(
        n_upsampling=5, n_filter=4, stride=2)

    input_params = {'shape': [128, 128, 3]}

    model = load_architecture(architecture, input_params)
    model.summary()

    def conv_layers(filters, pre_layer):
        conv = BatchNormalization()(
            Conv2D(filters,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pre_layer))

        return BatchNormalization()(
            Conv2D(filters,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv))

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
                     "strides": 2,
                     "padding": "same",
                     "kernel_initializer": "he_normal"}

    conv_6 = conv_layers(128, max_pool_5)
    conv_trans_1 = Conv2DTranspose(64, **conv_t_kwargs)(conv_6)

    upconv_1 = conv_layers(64, concatenate([conv_5, conv_trans_1]))
    conv_trans_2 = Conv2DTranspose(32, **conv_t_kwargs)(upconv_1)

    upconv_2 = conv_layers(32, concatenate([conv_4, conv_trans_2]))
    conv_trans_3 = Conv2DTranspose(16, **conv_t_kwargs)(upconv_2)

    upconv_3 = conv_layers(16, concatenate([conv_3, conv_trans_3]))
    conv_trans_4 = Conv2DTranspose(8, **conv_t_kwargs)(upconv_3)

    upconv_4 = conv_layers(8, concatenate([conv_2, conv_trans_4]))
    conv_trans_5 = Conv2DTranspose(4, **conv_t_kwargs)(upconv_4)

    upconv_5 = conv_layers(4, concatenate([conv_1, conv_trans_5]))
    output = Conv2D(1, kernel_size=3, activation='sigmoid', padding="same",
                    kernel_initializer="he_normal")(upconv_5)

    expected_model = Model(inputs=input_layer, outputs=output)
    expected_model.summary()

    assert check_same_models(model, expected_model)


def test_generate_unet_model_resize():
    architecture = generate_unet_architecture(
        n_upsampling=5, n_filter=4, stride=2)

    input_params = {'shape': [129, 129, 3]}

    model = load_architecture(architecture, input_params)

    assert np.all(model.input_shape == (None, 129, 129, 3))
    assert np.all(model.output_shape == (None, 129, 129, 1))


def test_create_unet_json():
    unet = generate_unet_architecture()
    generate_unet_architecture_json(FILE_NAME)
    actual = load_json_config(FILE_NAME)

    assert np.all(unet == actual)


def test_generate_vnet_model():
    architecture = generate_vnet_architecture(
        n_upsampling=5, n_filter=4, stride=2)

    input_params = {'shape': [128, 128, 128, 3]}

    model = load_architecture(architecture, input_params)

    def conv_layers(filters, pre_layer):
        conv = BatchNormalization()(
            Conv3D(filters,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pre_layer))

        return BatchNormalization()(
            Conv3D(filters,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv))

    input_layer = Input(shape=[128, 128, 128, 3])

    conv_1 = conv_layers(4, input_layer)
    max_pool_1 = MaxPooling3D()(conv_1)

    conv_2 = conv_layers(8, max_pool_1)
    max_pool_2 = MaxPooling3D()(conv_2)

    conv_3 = conv_layers(16, max_pool_2)
    max_pool_3 = MaxPooling3D()(conv_3)

    conv_4 = conv_layers(32, max_pool_3)
    max_pool_4 = MaxPooling3D()(conv_4)

    conv_5 = conv_layers(64, max_pool_4)
    max_pool_5 = MaxPooling3D()(conv_5)

    conv_t_kwargs = {"kernel_size": 3,
                     "strides": 2,
                     "padding": "same",
                     "kernel_initializer": "he_normal"}

    conv_6 = conv_layers(128, max_pool_5)
    conv_trans_1 = Conv3DTranspose(64, **conv_t_kwargs)(conv_6)

    upconv_1 = conv_layers(64, concatenate([conv_5, conv_trans_1]))
    conv_trans_2 = Conv3DTranspose(32, **conv_t_kwargs)(upconv_1)

    upconv_2 = conv_layers(32, concatenate([conv_4, conv_trans_2]))
    conv_trans_3 = Conv3DTranspose(16, **conv_t_kwargs)(upconv_2)

    upconv_3 = conv_layers(16, concatenate([conv_3, conv_trans_3]))
    conv_trans_4 = Conv3DTranspose(8, **conv_t_kwargs)(upconv_3)

    upconv_4 = conv_layers(8, concatenate([conv_2, conv_trans_4]))
    conv_trans_5 = Conv3DTranspose(4, **conv_t_kwargs)(upconv_4)

    upconv_5 = conv_layers(4, concatenate([conv_1, conv_trans_5]))
    output = Conv3D(1, kernel_size=3, activation='sigmoid', padding="same",
                    kernel_initializer="he_normal")(upconv_5)

    expected_model = Model(inputs=input_layer, outputs=output)

    assert check_same_models(model, expected_model)


def test_generate_vnet_model_resize():
    architecture = generate_vnet_architecture(
        n_upsampling=5, n_filter=4, stride=2)

    input_params = {'shape': [129, 129, 129, 3]}

    model = load_architecture(architecture, input_params)

    assert np.all(model.input_shape == (None, 129, 129, 129, 3))
    assert np.all(model.output_shape == (None, 129, 129, 129, 1))


def test_generate_densenet_model():
    architecture = generate_densenet_2d_architecture(
        n_upsampling=4, n_filter=48, stride=2)

    input_params = {'shape': [128, 128, 3]}

    model = load_architecture(architecture, input_params)
    model.summary()

    expected_model = load_architecture(
        load_json_config('tests/json/densenet_architecture.json'), input_params
    )

    assert check_same_models(model, expected_model)


def test_generate_densenet_model_resize():
    architecture = generate_densenet_2d_architecture(
        n_upsampling=4, n_filter=48, stride=2)

    input_params = {'shape': [129, 129, 3]}

    model = load_architecture(architecture, input_params)
    model.summary()

    assert np.all(model.input_shape == (None, 129, 129, 3))
    assert np.all(model.output_shape == (None, 129, 129, 1))


def test_generate_densenet_model_batchnorm():
    architecture = generate_densenet_2d_architecture(
        n_upsampling=4, n_filter=48, stride=2, batchnorm=True)

    input_params = {'shape': [128, 128, 3]}

    model = load_architecture(architecture, input_params)
    model.summary()


def test_generate_densenet_model_dropout():
    architecture = generate_densenet_2d_architecture(
        n_upsampling=4, n_filter=48, stride=2, dropout_rate=0.1)

    input_params = {'shape': [128, 128, 3]}

    model = load_architecture(architecture, input_params)
    model.summary()


def test_generate_densenet_model_3d():
    architecture = generate_densenet_3d_architecture(
        n_upsampling=3, n_filter=[4, 8, 12, 16], dense_block=[2, 3, 4, 5],
        stride=2)

    input_params = {'shape': [128, 128, 128, 3]}

    model = load_architecture(architecture, input_params)
    model.summary()


def test_create_densenet_json():
    unet = generate_densenet_2d_architecture()
    generate_densenet_2d_json(FILE_NAME)
    actual = load_json_config(FILE_NAME)

    assert np.all(unet == actual)


def test_create_densenet_3d_json():
    unet = generate_densenet_3d_architecture()
    generate_densenet_3d_json(FILE_NAME)
    actual = load_json_config(FILE_NAME)

    assert np.all(unet == actual)


def test_generate_resnet_model():
    architecture = generate_resnet_architecture(
        n_upsampling=3, n_filter=64, stride=2)

    input_params = {'shape': [128, 128, 3]}

    model = load_architecture(architecture, input_params)
    model.summary()

    expected_model = load_architecture(
        load_json_config('tests/json/resnet_architecture.json'), input_params
    )

    assert check_same_models(model, expected_model)


def test_generate_resnet_model_resize():
    architecture = generate_resnet_architecture(
        n_upsampling=3, n_filter=64, stride=2)

    input_params = {'shape': [129, 129, 3]}

    model = load_architecture(architecture, input_params)
    model.summary()

    assert np.all(model.input_shape == (None, 129, 129, 3))
    assert np.all(model.output_shape == (None, 129, 129, 1))


def test_generate_resnet_model_dropout():
    architecture = generate_resnet_architecture(
        n_upsampling=4, n_filter=48, stride=2, dropout_rate=0.1)

    input_params = {'shape': [128, 128, 3]}

    model = load_architecture(architecture, input_params)
    model.summary()


def test_generate_voxresnet_model():
    architecture = generate_voxresnet_architecture(
        n_upsampling=3, n_filter=64, stride=2)

    input_params = {'shape': [128, 128, 128, 3]}

    model = load_architecture(architecture, input_params)
    model.summary()


def test_create_resnet_json():
    unet = generate_resnet_architecture()
    generate_resnet_json(FILE_NAME)
    actual = load_json_config(FILE_NAME)

    assert np.all(unet == actual)


def test_create_voxresnet_json():
    unet = generate_voxresnet_architecture()
    generate_voxresnet_json(FILE_NAME)
    actual = load_json_config(FILE_NAME)

    assert np.all(unet == actual)
