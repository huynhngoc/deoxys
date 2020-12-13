
# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import json


def downsampling_block(idx=0, conv_layer='Conv2D', n_filter=64,
                       batchnorm=True, activation='relu',
                       dropout_rate=0,
                       kernel=3, maxpool='MaxPooling2D'):
    block = []

    conv_block = {
        'class_name': conv_layer,
        'config': {
            'filters': n_filter,
            'kernel_size': kernel,
            'activation': activation,
            'kernel_initializer': 'he_normal',
            'padding': 'same'
        }
    }

    if batchnorm:
        conv_block['normalizer'] = {
            'class_name': 'BatchNormalization'
        }

    conv_block_2 = conv_block.copy()
    conv_block_2['name'] = f'conv{idx}'

    block.append(conv_block)
    block.append(conv_block_2)
    block.append({
        'class_name': maxpool
    })

    if dropout_rate > 0:
        block.append({
            'class_name': 'Dropout',
            'config': {
                'rate': dropout_rate
            }
        })

    return block


def bottle_neck(conv_layer='Conv2D', n_filter=64,
                batchnorm=True, activation='relu', kernel=3):
    block = []

    conv_block = {
        'class_name': conv_layer,
        'config': {
            'filters': n_filter,
            'kernel_size': kernel,
            'activation': activation,
            'kernel_initializer': 'he_normal',
            'padding': 'same'
        }
    }

    if batchnorm:
        conv_block['normalizer'] = {
            'class_name': 'BatchNormalization'
        }

    block.append(conv_block)
    block.append(conv_block)

    return block


def upsampling_block(idx=0, conv_layer='Conv2D', n_filter=64,
                     batchnorm=True, activation='relu',
                     dropout_rate=0,
                     kernel=3, stride=1):
    block = []
    upconv_block = {
        'name': f'upconv{idx}',
        'class_name': f'{conv_layer}Transpose',
        'config': {
            'filters': n_filter,
            'kernel_size': kernel,
            'strides': stride,
            'kernel_initializer': 'he_normal',
            'padding': 'same'
        }
    }

    block.append(upconv_block)

    if dropout_rate > 0:
        block.append({
            'class_name': 'Dropout',
            'config': {
                'rate': dropout_rate
            },
            'inputs': [
                f'conv{idx}',
                f'upconv{idx}'
            ]
        })

    conv_block = {
        'class_name': conv_layer,
        'config': {
            'filters': n_filter,
            'kernel_size': kernel,
            'activation': activation,
            'kernel_initializer': 'he_normal',
            'padding': 'same'
        }
    }

    if batchnorm:
        conv_block['normalizer'] = {
            'class_name': 'BatchNormalization'
        }
    conv_block_2 = conv_block.copy()

    if dropout_rate == 0:
        conv_block['inputs'] = [
            f'conv{idx}',
            f'upconv{idx}'
        ]

    block.append(conv_block)
    block.append(conv_block_2)

    return block


def _generate_unet_layers(postfix, n_upsampling=4, n_filter=64,
                          batchnorm=True, activation='relu',
                          dropout_rate=0,
                          kernel=3, stride=1, n_class=2):
    conv_layer = f'Conv{postfix}'
    maxpool = f'MaxPooling{postfix}'
    layers = []

    if '__iter__' not in dir(n_filter):
        n_filter = [n_filter * (2**i) for i in range(n_upsampling + 1)]

    for i in range(n_upsampling):
        layers.extend(downsampling_block(
            idx=i + 1, conv_layer=conv_layer, n_filter=n_filter[i],
            batchnorm=batchnorm,
            activation=activation, dropout_rate=dropout_rate, kernel=kernel,
            maxpool=maxpool))

    layers.extend(bottle_neck(conv_layer=conv_layer,
                              n_filter=n_filter[-1],
                              batchnorm=batchnorm,
                              activation=activation,
                              kernel=kernel))

    for i in range(n_upsampling, 0, -1):
        layers.extend(upsampling_block(idx=i, conv_layer=conv_layer,
                                       n_filter=n_filter[i - 1],
                                       batchnorm=batchnorm,
                                       activation=activation,
                                       dropout_rate=dropout_rate,
                                       kernel=kernel, stride=stride))

    if n_class <= 2:
        layers.append({
            'class_name': conv_layer,
            'config': {
                'filters': 1,
                'kernel_size': kernel,
                'activation': 'sigmoid',
                'kernel_initializer': 'he_normal',
                'padding': 'same'
            }
        })
    else:
        layers.append({
            'class_name': conv_layer,
            'config': {
                'filters': n_class,
                'kernel_size': kernel,
                'activation': 'softmax',
                'kernel_initializer': 'he_normal',
                'padding': 'same'
            }
        })

    return layers


def generate_unet_architecture(n_upsampling=4, n_filter=64,
                               batchnorm=True, activation='relu',
                               dropout_rate=0,
                               kernel=3, stride=1, n_class=2):

    return {
        'type': 'Unet',
        'layers': _generate_unet_layers('2D', n_upsampling, n_filter,
                                        batchnorm, activation,
                                        dropout_rate,
                                        kernel, stride, n_class)
    }


def generate_vnet_architecture(n_upsampling=4, n_filter=64,
                               batchnorm=True, activation='relu',
                               dropout_rate=0,
                               kernel=3, stride=1, n_class=2):

    return {
        'type': 'Vnet',
        'layers': _generate_unet_layers('3D', n_upsampling, n_filter,
                                        batchnorm, activation,
                                        dropout_rate,
                                        kernel, stride, n_class)
    }


def generate_unet_architecture_json(filename, n_upsampling=4, n_filter=64,
                                    batchnorm=True, activation='relu',
                                    dropout_rate=0,
                                    kernel=3, stride=1, n_class=2):

    with open(filename, 'w') as f:
        json.dump(generate_unet_architecture(n_upsampling, n_filter,
                                             batchnorm, activation,
                                             dropout_rate,
                                             kernel, stride, n_class), f)


def generate_vnet_architecture_json(filename, n_upsampling=4, n_filter=64,
                                    batchnorm=True, activation='relu',
                                    dropout_rate=0,
                                    kernel=3, stride=1, n_class=2):

    with open(filename, 'w') as f:
        json.dump(generate_vnet_architecture(n_upsampling, n_filter,
                                             batchnorm, activation,
                                             dropout_rate,
                                             kernel, stride, n_class), f)
