
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

##########################################################################
# DenseNet
##########################################################################


def downsampling_dense_block(idx=0, conv_layer='Conv2D', n_filter=48,
                             batchnorm=False, activation='relu',
                             dropout_rate=0,
                             kernel=3, dense_block=3, strides=1):
    # downconv (BN + relu) --> (dropout) --> dense
    block = []

    # conv act as downsampling layer, except for first layer, with kernel=3
    conv_block = {
        'name': f'down_conv{idx}',
        'class_name': conv_layer,
        'config': {
            'filters': n_filter,
            'kernel_size': kernel if idx <= 1 else 1,
            'activation': activation,
            'kernel_initializer': 'he_normal',
            'padding': 'same',
            'strides': strides
        }
    }

    if batchnorm:
        conv_block['normalizer'] = {
            'class_name': 'BatchNormalization'
        }

    block.append(conv_block)

    if dropout_rate > 0:
        block.append({
            'class_name': 'Dropout',
            'config': {
                'rate': dropout_rate
            }
        })

    if idx > 1:
        block[-1]['inputs'] = [
            f'down_conv{idx-1}',
            f'dense_block{idx-1}'
        ]

    dense_block_dict = {
        'dense_block': dense_block,
        'name': f'dense_block{idx}',
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
        dense_block_dict['normalizer'] = {
            'class_name': 'BatchNormalization'
        }

    block.append(dense_block_dict)

    return block


def upsampling_dense_block(idx=0, conv_layer='Conv2D', n_filter=96,
                           batchnorm=False, activation='relu',
                           dropout_rate=0,
                           kernel=3, dense_block=3, strides=2):
    # upconv (relu + batchnorm) --> concat --> (dropout) --> dense_block
    block = []

    conv_block = {
        'name': f'up_conv{idx}',
        'class_name': f'{conv_layer}Transpose',
        'config': {
            'filters': n_filter,
            'kernel_size': 1,
            'activation': activation,
            'kernel_initializer': 'he_normal',
            'padding': 'same',
            'strides': strides
        }
    }

    if batchnorm:
        conv_block['normalizer'] = {
            'class_name': 'BatchNormalization'
        }

    block.append(conv_block)

    if dropout_rate > 0:
        block.append({
            'class_name': 'Dropout',
            'config': {
                'rate': dropout_rate
            },
            'inputs': [
                f'down_conv{idx}',
                f'dense_block{idx}',
                f'up_conv{idx}'
            ]
        })

    dense_block_dict = {
        'dense_block': dense_block,
        # 'name': f'dense_block{idx}',
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
        dense_block_dict['normalizer'] = {
            'class_name': 'BatchNormalization'
        }

    if dropout_rate == 0:
        dense_block_dict['inputs'] = [
            f'down_conv{idx}',
            f'dense_block{idx}',
            f'up_conv{idx}'
        ]

    block.append(dense_block_dict)

    return block


def _generate_densenet_layers(postfix, n_upsampling=4, n_filter=48,
                              dense_block=3,
                              batchnorm=False, activation='relu',
                              dropout_rate=0,
                              kernel=3, stride=2, n_class=2):
    conv_layer = f'Conv{postfix}'
    layers = []

    if '__iter__' not in dir(n_filter):
        n_filter = [n_filter + 16 * i for i in range(n_upsampling + 1)]

    if '__iter__' not in dir(dense_block):
        dense_block = [dense_block + i for i in range(n_upsampling + 1)]

    for i in range(n_upsampling + 1):
        layers.extend(downsampling_dense_block(
            idx=i + 1, conv_layer=conv_layer, n_filter=n_filter[i],
            batchnorm=batchnorm, dense_block=dense_block[i],
            activation=activation, dropout_rate=dropout_rate, kernel=kernel,
            strides=stride if i else 1))

    for i in range(n_upsampling, 0, -1):
        layers.extend(upsampling_dense_block(idx=i, conv_layer=conv_layer,
                                             n_filter=n_filter[i - 1],
                                             dense_block=dense_block[i-1],
                                             batchnorm=batchnorm,
                                             activation=activation,
                                             dropout_rate=dropout_rate,
                                             kernel=kernel, strides=stride))

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


def generate_densenet_2d_architecture(n_upsampling=4, n_filter=48,
                                      dense_block=3,
                                      batchnorm=False, activation='relu',
                                      dropout_rate=0,
                                      kernel=3, stride=2, n_class=2):

    return {
        'type': 'DenseNet',
        'layers': _generate_densenet_layers(
            '2D', n_upsampling=n_upsampling, n_filter=n_filter,
            dense_block=dense_block,
            batchnorm=batchnorm, activation=activation,
            dropout_rate=dropout_rate,
            kernel=kernel, stride=stride, n_class=n_class)
    }


def generate_densenet_3d_architecture(n_upsampling=4, n_filter=48,
                                      dense_block=3,
                                      batchnorm=False, activation='relu',
                                      dropout_rate=0,
                                      kernel=3, stride=2, n_class=2):

    return {
        'type': 'DenseNet',
        'layers': _generate_densenet_layers(
            '3D', n_upsampling=n_upsampling, n_filter=n_filter,
            dense_block=dense_block,
            batchnorm=batchnorm, activation=activation,
            dropout_rate=dropout_rate,
            kernel=kernel, stride=stride, n_class=n_class)
    }


def generate_densenet_2d_json(filename, n_upsampling=4, n_filter=48,
                              dense_block=3,
                              batchnorm=False, activation='relu',
                              dropout_rate=0,
                              kernel=3, stride=2, n_class=2):

    with open(filename, 'w') as f:
        json.dump(generate_densenet_2d_architecture(
            n_upsampling=n_upsampling, n_filter=n_filter,
            dense_block=dense_block,
            batchnorm=batchnorm, activation=activation,
            dropout_rate=dropout_rate,
            kernel=kernel, stride=stride, n_class=n_class), f)


def generate_densenet_3d_json(filename, n_upsampling=4, n_filter=48,
                              dense_block=3,
                              batchnorm=False, activation='relu',
                              dropout_rate=0,
                              kernel=3, stride=2, n_class=2):

    with open(filename, 'w') as f:
        json.dump(generate_densenet_3d_architecture(
            n_upsampling=n_upsampling, n_filter=n_filter,
            dense_block=dense_block,
            batchnorm=batchnorm, activation=activation,
            dropout_rate=dropout_rate,
            kernel=kernel, stride=stride, n_class=n_class), f)


##########################################################################
# ResNet
##########################################################################
def cnn_start_res_block(conv_layer='Conv2D', n_filter=64,
                        batchnorm=True, activation='relu',
                        kernel=3):
    # 2 conv
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


def downsampling_res_block(idx=0, conv_layer='Conv2D', res_block=2,
                           n_filter=64,
                           batchnorm=True, activation='relu',
                           dropout_rate=0,
                           kernel=3, maxpool='MaxPooling2D'):
    # maxpool --> (dropout) --> res_block x 2
    block = [
        {
            'name': f'pool{idx}',
            'class_name': maxpool
        }
    ]

    if dropout_rate > 0:
        block.append({
            'class_name': 'Dropout',
            'config': {
                'rate': dropout_rate
            }
        })

    res_block_dict = {
        'res_block': res_block,
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
        res_block_dict['normalizer'] = {
            'class_name': 'BatchNormalization'
        }

    block.append(res_block_dict)
    block.append(res_block_dict)

    return block


def upsampling_res_block(idx=0, conv_layer='Conv2D', n_filter=64,
                         batchnorm=True, activation='relu',
                         dropout_rate=0, stride=2,
                         kernel=3):
    # upconv --> concat --> (dropout) --> conv --> conv
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
                f'pool{idx}',
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
            f'pool{idx}',
            f'upconv{idx}'
        ]

    block.append(conv_block)
    block.append(conv_block_2)

    return block


def final_cnn_res_block(conv_layer='Conv2D', n_filter=64,
                        batchnorm=True, activation='relu',
                        dropout_rate=0, stride=2,
                        kernel=3):
    # upconv --> (dropout) --> conv --> conv
    block = []
    upconv_block = {
        # 'name': f'upconv{idx}',
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
            }
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

    conv_block2 = conv_block.copy()
    conv_block['resize_inputs'] = True

    block.append(conv_block)
    block.append(conv_block2)

    return block


def _generate_resnet_layers(postfix, n_upsampling=3, n_filter=64,
                            res_block=2,
                            batchnorm=True, activation='relu',
                            dropout_rate=0,
                            kernel=3, stride=2, n_class=2):

    conv_layer = f'Conv{postfix}'
    maxpool = f'MaxPooling{postfix}'
    layers = []

    layers.extend(cnn_start_res_block(
        conv_layer=conv_layer, n_filter=n_filter, batchnorm=batchnorm,
        activation=activation, kernel=kernel))

    for i in range(n_upsampling):
        layers.extend(downsampling_res_block(
            idx=i + 1, conv_layer=conv_layer,
            n_filter=n_filter, res_block=res_block,
            batchnorm=batchnorm,
            activation=activation, dropout_rate=dropout_rate, kernel=kernel,
            maxpool=maxpool))

    for i in range(n_upsampling - 1, 0, -1):
        layers.extend(upsampling_res_block(idx=i, conv_layer=conv_layer,
                                           n_filter=n_filter,
                                           batchnorm=batchnorm,
                                           activation=activation,
                                           dropout_rate=dropout_rate,
                                           kernel=kernel, stride=stride))

    layers.extend(final_cnn_res_block(
        conv_layer=conv_layer, n_filter=n_filter, batchnorm=batchnorm,
        activation=activation, dropout_rate=dropout_rate, stride=stride,
        kernel=kernel
    ))

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


def generate_resnet_architecture(n_upsampling=3, n_filter=64,
                                 res_block=2,
                                 batchnorm=True, activation='relu',
                                 dropout_rate=0,
                                 kernel=3, stride=2, n_class=2):

    return {
        'type': 'ResNet',
        'layers': _generate_resnet_layers(
            '2D', n_upsampling=n_upsampling, n_filter=n_filter,
            res_block=res_block,
            batchnorm=batchnorm, activation=activation,
            dropout_rate=dropout_rate,
            kernel=kernel, stride=stride, n_class=n_class)
    }


def generate_voxresnet_architecture(n_upsampling=3, n_filter=64,
                                    res_block=2,
                                    batchnorm=True, activation='relu',
                                    dropout_rate=0,
                                    kernel=3, stride=2, n_class=2):

    return {
        'type': 'VoxResNet',
        'layers': _generate_resnet_layers(
            '3D', n_upsampling=n_upsampling, n_filter=n_filter,
            res_block=res_block,
            batchnorm=batchnorm, activation=activation,
            dropout_rate=dropout_rate,
            kernel=kernel, stride=stride, n_class=n_class)
    }


def generate_resnet_json(filename, n_upsampling=3, n_filter=64,
                         res_block=2,
                         batchnorm=True, activation='relu',
                         dropout_rate=0,
                         kernel=3, stride=2, n_class=2):

    with open(filename, 'w') as f:
        json.dump(generate_resnet_architecture(
            n_upsampling=n_upsampling, n_filter=n_filter,
            res_block=res_block,
            batchnorm=batchnorm, activation=activation,
            dropout_rate=dropout_rate,
            kernel=kernel, stride=stride, n_class=n_class), f)


def generate_voxresnet_json(filename, n_upsampling=3, n_filter=64,
                            res_block=2,
                            batchnorm=True, activation='relu',
                            dropout_rate=0,
                            kernel=3, stride=2, n_class=2):

    with open(filename, 'w') as f:
        json.dump(generate_voxresnet_architecture(
            n_upsampling=n_upsampling, n_filter=n_filter,
            res_block=res_block,
            batchnorm=batchnorm, activation=activation,
            dropout_rate=dropout_rate,
            kernel=kernel, stride=stride, n_class=n_class), f)
