# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import copy
import json
import os
import tensorflow as tf

from .singleton import Singleton
from .file_utils import *
from .json_utils import *
from .data_utils import *


# KERAS_STANDALONE = None
# TENSORFLOW_EAGER_MODE = None
ITER_PER_EPOCH = None


def deep_copy(obj):
    if type(obj) == dict or type(obj) == list:
        try:
            return json.loads(json.dumps(obj))
        except json.JSONDecodeError:
            pass

    return copy.deepcopy(obj)


# def is_keras_standalone():
#     global KERAS_STANDALONE
#     if KERAS_STANDALONE is None:
#         KERAS_STANDALONE = False

#         if 'KERAS_MODE' in os.environ:
#             mode = os.environ.get('KERAS_MODE')
#             if mode.upper() == 'ALONE':
#                 KERAS_STANDALONE = True
#             elif mode.upper() == 'TENSORFLOW':
#                 KERAS_STANDALONE = False
#     return KERAS_STANDALONE


# def is_default_tf_eager_mode():
#     global TENSORFLOW_EAGER_MODE
#     if TENSORFLOW_EAGER_MODE is None:
#         TENSORFLOW_EAGER_MODE = (
#             not is_keras_standalone()) and tf.__version__.startswith('2.')

#     return TENSORFLOW_EAGER_MODE


def number_of_iteration():
    global ITER_PER_EPOCH
    if ITER_PER_EPOCH is None:
        ITER_PER_EPOCH = 0

        if 'ITER_PER_EPOCH' in os.environ:
            iter_num = os.environ.get('ITER_PER_EPOCH')
            try:
                ITER_PER_EPOCH = int(iter_num)
            except ValueError:
                print('Invalid number ITER_PER_EPOCH.'
                      'Using actual number of batches')
    return ITER_PER_EPOCH
