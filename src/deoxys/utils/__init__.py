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


KERAS_STANDALONE = None
TENSORFLOW_EAGER_MODE = None


def deep_copy(obj):
    if type(obj) == dict or type(obj) == list:
        try:
            return json.loads(json.dumps(obj))
        except json.JSONDecodeError:
            pass

    return copy.deepcopy(obj)


def is_keras_standalone():
    global KERAS_STANDALONE
    if KERAS_STANDALONE is None:
        KERAS_STANDALONE = False

        if 'KERAS_MODE' in os.environ:
            mode = os.environ.get('KERAS_MODE')
            if mode.upper() == 'ALONE':
                KERAS_STANDALONE = True
            elif mode.upper() == 'TENSORFLOW':
                KERAS_STANDALONE = False
    return KERAS_STANDALONE


def is_default_tf_eager_mode():
    global TENSORFLOW_EAGER_MODE
    if TENSORFLOW_EAGER_MODE is None:
        TENSORFLOW_EAGER_MODE = (
            not is_keras_standalone()) and tf.__version__.startswith('2.')

    return TENSORFLOW_EAGER_MODE
