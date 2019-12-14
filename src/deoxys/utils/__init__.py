# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import copy
import json

from .singleton import Singleton
from .file_utils import *
from .json_utils import *
from .plot_utils import *
from .data_utils import *


def deep_copy(obj):
    if type(obj) == dict or type(obj) == list:
        try:
            return json.loads(json.dumps(obj))
        except json.JSONDecodeError:
            pass

    return copy.deepcopy(obj)
