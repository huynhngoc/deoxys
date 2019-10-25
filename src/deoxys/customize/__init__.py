# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"

from ..model.layers import register_layer, unregister_layer
from ..model.optimizers import register_optimizer, unregister_optimizer
from ..model.losses import register_loss, unregister_loss

from ..model.loaders import register_architecture
