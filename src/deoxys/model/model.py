# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.models import model_from_config, model_from_json
from tensorflow.keras.models import load_model, save_model


class Model:
    """
    Model
    """

    def __init__(self):
        pass

    def from_config(self, json):
        """
        Create model from JSON configuration

        :param json: [description]
        :type json: [type]
        """
        pass

    def load(self, filename):
        """
        Load model from file

        :param filename: path to the file
        :type filename: str
        """
        pass

    def save(self, filename):
        """
        Save model to file

        :param filename: name of the file
        :type filename: str
        """
        pass

    def fit(self, *args, **kwargs):
        """
        Train model
        """
        pass

    def predict(self):
        """
        Predict from model
        """
        pass
