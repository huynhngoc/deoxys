# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


class Model:
    """
    Model
    """

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
