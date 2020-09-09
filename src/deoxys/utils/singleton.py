# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


class Singleton(type):
    """
    A meta class to create a singleton in this project.

    Example:
    ```
    class SingletonClassName(metaclass=Singleton):
        # attributes and method here
    ```
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )

        return cls._instances[cls]
