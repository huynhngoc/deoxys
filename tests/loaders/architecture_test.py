# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
from deoxys.loaders.architecture import BaseModelLoader, ModelLoaderFactory, \
    SequentialModelLoader, UnetModelLoader, DenseModelLoader
from deoxys.customize import register_architecture, custom_architecture


@pytest.fixture(autouse=True)
def clear_singleton():
    ModelLoaderFactory._loaders = {
        'Sequential': SequentialModelLoader,
        'Unet': UnetModelLoader,
        'Dense': DenseModelLoader
    }  # clear singleton


@pytest.fixture
def modelloader_class():
    class TestModelLoader(BaseModelLoader):
        pass

    yield TestModelLoader


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_architecture("TestModelLoader", object)


def test_register_modelloader_success(modelloader_class):
    register_architecture("TestModelLoader", modelloader_class)

    assert ModelLoaderFactory(
    )._loaders["TestModelLoader"] is modelloader_class


def test_register_duplicate_modelloader(modelloader_class):
    register_architecture("TestModelLoader", modelloader_class)

    with pytest.raises(KeyError):
        register_architecture("TestModelLoader", modelloader_class)


def test_decorator():
    @custom_architecture
    class TestModelLoader2(BaseModelLoader):
        pass

    assert ModelLoaderFactory._loaders["TestModelLoader2"] is TestModelLoader2
