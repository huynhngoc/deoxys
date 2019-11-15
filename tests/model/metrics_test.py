# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
from tensorflow.keras.metrics import Metric
from deoxys.model.metrics import Metrics
from deoxys.customize import register_metric, \
    unregister_metric, custom_metric
from deoxys.utils import Singleton


@pytest.fixture(autouse=True)
def clear_singleton():
    Singleton._instances = {}  # clear singleton


@pytest.fixture
def metric_class():
    class TestMetric(Metric):
        pass

    yield TestMetric


def test_is_singleton():
    metrics_instance = Metrics()
    another_instance = Metrics()

    assert metrics_instance is another_instance


def test_register_random_obj():
    with pytest.raises(ValueError):
        register_metric("TestMetric", object)


def test_register_metric_success(metric_class):
    register_metric("TestMetric", metric_class)

    assert Metrics()._metrics["TestMetric"] is metric_class


def test_register_duplicate_metric(metric_class):
    register_metric("TestMetric", metric_class)

    with pytest.raises(KeyError):
        register_metric("TestMetric", metric_class)


def test_unregister_metric(metric_class):
    register_metric("TestMetric", metric_class)

    assert Metrics()._metrics["TestMetric"] is metric_class
    unregister_metric("TestMetric")

    assert "TestMetric" not in Metrics()._metrics


def test_decorator():
    @custom_metric
    class TestMetric2(Metric):
        pass

    assert Metrics()._metrics["TestMetric2"] is TestMetric2
