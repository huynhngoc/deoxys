# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


import pytest
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from deoxys.model.metrics import Metrics, BinaryFbeta, Fbeta
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


def test_binary_fbeta_metric():
    true = [
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 1]
        ],
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]
    ]

    pred = [
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]
    ]

    true = K.constant(true)
    pred = K.constant(pred)
    tp = 17
    fp = 2
    fn = 3

    eps = 1e-8

    def fscore(beta, tp, fp, fn):
        numerator = (1 + beta**2) * tp + eps
        denominator = (1 + beta**2) * tp + beta**2 * fn + fp + eps

        return numerator / denominator

    beta = 1
    metric = BinaryFbeta(beta=beta)
    assert np.isclose(K.eval(metric(true, pred)), fscore(beta, tp, fp, fn))

    beta = 2
    metric = BinaryFbeta(beta=beta)
    assert np.isclose(K.eval(metric(true, pred)), fscore(beta, tp, fp, fn))


def test_fbeta_metric():
    true = [
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ],
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 1]
        ],
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]
    ]

    pred = [
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]
    ]

    true = K.constant(true)
    pred = K.constant(pred)

    values = [{'tp': 2,
               'fp': 0,
               'fn': 1},
              {'tp': 4,
               'fp': 0,
               'fn': 1},
              {'tp': 6,
               'fp': 1,
               'fn': 1},
              {'tp': 5,
               'fp': 1,
               'fn': 0}]

    eps = 1e-8

    def fscore(beta, values):
        res = 0
        for value in values:
            tp, fp, fn = value['tp'], value['fp'], value['fn']
            numerator = (1 + beta**2) * tp + eps
            denominator = (1 + beta**2) * tp + beta**2 * fn + fp + eps

            res += numerator / denominator

        return res / len(values)

    beta = 1
    metric = Fbeta(beta=beta)
    assert np.isclose(K.eval(metric(true, pred)), fscore(beta, values))

    beta = 2
    metric = Fbeta(beta=beta)
    assert np.isclose(K.eval(metric(true, pred)), fscore(beta, values))

    beta = 1
    metric = Fbeta(beta=beta)
    assert np.isclose(
        K.eval(metric(true, pred, sample_weight=[1, 1, 0, 0])),
        fscore(beta, values[:2]))

    beta = 2
    metric = Fbeta(beta=beta)
    assert np.isclose(
        K.eval(metric(true, pred, sample_weight=[1, 1, 2, 1])),
        fscore(beta, values + values[2:3]))
