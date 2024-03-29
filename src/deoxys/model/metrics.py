# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from ..utils import Singleton
# from ..keras import backend as K
from tensorflow.keras.metrics import Metric, deserialize
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras import backend
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.metrics import _ConfusionMatrixConditionCount
from tensorflow.python.keras.utils.metrics_utils \
    import update_confusion_matrix_variables, parse_init_thresholds, \
    ConfusionMatrix


class Fbeta(Metric):
    def __init__(self, threshold=None, name='Fbeta', dtype=None, beta=1):
        super().__init__(name=name, dtype=dtype)

        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta

        self.total = self.add_weight(
            'total', initializer='zeros')
        self.count = self.add_weight(
            'count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        size = len(y_pred.get_shape().as_list())
        reduce_ax = list(range(1, size))
        eps = 1e-8

        y_true = tf.cast(y_true, y_pred.dtype)

        true_positive = tf.reduce_sum(y_pred * y_true, axis=reduce_ax)
        target_positive = tf.reduce_sum(tf.square(y_true), axis=reduce_ax)
        predicted_positive = tf.reduce_sum(
            tf.square(y_pred), axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )
        if sample_weight:
            weight = tf.cast(sample_weight, self.dtype)
            # total_ops = K.update_add(
            #     self.total,
            #     tf.reduce_sum(weight * fb_numerator / fb_denominator))
            self.total.assign_add(tf.reduce_sum(
                weight * fb_numerator / fb_denominator))
        else:
            # total_ops = K.update_add(
            #     self.total, tf.reduce_sum(fb_numerator / fb_denominator))
            self.total.assign_add(tf.reduce_sum(fb_numerator / fb_denominator))

        count = tf.reduce_sum(weight) if sample_weight else tf.cast(
            tf.shape(y_pred)[0], y_pred.dtype)

        # count_ops = K.update_add(self.count, count)
        self.count.assign_add(count)

    def result(self):
        return self.total / self.count

    def get_config(self):
        config = {'threshold': self.threshold,
                  'beta': self.beta}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dice(Metric):
    def __init__(self, threshold=None, name='dice', dtype=None, beta=1):
        super().__init__(name=name, dtype=dtype)

        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta

        self.total = self.add_weight(
            'total', initializer='zeros')
        self.count = self.add_weight(
            'count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        size = len(y_pred.get_shape().as_list())
        reduce_ax = list(range(1, size))
        eps = 1e-8

        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.cast(y_pred > self.threshold, y_true.dtype)

        true_positive = tf.reduce_sum(y_pred * y_true, axis=reduce_ax)
        target_positive = tf.reduce_sum(y_true, axis=reduce_ax)
        predicted_positive = tf.reduce_sum(y_pred, axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )
        if sample_weight:
            weight = tf.cast(sample_weight, self.dtype)
            # total_ops = K.update_add(
            #     self.total,
            #     tf.reduce_sum(weight * fb_numerator / fb_denominator))
            self.total.assign_add(
                tf.reduce_sum(weight * fb_numerator / fb_denominator)
            )
        else:
            # total_ops = K.update_add(
            #     self.total, tf.reduce_sum(fb_numerator / fb_denominator))
            self.total.assign_add(
                tf.reduce_sum(fb_numerator / fb_denominator)
            )

        count = tf.reduce_sum(weight) if sample_weight else tf.cast(
            tf.shape(y_pred)[0], y_pred.dtype)

        # count_ops = K.update_add(self.count, count)
        self.count.assign_add(count)

    def result(self):
        return self.total / self.count

    def get_config(self):
        config = {'threshold': self.threshold,
                  'beta': self.beta}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BinaryFbeta(Metric):
    """
    Calculate the micro f1 score in the set of data
    """

    def __init__(self,
                 thresholds=None,
                 name='BinaryFbeta',
                 dtype=None, beta=1):
        # super(Metric, self).__init__(name=name, dtype=dtype)
        super().__init__(name=name, dtype=dtype)

        self.init_thresholds = thresholds
        self.thresholds = parse_init_thresholds(
            thresholds, default_threshold=0.5)
        num_thresholds = len(self.thresholds)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(num_thresholds,),
            initializer='zeros')
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(num_thresholds,),
            initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(num_thresholds,),
            initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(num_thresholds,),
            initializer='zeros')
        self.beta = beta

    def update_state(self, y_true, y_pred, sample_weight=None):
        # https://github.com/tensorflow/tensorflow/issues/30711
        # Remove return statement in case tensorflow.keras
        update_confusion_matrix_variables(
            {ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
             ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
             ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
             ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives},
            y_true,
            y_pred,
            thresholds=self.thresholds,
            sample_weight=sample_weight)

    def result(self):
        res = []
        eps = 1e-8
        for i in range(len(self.thresholds)):
            fb_numerator = (1 + self.beta ** 2) * self.true_positives + eps
            fb_denominator = (
                (self.beta ** 2) * (self.true_positives + self.false_negatives)
                + self.true_positives + self.false_positives + eps
            )
            fscore = fb_numerator / fb_denominator
            res.append(fscore)
        if len(res) == 1:
            return res[0]
        return res

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {'thresholds': self.init_thresholds}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Metrics(metaclass=Singleton):
    """
    A singleton that contains all the registered customized metrics
    """

    def __init__(self):
        self._metrics = {
            'BinaryFbeta': BinaryFbeta,
            'Fbeta': Fbeta,
            'Dice': Dice
        }

    def register(self, key, metric):
        if not issubclass(metric, Metric):
            raise ValueError(
                "The customized metric has to be a subclass"
                + " of keras.metrics.Metric"
            )

        if key in self._metrics:
            raise KeyError(
                "Duplicated key, please use another key for this metric"
            )
        else:
            self._metrics[key] = metric

    def unregister(self, key):
        if key in self._metrics:
            del self._metrics[key]

    @property
    def metrics(self):
        return self._metrics


def register_metric(key, metric):
    """
    Register the customized metric.
    If the key name is already registered, it will raise a KeyError exception

    Parameters
    ----------
    key : str
        The unique key-name of the metric
    loss : tensorflow.keras.metrics.Metric
        The customized metric class
    """
    Metrics().register(key, metric)


def unregister_metric(key):
    """
    Remove the registered metric with the key-name

    Parameters
    ----------
    key : str
        The key-name of the metric to be removed
    """
    Metrics().unregister(key)


def metric_from_config(config):
    if type(config) == dict:
        if 'class_name' not in config:
            raise ValueError('class_name is needed to define metric')

        if 'config' not in config:
            # auto add empty config for metric with only class_name
            config['config'] = {}
        return deserialize(
            config,
            custom_objects=Metrics().metrics)

    return deserialize(config, custom_objects=Metrics().metrics)
