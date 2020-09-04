# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..utils import Singleton, is_keras_standalone
from ..keras import backend as K
from ..keras.metrics import Metric, deserialize

if is_keras_standalone():
    from keras.metrics import _ConfusionMatrixConditionCount
    from keras.utils.metrics_utils \
        import update_confusion_matrix_variables, parse_init_thresholds, \
        ConfusionMatrix
else:
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

        y_true = K.cast(y_true, y_pred.dtype)

        true_positive = K.sum(y_pred * y_true, axis=reduce_ax)
        target_positive = K.sum(K.square(y_true), axis=reduce_ax)
        predicted_positive = K.sum(
            K.square(y_pred), axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )
        if sample_weight:
            weight = K.cast(sample_weight, self.dtype)
            total_ops = K.update_add(
                self.total,
                K.sum(weight * fb_numerator / fb_denominator))
        else:
            total_ops = K.update_add(
                self.total, K.sum(fb_numerator / fb_denominator))

        count = K.sum(weight) if sample_weight else K.cast(
            K.shape(y_pred)[0], y_pred.dtype)

        count_ops = K.update_add(self.count, count)

        if is_keras_standalone():
            return [total_ops, count_ops]

    def result(self):
        return self.total / self.count

    def get_config(self):
        config = {'threshold': self.threshold,
                  'beta': self.beta}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BinaryFbeta(_ConfusionMatrixConditionCount):
    """
    Calculate the micro f1 score in the set of data
    """

    def __init__(self,
                 thresholds=None,
                 name='BinaryFbeta',
                 dtype=None, beta=1):
        super(Metric, self).__init__(name=name, dtype=dtype)

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
        update = update_confusion_matrix_variables(
            {ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
             ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
             ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
             ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives},
            y_true,
            y_pred,
            thresholds=self.thresholds,
            sample_weight=sample_weight)

        if is_keras_standalone():
            return update

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
            'Fbeta': Fbeta
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

    :param key: the unique key-name of the metric
    :type key: str
    :param metric: the customized metric class
    :type metric: keras.metrics.Metric
    """
    Metrics().register(key, metric)


def unregister_metric(key):
    """
    Remove the registered metric with the key-name

    :param key: the key-name of the metric to be removed
    :type key: str
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
