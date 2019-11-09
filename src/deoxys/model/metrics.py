# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from tensorflow.keras.metrics import Metric, deserialize
from tensorflow.python.keras.metrics import _ConfusionMatrixConditionCount
from tensorflow.python.keras.utils.metrics_utils \
    import update_confusion_matrix_variables, parse_init_thresholds, \
    ConfusionMatrix
from ..utils import Singleton


class BinaryFbeta(_ConfusionMatrixConditionCount):
    def __init__(self,
                 thresholds=None,
                 name=None,
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
        self.beta = 1

    def update_state(self, y_true, y_pred, sample_weight=None):
        # https://github.com/tensorflow/tensorflow/issues/30711
        # Remove return statement
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
            'BinaryFbeta': BinaryFbeta
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
