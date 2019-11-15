# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..model import model_from_full_config, model_from_config


class Experiment:
    def __init__(self):
        self.model = None

        self.architecture = None
        self.input_params = None
        self.model_params = None
        self.train_params = None
        self.data_reader = None
        self.weights_file = None

    def from_full_config(self, file, weights_file=None, **kwargs):
        self.model = model_from_full_config(file,
                                            weights_file=weights_file,
                                            **kwargs)

        return self

    def from_config(self, architecture, input_params,
                    model_params, train_params,
                    dataset_params, weights_file=None):
        self.model = model_from_config(architecture, input_params,
                                       model_params, train_params,
                                       dataset_params,
                                       weights_file=weights_file)

        return self

    def run_experiment(self, test_checkpoint=0, test_file=None,
                       plot_performance=False, model_filename=None):
        if self._check_run():
            train_history = None
            test_history = None

            kwargs = {}
            # if test_checkpoint:
            #     test_history = []
            #     kwargs['callbacks'] = [
            #         TestCheckpoint(self.model,
            #                        filename=test_file)]

            train_history = self.model.fit_train(**kwargs)
            if model_filename:
                self.model.save(model_filename)

            if plot_performance:
                pass

            return train_history, test_history

    def set_params(self, params):
        """[summary]

        :param params: Dictionary of params
        :type params: [type]
        :return: [description]
        :rtype: [type]
        """

        # for key in params:
        #     if key == 'architecture':
        #         # check if a model exists
        #         if self.model:
        #             self.model = model_from_config(
        #                 architecture=params[key],
        #                 input_params=self.input_params,
        #                 model_params=self.model_params,
        #                 train_params=self.train_params,
        #                 weights_file=self.weights_file
        #             )
        #         else:
        #             if self.input_params:

        #     self.architecture = params[key]
        # self.input_params = None
        # self.model_params = None
        # self.train_params = None
        # self.data_reader = None

        return self

    def _check_run(self):
        if self.model:
            if self.model._data_reader:
                if self.model.is_compiled:
                    return True
        raise RuntimeError("Cannot run experiment with incomplete model")
        return False
