# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"
__version__ = "0.0.1"


from ..database import Tables, SessionStatus, SessionAttr, ExperimentAttr, \
    RefAttr, ConfigRef
from itertools import product
from . import ExperimentDB


class MultiExperimentDB:

    def __init__(self, dbclient):
        self.dbclient = dbclient

    @property
    def experiments(self):
        """
        List all experiements

        :return: [description]
        :rtype: [type]
        """
        all_exp = self.dbclient.find_all(Tables.EXPERIMENTS)

        return self.dbclient.to_pandas(all_exp)

    def sessions_from_experiments(self, experiment_id):
        """
        List all sessions (created, training, finished, failed) of
        an experiments

        :param experiment_id: [description]
        :type experiment_id: [type]
        :return: [description]
        :rtype: [type]
        """
        sessions = self.dbclient.find_by_col(
            Tables.SESSIONS, SessionAttr.EXPERIMENT_ID, experiment_id)

        return self.dbclient.to_pandas(sessions)

    def run_new_session(self, experiment_id, epochs, log_base_path='logs',
                        model_checkpoint_period=2,
                        prediction_checkpoint_period=2):
        """
        Start a new session of an experiment

        :param experiment_id: [description]
        :type experiment_id: [type]
        :param epochs: [description]
        :type epochs: [type]
        :param log_base_path: [description], defaults to 'logs'
        :type log_base_path: str, optional
        :param model_checkpoint_period: [description], defaults to 2
        :type model_checkpoint_period: int, optional
        :param prediction_checkpoint_period: [description], defaults to 2
        :type prediction_checkpoint_period: int, optional
        :return: [description]
        :rtype: [type]
        """
        exp = ExperimentDB(
            self.dbclient, experiment_id=experiment_id,
            log_base_path=log_base_path
        ).run_experiment(
            model_checkpoint_period=model_checkpoint_period,
            prediction_checkpoint_period=prediction_checkpoint_period,
            save_origin_images=False, verbose=1, epochs=epochs)

        return exp

    def run_multiple_new_session(self, num, experiment_id, epochs,
                                 log_base_path='logs',
                                 model_checkpoint_period=2,
                                 prediction_checkpoint_period=2):
        """
        Run `num` number of new sessions of an experiment

        :param num: [description]
        :type num: [type]
        :param experiment_id: [description]
        :type experiment_id: [type]
        :param epochs: [description]
        :type epochs: [type]
        :param log_base_path: [description], defaults to 'logs'
        :type log_base_path: str, optional
        :param model_checkpoint_period: [description], defaults to 2
        :type model_checkpoint_period: int, optional
        :param prediction_checkpoint_period: [description], defaults to 2
        :type prediction_checkpoint_period: int, optional
        """
        return_exps = []
        for _ in range(num):
            try:
                exp = ExperimentDB(
                    self.dbclient, experiment_id=experiment_id,
                    log_base_path=log_base_path
                ).run_experiment(
                    model_checkpoint_period=model_checkpoint_period,
                    prediction_checkpoint_period=prediction_checkpoint_period,
                    save_origin_images=False, verbose=1, epochs=epochs)
                return_exps.append(exp)
            except Exception:
                pass

        return return_exps

    def continue_session(self, session_id, epochs, log_base_path='logs',
                         model_checkpoint_period=2,
                         prediction_checkpoint_period=2):
        """
        Continue a session from a model checkpoint

        :param session_id: [description]
        :type session_id: [type]
        :param epochs: [description]
        :type epochs: [type]
        :param log_base_path: [description], defaults to 'logs'
        :type log_base_path: str, optional
        :param model_checkpoint_period: [description], defaults to 2
        :type model_checkpoint_period: int, optional
        :param prediction_checkpoint_period: [description], defaults to 2
        :type prediction_checkpoint_period: int, optional
        :return: [description]
        :rtype: [type]
        """
        exp = ExperimentDB(
            self.dbclient, session_id=session_id, log_base_path=log_base_path
        ).run_experiment(
            model_checkpoint_period=model_checkpoint_period,
            prediction_checkpoint_period=prediction_checkpoint_period,
            save_origin_images=False, verbose=1, epochs=epochs)

        return exp

    def continue_multiple_session(self, session_id_list, epochs,
                                  log_base_path='logs',
                                  model_checkpoint_period=2,
                                  prediction_checkpoint_period=2):
        """
        Continue multiple sessions

        :param session_id_list: [description]
        :type session_id_list: [type]
        :param epochs: [description]
        :type epochs: [type]
        :param log_base_path: [description], defaults to 'logs'
        :type log_base_path: str, optional
        :param model_checkpoint_period: [description], defaults to 2
        :type model_checkpoint_period: int, optional
        :param prediction_checkpoint_period: [description], defaults to 2
        :type prediction_checkpoint_period: int, optional
        """
        return_exps = []
        for session_id in session_id_list:
            try:
                exp = ExperimentDB(
                    self.dbclient, session_id=session_id,
                    log_base_path=log_base_path
                ).run_experiment(
                    model_checkpoint_period=model_checkpoint_period,
                    prediction_checkpoint_period=prediction_checkpoint_period,
                    save_origin_images=False, verbose=1, epochs=epochs)

                return_exps.append(exp)
            except Exception:
                pass

        return return_exps

    def new_experiment_from_full_config(self, name, config, description=''):
        """
        Add a new experiement from a configuration JSON

        :param name: [description]
        :type name: [type]
        :param config: [description]
        :type config: [type]
        :param description: [description], defaults to ''
        :type description: str, optional
        :return: [description]
        :rtype: [type]
        """
        new_exp = {
            ExperimentAttr.NAME: name,
            ExperimentAttr.CONFIG: config,
            ExperimentAttr.DESC: description
        }

        return self.dbclient.insert(Tables.EXPERIMENTS, new_exp).inserted_id

    def new_experiment_from_h5file(self, name, file_path, description=''):
        """
        Add a new experiment from a h5 model file.
        This is used in case we want to train a modified model on disk.

        :param name: [description]
        :type name: [type]
        :param file_path: [description]
        :type file_path: [type]
        :param description: [description], defaults to ''
        :type description: str, optional
        :return: [description]
        :rtype: [type]
        """
        new_exp = {
            ExperimentAttr.NAME: name,
            ExperimentAttr.SAVED_MODEL_LOC: file_path,
            ExperimentAttr.DESC: description
        }

        return self.dbclient.insert(Tables.EXPERIMENTS, new_exp).inserted_id

    def new_experiment_from_components(self, dataset_params, input_params,
                                       architecture, model_params,
                                       train_params=None,
                                       name='', description=''):
        """
        Create a new experiment by combining pre-defined JSON config

        :param dataset_params: [description]
        :type dataset_params: [type]
        :param input_params: [description]
        :type input_params: [type]
        :param architecture: [description]
        :type architecture: [type]
        :param model_params: [description]
        :type model_params: [type]
        :param train_params: [description], defaults to None
        :type train_params: [type], optional
        :param name: [description], defaults to ''
        :type name: str, optional
        :param description: [description], defaults to ''
        :type description: str, optional
        :return: [description]
        :rtype: [type]
        """
        exp_name = name or ''
        try:
            # find each component by id then create the full JSON
            dataset_params_obj = self.dbclient.find_by_id(
                ConfigRef.DATASET_PARAMS, dataset_params)
            dataset_params_cf = dataset_params_obj[RefAttr.CONFIG]

            input_params_obj = self.dbclient.find_by_id(
                ConfigRef.INPUT_PARAMS, input_params)
            input_params_cf = input_params_obj[RefAttr.CONFIG]

            architecture_obj = self.dbclient.find_by_id(
                ConfigRef.ARCHITECTURE, architecture)
            architecture_cf = architecture_obj[RefAttr.CONFIG]

            model_params_obj = self.dbclient.find_by_id(
                ConfigRef.MODEL_PARAMS, model_params)
            model_params_cf = model_params_obj[RefAttr.CONFIG]

            if train_params:
                train_params_obj = self.dbclient.find_by_id(
                    ConfigRef.TRAIN_PARAMS, train_params)
                train_params_cf = train_params_obj[RefAttr.CONFIG]
            else:
                train_params_cf = '{ }'

            config = self._new_config_from_components(
                dataset_params_cf, input_params_cf, architecture_cf,
                model_params_cf, train_params_cf
            )

            if not name:
                exp_name = ' - '.join([
                    dataset_params_obj[RefAttr.NAME],
                    input_params_obj[RefAttr.NAME],
                    architecture_obj[RefAttr.NAME],
                    model_params_obj[RefAttr.NAME],
                    train_params_obj[RefAttr.NAME] if train_params else '',
                ])
            return self.new_experiment_from_full_config(
                exp_name, config, description=description)
        except Exception:
            return None

    def new_multi_experiments_from_components(self, dataset_params,
                                              input_params,
                                              architecture, model_params,
                                              train_params=None):
        """
        Create multiple experiments by combination of all components

        :param dataset_params: [description]
        :type dataset_params: [type]
        :param input_params: [description]
        :type input_params: [type]
        :param architecture: [description]
        :type architecture: [type]
        :param model_params: [description]
        :type model_params: [type]
        :param train_params: [description], defaults to None
        :type train_params: [type], optional
        :param name: [description], defaults to ''
        :type name: str, optional
        :param description: [description], defaults to ''
        :type description: str, optional
        """
        # Assuming they're all list
        components = [dataset_params, input_params, architecture, model_params,
                      train_params if train_params else [None]]
        args_list = product(*components)

        inserted_ids = []
        for args in args_list:
            inserted_ids.append(
                self.new_multi_experiments_from_components(*args))
        return inserted_ids

    def rename_experiment(self, experiment_id, new_name):
        """
        Rename an experiment by id

        :param experiment_id: [description]
        :type experiment_id: [type]
        :param new_name: [description]
        :type new_name: [type]
        """
        return self.dbclient.update_by_id(Tables.EXPERIMENTS, experiment_id, {
            ExperimentAttr.NAME: new_name
        })

    def update_experiment_description(self, experiment_id, description):
        """
        Rename an experiment by id

        :param experiment_id: [description]
        :type experiment_id: [type]
        :param new_name: [description]
        :type new_name: [type]
        """
        return self.dbclient.update_by_id(Tables.EXPERIMENTS, experiment_id, {
            ExperimentAttr.DESC: description
        })

    def new_architecture_config(self, name, config, description=''):
        """
        Add a new JSON config for architecture.

        :param name: [description]
        :type name: [type]
        :param config: [description]
        :type config: [type]
        :param description: [description], defaults to ''
        :type description: str, optional
        :return: [description]
        :rtype: [type]
        """
        return self._new_ref_config(
            ConfigRef.ARCHITECTURE, name, config, description)

    def new_dataset_config(self, name, config, description=''):
        """
        Add a new JSON config for dataset_params

        :param name: [description]
        :type name: [type]
        :param config: [description]
        :type config: [type]
        :param description: [description], defaults to ''
        :type description: str, optional
        :return: [description]
        :rtype: [type]
        """
        return self._new_ref_config(
            ConfigRef.DATASET_PARAMS, name, config, description)

    def new_input_config(self, name, config, description=''):
        """
        Add a new JSON config for input_params

        :param name: [description]
        :type name: [type]
        :param config: [description]
        :type config: [type]
        :param description: [description], defaults to ''
        :type description: str, optional
        :return: [description]
        :rtype: [type]
        """
        return self._new_ref_config(
            ConfigRef.INPUT_PARAMS, name, config, description)

    def new_train_config(self, name, config, description=''):
        """
        Add a new JSON config for train_params

        :param name: [description]
        :type name: [type]
        :param config: [description]
        :type config: [type]
        :param description: [description], defaults to ''
        :type description: str, optional
        :return: [description]
        :rtype: [type]
        """
        return self._new_ref_config(
            ConfigRef.TRAIN_PARAMS, name, config, description)

    def new_model_params_config(self, name, config, description=''):
        """
        Add a new JSON config for model_params

        :param name: [description]
        :type name: [type]
        :param config: [description]
        :type config: [type]
        :param description: [description], defaults to ''
        :type description: str, optional
        :return: [description]
        :rtype: [type]
        """
        return self._new_ref_config(
            ConfigRef.MODEL_PARAMS, name, config, description)

    def _new_ref_config(self, table, name, config, descripiton=''):
        new_obj = {
            RefAttr.NAME: name,
            RefAttr.CONFIG: config,
            RefAttr.DESC: descripiton
        }

        return self.dbclient.insert(
            table, new_obj).inserted_id

    def _new_config_from_components(self, dataset_params, input_params,
                                    architecture, model_params, train_params):
        return '{"{}":{}, "{}":{},"{}":{}, "{}":{}, "{}":{}}'.format(
            "dataset_params", dataset_params,
            "input_params", input_params,
            "architecture", architecture,
            "model_params", model_params,
            "train_params", train_params
        )
