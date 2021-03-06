# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


from ..database import Tables, SessionStatus, SessionAttr, ExperimentAttr, \
    RefAttr, ConfigRef, LogAttr
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import ExperimentDB


class MultiExperimentDB:  # pragma: no cover

    def __init__(self, dbclient):
        self.dbclient = dbclient

    @property
    def experiments(self):
        """
        List all experiements
        """
        all_exp = self.dbclient.find_all(Tables.EXPERIMENTS)

        return self.dbclient.to_pandas(all_exp)

    def sessions_from_experiments(self, experiment_id):
        """
        List all sessions (created, training, finished, failed) of
        an experiments
        """
        sessions = self.dbclient.find_by_col(
            Tables.SESSIONS, SessionAttr.EXPERIMENT_ID,
            self.dbclient.to_fk(experiment_id))

        return self.dbclient.to_pandas(sessions)

    def run_new_session(self, experiment_id, epochs, log_base_path='logs',
                        model_checkpoint_period=2,
                        prediction_checkpoint_period=2):
        """
        Start a new session of an experimenton]
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

    def session_performance(self, session_id, metrics=None,
                            ax=None, shading='std'):
        perf = self.dbclient.to_pandas(
            self.dbclient.find_by_col(
                Tables.LOGS, LogAttr.SESSION_ID,
                self.dbclient.to_fk(session_id)))

        not_include = ['_id', LogAttr.SESSION_ID, LogAttr.EPOCH]
        metric_list = metrics or [
            col for col in perf.columns if col not in not_include]
        perf = perf.groupby(LogAttr.EPOCH).agg({
            metric: ['mean', lambda val: np.mean(
                val) - np.std(val), lambda val: np.mean(val) + np.std(val)]
            for metric in metric_list
        })

        postfixes = ['', '_nev', '_pos']
        perf.columns = ['{}{}'.format(metric, postfix)
                        for metric in metric_list for postfix in postfixes]

        if type(session_id) == list:
            # plot mean data instead
            pass
        if ax is None:
            ax = plt.axes()
        epochs = perf.index
        for metric in metric_list:
            ax.plot(epochs, perf[metric], label=metric)
            ax.fill_between(epochs, perf[metric + '_nev'],
                            perf[metric + '_pos'], alpha=0.2)

        ax.legend()
        return ax

    def experiment_performance(self, experiment_id, metrics=None, ax=None,
                               shading='std'):
        sessions = self.dbclient.to_pandas(
            self.dbclient.find_by_col(
                Tables.SESSIONS, SessionAttr.EXPERIMENT_ID,
                self.dbclient.to_fk(experiment_id)))

        return self.session_performance(
            list(sessions['_id'].values), metrics, ax, shading)

    def new_experiment_from_full_config(self, name, config, description=''):
        """
        Add a new experiement from a configuration JSON
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
        """
        return self.dbclient.update_by_id(Tables.EXPERIMENTS, experiment_id, {
            ExperimentAttr.NAME: new_name
        })

    def update_experiment_description(self, experiment_id, description):
        """
        Rename an experiment by id
        """
        return self.dbclient.update_by_id(Tables.EXPERIMENTS, experiment_id, {
            ExperimentAttr.DESC: description
        })

    def new_architecture_config(self, name, config, description=''):
        """
        Add a new JSON config for architecture.
        """
        return self._new_ref_config(
            ConfigRef.ARCHITECTURE, name, config, description)

    def new_dataset_config(self, name, config, description=''):
        """
        Add a new JSON config for dataset_params
        """
        return self._new_ref_config(
            ConfigRef.DATASET_PARAMS, name, config, description)

    def new_input_config(self, name, config, description=''):
        """
        Add a new JSON config for input_params
        """
        return self._new_ref_config(
            ConfigRef.INPUT_PARAMS, name, config, description)

    def new_train_config(self, name, config, description=''):
        """
        Add a new JSON config for train_params
        """
        return self._new_ref_config(
            ConfigRef.TRAIN_PARAMS, name, config, description)

    def new_model_params_config(self, name, config, description=''):
        """
        Add a new JSON config for model_params
        """
        return self._new_ref_config(
            ConfigRef.MODEL_PARAMS, name, config, description)

    def json_data(self, val):
        return self.dbclient.df_to_json(val)

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
