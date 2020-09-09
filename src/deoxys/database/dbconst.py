# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


class Tables:
    EXPERIMENTS = 'experiments'
    SESSIONS = 'sessions'
    LOGS = 'perf_logs'
    MODELS = 'models'
    PREDICTIONS = 'predictions'


class ConfigRef:
    ARCHITECTURE = 'cf_architecture'
    DATASET_PARAMS = 'cf_dataset_params'
    INPUT_PARAMS = 'cf_input_params'
    MODEL_PARAMS = 'cf_model_params'
    TRAIN_PARAMS = 'cf_train_params'


class Providers:
    MONGODB = 'MongoDB'
    MYSQL = 'MySql'


class LogAttr:
    SESSION_ID = 'session'
    EPOCH = 'epoch'


class SessionAttr:
    EXPERIMENT_ID = 'experiment'
    CREATED_TIME = 'created'
    LAST_MODIFIED = 'lastModified'
    CURRENT_EPOCH = 'curr_epoch'
    STATUS = 'status'


class HDF5Attr:
    SESSION_ID = 'session'
    EPOCH = 'epoch'
    FILE_LOCATION = 'location'


class ExperimentAttr:
    CONFIG = 'config'
    SAVED_MODEL_LOC = 'file_location'
    NAME = 'name'
    DESC = 'description'
    REF_ARCHITECTURE = 'architecture'
    REF_DATASET_PARAMS = 'dataset_params'
    REF_INPUT_PARAMS = 'input_params'
    REF_MODEL_PARAMS = 'model_params'
    REF_TRAIN_PARAMS = 'train_params'


class RefAttr:
    NAME = 'name'
    CONFIG = 'config'
    DESC = 'description'


class SessionStatus:
    CREATED = 'created'
    TRAINING = 'training'
    FINISHED = 'finished'
    FAILED = 'failed'
