# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pytest
from deoxys.experiment import Experiment


def test_run_experiment():
    exp = Experiment(
        log_base_path='../../oxford_perf/logs',
        best_model_monitors=['val_loss', 'val_accuracy']
    ).from_full_config(
        'tests/json/sample_dataset_xs_config.json'
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        prediction_checkpoint_period=1,
        epochs=2
    ).plot_performance().plot_prediction(
        masked_images=[i for i in range(10)],
    ).plot_prediction(
        [i for i in range(20)],
        img_name='Prediction_{index:02d}_contour.png',
        contour=True
    ).plot_prediction(
        [i for i in range(20)],
        img_name='Prediction_{index:02d}.png',
        contour=False
    )

    # Best model in this run
    print(exp.best_model())
