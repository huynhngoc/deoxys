# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pytest
from deoxys.experiment import Experiment, ExperimentPipeline


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
        [i for i in range(10)],
        img_name='Prediction_{index:02d}_contour.png',
        contour=True
    ).plot_prediction(
        [i for i in range(10)],
        img_name='Prediction_{index:02d}.png',
        contour=False
    )

    # Best model in this run
    print(exp.best_model())


def test_run_experiment_different_size():
    exp = Experiment(
        log_base_path='../../oxford_perf_2/logs',
        best_model_monitors=['val_loss', 'val_accuracy']
    ).from_full_config(
        'tests/json/sample_dataset_xs_config_diff_size.json'
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        prediction_checkpoint_period=1,
        epochs=2
    ).plot_performance().plot_prediction(
        masked_images=[i for i in range(10)],
        base_image_name='00/x',
        truth_image_name='00/y',
        predicted_image_name='00/predicted'
    )

    # Best model in this run
    print(exp.best_model())

    exp.run_test(
        masked_images=[i for i in range(10)],
        base_image_name='00/x',
        truth_image_name='00/y',
        predicted_image_name='00/predicted')


def test_continue_experiment():
    exp = Experiment(
        log_base_path='../../oxford_perf/logs',
        best_model_monitors=['val_loss', 'val_accuracy']
    ).from_file(
        'tests/h5_files/model.002.h5'
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        prediction_checkpoint_period=1,
        epochs=3,
        initial_epoch=2
    ).plot_performance().plot_prediction(
        masked_images=[i for i in range(10)],
    )


def test_run_test():
    exp = Experiment(
        log_base_path='../../oxford_perf/logs',
        best_model_monitors=['val_loss', 'val_accuracy']
    ).from_file(
        'tests/h5_files/model.002.h5'
    ).run_test(masked_images=[i for i in range(10)])


def test_run_pipeline():
    ex = ExperimentPipeline(
        log_base_path='../../hn_perf/2d_xs'
    ).from_full_config(
        'tests/json/sample_dataset_xs_config_diff_size.json'
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        prediction_checkpoint_period=1,
        epochs=2
    ).apply_post_processors(
        map_meta_data='patient_idx'
    ).plot_prediction(
        masked_images=[i for i in range(10)], best_num=1, worst_num=1
    ).load_best_model(
    ).run_test(
    ).apply_post_processors(
        map_meta_data='patient_idx',
        run_test=True
    ).plot_3d_test_images(best_num=1, worst_num=1)
