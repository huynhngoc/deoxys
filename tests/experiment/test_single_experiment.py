# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


import pytest
from deoxys.experiment import Experiment, ExperimentPipeline, \
    SegmentationExperimentPipeline, DefaultExperimentPipeline
import pandas as pd
import numpy as np


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
    Experiment(
        log_base_path='../../oxford_perf/logs',
        best_model_monitors=['val_loss', 'val_accuracy']
    ).from_file(
        'tests/h5_files/model.002.h5'
    ).run_test(masked_images=[i for i in range(3)])


def test_run_pipeline():
    ExperimentPipeline(
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
        masked_images=[i for i in range(3)], best_num=1, worst_num=1
    ).load_best_model(
        monitor='val_dice', use_raw_log=True
    ).run_test(
        masked_images=[i for i in range(3)]
    ).apply_post_processors(
        map_meta_data='patient_idx',
        run_test=True,
        recipe='2d'
    ).plot_3d_test_images(best_num=1, worst_num=1)


def test_run_pipeline_3d():
    ExperimentPipeline(
        log_base_path='../../hn_perf/3d_xs'
    ).from_full_config(
        'tests/json/sample_dataset_xs_3d_full.json'
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        prediction_checkpoint_period=1,
        epochs=1
    ).apply_post_processors(
        map_meta_data='patient_idx'
    ).load_best_model(
    ).run_test(
    ).apply_post_processors(
        map_meta_data='patient_idx',
        run_test=True,
        recipe='3d'
    )


def test_run_pipeline_3d_patch():
    ExperimentPipeline(
        log_base_path='../../hn_perf/3d_xs_patch'
    ).from_full_config(
        'tests/json/sample_dataset_xs_3d_patch.json'
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        prediction_checkpoint_period=1,
        epochs=1
    ).apply_post_processors(
        map_meta_data='patient_idx'
    ).load_best_model(
    ).run_test(
    ).apply_post_processors(
        map_meta_data='patient_idx',
        run_test=True,
        recipe='patch'
    )


def test_run_seg_pipeline():
    SegmentationExperimentPipeline(
        log_base_path='../../hn_perf/2d_xs_seg'
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
        masked_images=[i for i in range(3)], best_num=1, worst_num=1
    ).load_best_model(
        monitor='val_dice', use_raw_log=True
    ).run_test(
        masked_images=[i for i in range(3)]
    ).apply_post_processors(
        map_meta_data='patient_idx',
        run_test=True,
        recipe='2d',
        metrics=['f1_score', 'precision']
    ).plot_3d_test_images(best_num=1, worst_num=1)

    res_df = pd.read_csv('../../hn_perf/2d_xs_seg/test/result.csv')
    assert np.all(res_df.columns[1:] == ['f1_score', 'precision'])


def test_run_seg_pipeline_3d():
    SegmentationExperimentPipeline(
        log_base_path='../../hn_perf/3d_xs_seg'
    ).from_full_config(
        'tests/json/sample_dataset_xs_3d_full.json'
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        prediction_checkpoint_period=1,
        epochs=1
    ).apply_post_processors(
        map_meta_data='patient_idx',
        metrics=['f1_score', 'f1_score'],
        metrics_kwargs=[{'metric_name': 'f1_score'},
                        {'metric_name': 'f2_score', 'beta': 2}]
    ).load_best_model(
    ).run_test(
    ).apply_post_processors(
        map_meta_data='patient_idx',
        run_test=True,
        recipe='3d',
        metrics=['f1_score', 'TPR']
    )

    res_df = pd.read_csv('../../hn_perf/3d_xs_seg/log_new.csv')
    assert np.all(res_df.columns == ['epochs', 'f1_score', 'f2_score'])

    res_df = pd.read_csv('../../hn_perf/3d_xs_seg/test/result.csv')
    assert np.all(res_df.columns[1:] == ['f1_score', 'recall'])


def test_run_seg_pipeline_3d_patch():
    SegmentationExperimentPipeline(
        log_base_path='../../hn_perf/3d_xs_patch_seg'
    ).from_full_config(
        'tests/json/sample_dataset_xs_3d_patch.json'
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        prediction_checkpoint_period=1,
        epochs=1
    ).apply_post_processors(
        map_meta_data='patient_idx',
        metrics=['f1_score', 'recall', 'precision', 'FPR'],
        metrics_kwargs=[{'metric_name': 'fbeta'},
                        {'metric_name': 'recall'},
                        {},
                        {'metric_name': 'false_positive_rate'}]
    ).load_best_model(
        monitor='fbeta'
    ).run_test(
    ).apply_post_processors(
        map_meta_data='patient_idx',
        run_test=True,
        recipe='patch',
        metrics=['f1_score', 'recall', 'precision']
    )

    res_df = pd.read_csv('../../hn_perf/3d_xs_patch_seg/log_new.csv')
    assert np.all(res_df.columns == [
                  'epochs', 'fbeta', 'recall',
                  'precision', 'false_positive_rate'])

    res_df = pd.read_csv('../../hn_perf/3d_xs_patch_seg/test/result.csv')
    assert np.all(res_df.columns[1:] == ['f1_score', 'recall', 'precision'])


def test_run_default_pipeline_3d():
    def binarize(targets, predictions):
        return targets, (predictions > 0.5).astype(targets.dtype)

    DefaultExperimentPipeline(
        log_base_path='../../hn_perf/3d_xs_df'
    ).from_full_config(
        'tests/json/sample_dataset_xs_3d_full_classification.json'
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        prediction_checkpoint_period=1,
        epochs=1
    ).apply_post_processors(
        map_meta_data='patient_idx',
        metrics=['AUC', 'BinaryCrossentropy',
                 'BinaryAccuracy', 'BinaryFbeta', 'roc_auc', 'f1'],
        metrics_sources=['tf',  'tf', 'tf', 'tf', 'sklearn', 'sklearn'],
        process_functions=[None, None, None, None, None, binarize]
    ).load_best_model(
        monitor='AUC', mode='max'
    ).run_test(
    ).apply_post_processors(
        map_meta_data='patient_idx',
        run_test=True,
        recipe='auto',
        metrics=['BinaryFbeta', 'BinaryFbeta'],
        metrics_sources='tf',
        metrics_kwargs=[{'metric_name': 'f1_score'},
                        {'metric_name': 'f2_score', 'beta': 2}],
    )

    res_df = pd.read_csv('../../hn_perf/3d_xs_df/log_new.csv')
    assert np.all(res_df.columns == ['epochs', 'AUC',
                                     'BinaryCrossentropy',
                                     'BinaryAccuracy', 'BinaryFbeta',
                                     'roc_auc', 'f1'])

    res_df = pd.read_csv('../../hn_perf/3d_xs_df/test/result.csv')
    assert np.all(res_df.columns[1:] == ['f1_score', 'f2_score'])
