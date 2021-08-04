from .single_experiment import Experiment
from ..model.callbacks import PredictionCheckpoint
from .postprocessor import SegmentationPostProcessor
from ..utils import load_json_config

from deoxys_vis import plot_log_performance_from_csv, mask_prediction, read_csv

import numpy as np
import h5py
import os


class ExperimentPipeline(Experiment):
    def __init__(self,
                 log_base_path='logs',
                 temp_base_path='',
                 best_model_monitors='val_loss',
                 best_model_modes='auto'):

        self.temp_base_path = temp_base_path or (log_base_path + '_temp')
        self.post_processors = None

        super().__init__(log_base_path, best_model_monitors, best_model_modes)

    def _create_prediction_checkpoint(self, base_path, period, use_original):
        temp_base_path = self.temp_base_path
        if not os.path.exists(temp_base_path):
            os.makedirs(temp_base_path)

        if not os.path.exists(temp_base_path + self.PREDICTION_PATH):
            os.makedirs(temp_base_path + self.PREDICTION_PATH)

        pred_base_path = temp_base_path

        return PredictionCheckpoint(
            filepath=pred_base_path + self.PREDICTION_PATH +
            self.PREDICTION_NAME,
            period=period, use_original=use_original)

    def plot_prediction(self, masked_images=None,
                        contour=True,
                        base_image_name='x',
                        truth_image_name='y',
                        predicted_image_name='predicted',
                        predicted_image_title_name='Image {index:05d}',
                        img_name='{index:05d}.png',
                        monitor='',
                        best_num=2, worst_num=2):
        if masked_images:
            self._plot_prediction(
                self.temp_base_path,
                masked_images,
                contour=contour,
                base_image_name=base_image_name,
                truth_image_name=truth_image_name,
                predicted_image_name=predicted_image_name,
                predicted_image_title_name=predicted_image_title_name,
                img_name=img_name)

        prediction_path = self.log_base_path + self.PREDICTION_PATH

        if (self.post_processors is not None) and os.path.exists(
                prediction_path):
            print('\nCreating prediction images...')

            images_list = self.post_processors.get_best_performance_images(
                monitor=monitor,
                best_num=best_num,
                worst_num=worst_num
            )
            print(images_list)
            for item in images_list:
                filename = item['file_name']

                for image_id, score in zip(item['ids'], item['values']):
                    self.plot_prediction_3d(
                        filename, image_id, score,
                        base_image_name=f'x/{image_id}',
                        truth_image_name=f'y/{image_id}',
                        predicted_image_name=f'predicted/{image_id}',
                    )
        return self

    def plot_prediction_3d(self, filename, image_id, score,
                           base_image_name,
                           truth_image_name, predicted_image_name):

        prediced_image_path = self.log_base_path + self.PREDICTED_IMAGE_PATH
        prediction_path = self.log_base_path + self.PREDICTION_PATH
        if not os.path.exists(prediced_image_path):
            os.makedirs(prediced_image_path)

        images_path = prediced_image_path + filename
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        images_path_single = images_path + '/' + str(image_id)
        if not os.path.exists(images_path_single):
            os.makedirs(images_path_single)

        with h5py.File(prediction_path + filename, 'r') as f:
            images = f[base_image_name][:]
            true_mask = f[truth_image_name][:]
            pred_mask = f[predicted_image_name][:]

        print('plotting 3d images...', images_path_single)
        try:
            mask_prediction(images_path_single + '.html',
                            image=images,
                            true_mask=true_mask,
                            pred_mask=pred_mask,
                            title=f'Patient {image_id}, DSC {score}')
        except Exception as e:
            print('An error occurred while plotting 3d images', e)

        print('plotting 2d images...')
        for i in range(len(images)):
            mask_prediction(
                images_path_single + f'/{i:03d}.png',
                image=images[i],
                true_mask=true_mask[i],
                pred_mask=pred_mask[i],
                title=f'Slice {i:03d} - Patient {image_id}, DSC {score}')

    def run_test(self, use_best_model=False,
                 masked_images=None,
                 use_original_image=False,
                 contour=True,
                 base_image_name='x',
                 truth_image_name='y',
                 predicted_image_name='predicted',
                 image_name='{index:05d}.png',
                 image_title_name='Image {index:05d}'):

        log_base_path = self.temp_base_path

        test_path = log_base_path + self.TEST_OUTPUT_PATH

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        if use_best_model:
            raise NotImplementedError
        else:
            score = self.model.evaluate_test(verbose=1)
            print(score)
            filepath = test_path + self.PREDICT_TEST_NAME

            self._predict_test(filepath, use_original_image=use_original_image)

            if masked_images:
                self._plot_predicted_images(
                    data_path=filepath,
                    out_path=test_path,
                    images=masked_images,
                    base_image_name=base_image_name,
                    truth_image_name=truth_image_name,
                    predicted_image_name=predicted_image_name,
                    title=image_title_name,
                    contour=contour,
                    name=image_name)

        return self

    def _initialize_post_processors(self, post_processor_class=None,
                                    analysis_base_path='',
                                    map_meta_data='patient_idx,slice_idx',
                                    main_meta_data='', run_test=False,
                                    new_dataset_params=None):
        print("Initializing postprocessor")
        if post_processor_class is None:
            pp = SegmentationPostProcessor(
                self.log_base_path,
                temp_base_path=self.temp_base_path,
                analysis_base_path=analysis_base_path,
                map_meta_data=map_meta_data,
                main_meta_data=main_meta_data, run_test=run_test,
                new_dataset_params=new_dataset_params
            )
        else:
            pp = post_processor_class(
                self.log_base_path,
                temp_base_path=self.temp_base_path,
                analysis_base_path=analysis_base_path,
                map_meta_data=map_meta_data,
                main_meta_data=main_meta_data, run_test=run_test,
                new_dataset_params=new_dataset_params
            )

        return pp

    def apply_post_processors(self, post_processor_class=None,
                              recipe='auto', analysis_base_path='',
                              map_meta_data='patient_idx,slice_idx',
                              main_meta_data='', run_test=False):
        if self.post_processors is None:
            pp = self._initialize_post_processors(
                post_processor_class=post_processor_class,
                analysis_base_path=analysis_base_path,
                map_meta_data=map_meta_data,
                main_meta_data=main_meta_data,
                run_test=run_test
            )
        else:
            pp = self.post_processors
            pp.run_test = run_test

        if type(recipe) == str:
            if recipe == 'auto':
                print('Automatically applying postprocessor '
                      'based on log folder name')
                if '2d' in self.log_base_path:
                    pp_recipe = '2d'
                elif 'patch' in self.log_base_path:
                    pp_recipe = 'patch'
                elif '3d' in self.log_base_path:
                    pp_recipe = '3d'
                else:
                    print('Cannot determine recipe, no postprocessors applied')
            else:
                pp_recipe = recipe

            if pp_recipe == '2d':
                print('Applying postprocesesor to 2d images')
                pp.map_2d_meta_data().calculate_fscore_single(
                ).merge_2d_slice().calculate_fscore()
            elif pp_recipe == 'patch':
                print('Applying postprocesesor to image patches')
                pp.merge_3d_patches().calculate_fscore()
            elif pp_recipe == '3d':
                print('Applying postprocesesor to 3d images')
                pp.map_2d_meta_data().calculate_fscore_single_3d()
            else:
                print('No postprocessors for recipe', pp_recipe)
        elif '__iter__' in dir(recipe):
            print('Running customized recipe.')
            for func_name in recipe:
                try:
                    getattr(pp, func_name)()
                except AttributeError:
                    print(func_name, 'is not implemented in', type(pp))
                except Exception as e:
                    print('Error while calling function '
                          f'{func_name} in {type(pp)}:', e)
        else:
            print('Cannot determine recipe.')

        self.post_processors = pp

        return self

    def load_best_model(self, monitor='', post_processor_class=None,
                        recipe='auto', analysis_base_path='',
                        map_meta_data='patient_idx,slice_idx',
                        main_meta_data='', **kwargs):

        if self.post_processors is None:
            pp = self._initialize_post_processors(
                post_processor_class=post_processor_class,
                analysis_base_path=analysis_base_path,
                map_meta_data=map_meta_data,
                main_meta_data=main_meta_data,
                run_test=False
            )
            self.post_processors = pp
        else:
            pp = self.post_processors

        try:
            path_to_model = pp.get_best_model(monitor, **kwargs)
        except Exception as e:
            print("Error while getting best model:", e)
            print("Apply post processing on validation data first")
            path_to_model = self.apply_post_processors(
                post_processor_class=post_processor_class,
                analysis_base_path=analysis_base_path,
                map_meta_data=map_meta_data,
                main_meta_data=main_meta_data,
                run_test=False
            ).post_processors.get_best_model(monitor, **kwargs)

        print('loading model', path_to_model)

        return self.from_file(path_to_model)

    def plot_3d_test_images(self, best_num=2, worst_num=2):
        if self.post_processors is None:
            print('No post processors to handle this function')
            return self
        pp = self.post_processors
        info = pp.get_best_performance_images_test_set(
            monitor='', best_num=best_num, worst_num=worst_num)

        test_path = self.log_base_path + self.TEST_OUTPUT_PATH
        filename = test_path + self.PREDICT_TEST_NAME

        for image_id, score in zip(info['ids'], info['values']):
            base_image_name = f'x/{image_id}'
            truth_image_name = f'y/{image_id}'
            predicted_image_name = f'predicted/{image_id}'

            images_path_single = test_path + '/' + str(image_id)
            if not os.path.exists(images_path_single):
                os.mkdir(images_path_single)

            with h5py.File(filename, 'r') as f:
                images = f[base_image_name][:]
                true_mask = f[truth_image_name][:]
                pred_mask = f[predicted_image_name][:]

            print('plotting 3d images...', images_path_single)
            mask_prediction(images_path_single + '.html',
                            image=images,
                            true_mask=true_mask,
                            pred_mask=pred_mask,
                            title=f'Patient {image_id}, DSC {score}')
            print('plotting 2d images...')
            for i in range(len(images)):
                mask_prediction(
                    images_path_single + f'/{i:03d}.png',
                    image=images[i],
                    true_mask=true_mask[i],
                    pred_mask=pred_mask[i],
                    title=f'Slice {i:03d} - Patient {image_id}, DSC {score}')

        return self

    def load_new_dataset(self, dataset_filename,
                         monitor='', post_processor_class=None,
                         analysis_base_path='',
                         map_meta_data='patient_idx,slice_idx',
                         main_meta_data='', run_test=False):
        new_dataset_params = load_json_config(dataset_filename)
        if self.post_processors is None:
            pp = self._initialize_post_processors(
                post_processor_class=post_processor_class,
                analysis_base_path=analysis_base_path,
                map_meta_data=map_meta_data,
                main_meta_data=main_meta_data,
                run_test=run_test,
                new_dataset_params=new_dataset_params
            )
            self.post_processors = pp
        else:
            pp = self.post_processors
            pp.run_test = True
            pp.update_data_reader(new_dataset_params)

        self.model._data_reader = pp.data_reader
        if self.model.config:
            self.model.config['dataset_params'] = pp.dataset_params

        return self

    def run_external(self, dataset_filename,
                     monitor='', post_processor_class=None,
                     analysis_base_path='',
                     map_meta_data='patient_idx,slice_idx',
                     main_meta_data=''):
        # load external dataset into model, with run_test=True
        self.load_new_dataset(
            dataset_filename,
            monitor=monitor, post_processor_class=post_processor_class,
            analysis_base_path=analysis_base_path,
            map_meta_data=map_meta_data,
            main_meta_data=main_meta_data, run_test=True
        )

        log_base_path = self.temp_base_path

        test_path = log_base_path + self.TEST_OUTPUT_PATH

        if not os.path.exists(test_path):
            os.makedirs(test_path)

        score = self.model.evaluate_test(verbose=1)
        print(score)
        filepath = test_path + self.PREDICT_TEST_NAME

        self._predict_test(filepath)

        return self
