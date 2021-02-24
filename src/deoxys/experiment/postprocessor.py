from ..loaders import load_data
from ..utils import load_json_config


from deoxys_image.patch_sliding import get_patch_indice

import numpy as np
import h5py
import pandas as pd
import os
import shutil


class H5Metric:
    def __init__(self, ref_file, save_file, metric_name='score',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4,
                 map_file=None, map_column=None):
        self.metric_name = metric_name
        self.ref_file = ref_file

        self.predicted = predicted_dataset
        self.target = target_dataset

        with h5py.File(ref_file, 'r') as f:
            keys = list(f.keys())
        if target_dataset not in keys:
            self.predicted = [f'{key}/{predicted_dataset}' for key in keys]
            self.target = [f'{key}/{target_dataset}' for key in keys]

        self.batch_size = batch_size

        self.res_file = save_file
        self.map_file = map_file
        self.map_column = map_column

    def get_img_batch(self):
        self.scores = []

        if self.map_file is None:
            if type(self.predicted) == str:
                with h5py.File(self.ref_file, 'r') as f:
                    size = f[self.target].shape[0]

                for i in range(0, size, self.batch_size):
                    with h5py.File(self.ref_file, 'r') as f:
                        predicted = f[self.predicted][i:i+self.batch_size]
                        targets = f[self.target][i:i+self.batch_size]
                    yield targets, predicted
            else:
                for pred, target in zip(self.predicted, self.target):
                    with h5py.File(self.ref_file, 'r') as f:
                        size = f[target].shape[0]

                    for i in range(0, size, self.batch_size):
                        with h5py.File(self.ref_file, 'r') as f:
                            predicted = f[pred][i:i+self.batch_size]
                            targets = f[target][i:i+self.batch_size]
                        yield targets, predicted
        else:  # handle 3d with different sizes
            map_df = pd.read_csv(self.map_file)
            map_data = map_df[self.map_column].values

            for idx in map_data:
                with h5py.File(self.ref_file, 'r') as f:
                    predicted = f[self.predicted][str(idx)][:]
                    targets = f[self.target][str(idx)][:]
                yield np.expand_dims(targets, axis=0), np.expand_dims(
                    predicted, axis=0)

    def update_score(self, scores):
        self.scores.extend(scores)

    def save_score(self):
        if os.path.isfile(self.res_file):
            df = pd.read_csv(self.res_file)
            df[f'{self.metric_name}'] = self.scores
        else:
            df = pd.DataFrame(self.scores, columns=[f'{self.metric_name}'])

        df.to_csv(self.res_file, index=False)

    def post_process(self, **kwargs):
        for targets, prediction in self.get_img_batch():
            scores = self.calculate_metrics(
                targets, prediction, **kwargs)
            self.update_score(scores)

        self.save_score()

    def calculate_metrics(targets, predictions, **kwargs):
        raise NotImplementedError


class H5CalculateFScore(H5Metric):
    def __init__(self, ref_file, save_file, metric_name='f1_score',
                 predicted_dataset='predicted',
                 target_dataset='y', batch_size=4, beta=1, threshold=None,
                 map_file=None, map_column=None):
        super().__init__(ref_file, save_file, metric_name,
                         predicted_dataset,
                         target_dataset, batch_size,
                         map_file, map_column)
        self.threshold = 0.5 if threshold is None else threshold
        self.beta = beta

    def calculate_metrics(self, y_true, y_pred, **kwargs):
        assert len(y_true) == len(y_pred), "Shape not match"
        eps = 1e-8
        size = len(y_true.shape)
        reduce_ax = tuple(range(1, size))

        y_pred = (y_pred > self.threshold).astype(y_pred.dtype)

        true_positive = np.sum(y_pred * y_true, axis=reduce_ax)
        target_positive = np.sum(y_true, axis=reduce_ax)
        predicted_positive = np.sum(y_pred, axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive
        fb_denominator = (
            (self.beta ** 2) * target_positive + predicted_positive + eps
        )

        return fb_numerator / fb_denominator


class H5MetaDataMapping:
    def __init__(self, ref_file, save_file, folds, fold_prefix='fold',
                 dataset_names=None):
        self.ref_file = ref_file
        self.save_file = save_file
        if fold_prefix:
            self.folds = ['{}_{}'.format(
                fold_prefix, fold) for fold in folds]
        else:
            self.folds = folds

        self.dataset_names = dataset_names

    def post_process(self, *args, **kwargs):
        data = {dataset_name: [] for dataset_name in self.dataset_names}
        for fold in self.folds:
            with h5py.File(self.ref_file, 'r') as f:
                for dataset_name in self.dataset_names:
                    data[dataset_name].extend(f[fold][dataset_name][:])

        df = pd.DataFrame(data)
        df.to_csv(self.save_file, index=False)


class H5Merge2dSlice:
    def __init__(self, ref_file, map_file, map_column, merge_file, save_file,
                 predicted_dataset='predicted', target_dataset='y',
                 input_dataset='x'):
        self.ref_file = ref_file
        self.map_file = map_file
        self.map_column = map_column
        self.merge_file = merge_file
        self.save_file = save_file

        self.predicted = predicted_dataset
        self.target = target_dataset
        self.inputs = input_dataset

        with h5py.File(ref_file, 'r') as f:
            keys = list(f.keys())
        if input_dataset not in keys:
            self.predicted = [f'{key}/{predicted_dataset}' for key in keys]
            self.target = [f'{key}/{target_dataset}' for key in keys]
            self.inputs = [f'{key}/{input_dataset}' for key in keys]

    def post_process(self):
        map_df = pd.read_csv(self.map_file)
        map_data = map_df[self.map_column].values

        unique_val = []

        first, last = map_data[0], map_data[-1]

        tmp = np.concatenate([[first], map_data, [last]])
        indice = np.where(tmp[1:] != tmp[:-1])[0]
        indice = np.concatenate([[0], indice, [len(map_data)]])

        if type(self.inputs) == str:
            with h5py.File(self.merge_file, 'w') as mf:
                mf.create_group(self.inputs)
                mf.create_group(self.target)
                mf.create_group(self.predicted)

            for i in range(len(indice) - 1):
                start = indice[i]
                end = indice[i+1]

                unique_val.append(map_data[start])

                assert map_data[start] == map_data[end-1], "id not match"

                curr_name = str(map_data[start])
                with h5py.File(self.ref_file, 'r') as f:
                    img = f[self.inputs][start:end]
                with h5py.File(self.merge_file, 'a') as mf:
                    mf[self.inputs].create_dataset(
                        curr_name, data=img, compression="gzip")

                with h5py.File(self.ref_file, 'r') as f:
                    img = f[self.target][start:end]
                with h5py.File(self.merge_file, 'a') as mf:
                    mf[self.target].create_dataset(
                        curr_name, data=img, compression="gzip")

                with h5py.File(self.ref_file, 'r') as f:
                    img = f[self.predicted][start:end]
                with h5py.File(self.merge_file, 'a') as mf:
                    mf[self.predicted].create_dataset(
                        curr_name, data=img, compression="gzip")
        else:
            inputs = self.inputs[0].split('/')[-1]
            target = self.target[0].split('/')[-1]
            predicted = self.predicted[0].split('/')[-1]
            with h5py.File(self.merge_file, 'w') as mf:
                mf.create_group(inputs)
                mf.create_group(target)
                mf.create_group(predicted)

            offset = 0
            curr_data_idx = 0

            with h5py.File(self.ref_file, 'r') as f:
                total = f[self.inputs[curr_data_idx]].shape[0]

            for i in range(len(indice) - 1):
                if indice[i] - offset >= total:
                    offset = indice[i]
                    curr_data_idx += 1

                    with h5py.File(self.ref_file, 'r') as f:
                        total = f[self.inputs[curr_data_idx]].shape[0]

                map_start, map_end = indice[i], indice[i+1]

                start = indice[i] - offset
                end = indice[i+1] - offset

                unique_val.append(map_data[map_start])

                assert map_data[map_start] == map_data[map_end -
                                                       1], "id not match"

                curr_name = str(map_data[map_start])

                with h5py.File(self.ref_file, 'r') as f:
                    img = f[self.inputs[curr_data_idx]][start:end]
                with h5py.File(self.merge_file, 'a') as mf:
                    mf[inputs].create_dataset(
                        curr_name, data=img, compression="gzip")

                with h5py.File(self.ref_file, 'r') as f:
                    img = f[self.target[curr_data_idx]][start:end]
                with h5py.File(self.merge_file, 'a') as mf:
                    mf[target].create_dataset(
                        curr_name, data=img, compression="gzip")

                with h5py.File(self.ref_file, 'r') as f:
                    img = f[self.predicted[curr_data_idx]][start:end]
                with h5py.File(self.merge_file, 'a') as mf:
                    mf[predicted].create_dataset(
                        curr_name, data=img, compression="gzip")

        df = pd.DataFrame(data=unique_val, columns=[self.map_column])
        df.to_csv(self.save_file, index=False)


class H5MergePatches:
    def __init__(self, ref_file, predicted_file,
                 map_column, merge_file, save_file,
                 patch_size, overlap,
                 folds, fold_prefix='fold',
                 original_input_dataset='x',
                 original_target_dataset='y',
                 predicted_dataset='predicted', target_dataset='y',
                 input_dataset='x'
                 ):

        self.ref_file = ref_file
        self.predicted_file = predicted_file
        self.map_column = map_column
        self.merge_file = merge_file
        self.save_file = save_file

        self.ref_inputs = original_input_dataset
        self.ref_targets = original_target_dataset

        self.predicted = predicted_dataset
        self.target = target_dataset
        self.inputs = input_dataset

        if fold_prefix:
            self.folds = ['{}_{}'.format(
                fold_prefix, fold) for fold in folds]
        else:
            self.folds = folds

        self.patch_size = patch_size
        self.overlap = overlap

        print('patch', patch_size)

    def _save_inputs_target_to_merge_file(self, fold, meta, index):
        with h5py.File(self.ref_file, 'r') as f:
            inputs = f[fold][self.ref_inputs][index]
            targets = f[fold][self.ref_targets][index]

        print(inputs.shape, self.inputs, meta)

        with h5py.File(self.merge_file, 'a') as mf:
            mf[self.inputs].create_dataset(
                meta, data=inputs, compression="gzip")
            mf[self.target].create_dataset(
                meta, data=targets, compression="gzip")

    def _merge_patches_to_merge_file(self, meta, start_cursor):
        with h5py.File(self.merge_file, 'r') as mf:
            shape = mf[self.target][meta].shape[:-1]

        # fix patch size
        if '__iter__' not in dir(self.patch_size):
            self.patch_size = [self.patch_size] * len(shape)

        indice = get_patch_indice(shape, self.patch_size, self.overlap)
        next_cursor = start_cursor + len(indice)

        with h5py.File(self.predicted_file, 'r') as f:
            data = f[self.predicted][start_cursor: next_cursor]

        predicted = np.zeros(shape)
        weight = np.zeros(shape)
        print(data.shape, predicted.shape)

        for i in range(len(indice)):
            x, y, z = indice[i]
            w, h, d = self.patch_size
            predicted[x:x+w, y:y+h, z:z+d] = predicted[x:x+w, y:y+h, z:z+d] \
                + data[i][..., 0]
            weight[x:x+w, y:y+h, z:z+d] = weight[x:x+w, y:y+h, z:z+d] \
                + np.ones(self.patch_size)

        predicted = (predicted/weight)[..., np.newaxis]

        with h5py.File(self.merge_file, 'a') as mf:
            mf[self.predicted].create_dataset(
                meta, data=predicted, compression="gzip")

        return next_cursor

    def post_process(self):
        # create merge file
        with h5py.File(self.merge_file, 'w') as mf:
            mf.create_group(self.inputs)
            mf.create_group(self.target)
            mf.create_group(self.predicted)

        data = []
        start_cursor = 0
        for fold in self.folds:
            with h5py.File(self.ref_file, 'r') as f:
                meta_data = f[fold][self.map_column][:]
                data.extend(meta_data)
                for index, meta in enumerate(meta_data):
                    self._save_inputs_target_to_merge_file(
                        fold, str(meta), index)
                    start_cursor = self._merge_patches_to_merge_file(
                        str(meta), start_cursor)

        # create map file
        df = pd.DataFrame(data, columns=[self.map_column])
        df.to_csv(self.save_file, index=False)


class PostProcessor:
    MODEL_PATH = '/model'
    MODEL_NAME = '/model.{epoch:03d}.h5'
    BEST_MODEL_PATH = '/best'
    PREDICTION_PATH = '/prediction'
    PREDICTION_NAME = '/prediction.{epoch:03d}.h5'
    LOG_FILE = '/logs.csv'
    PERFORMANCE_PATH = '/performance'
    PREDICTED_IMAGE_PATH = '/images'
    TEST_OUTPUT_PATH = '/test'
    PREDICT_TEST_NAME = '/prediction_test.h5'
    SINGLE_MAP_PATH = '/single_map'
    SINGLE_MAP_NAME = '/logs.{epoch:03d}.csv'

    MAP_PATH = '/logs'
    MAP_NAME = '/logs.{epoch:03d}.csv'

    TEST_SINGLE_MAP_NAME = '/single_result.csv'
    TEST_MAP_NAME = '/result.csv'

    def __init__(self, log_base_path='logs',
                 temp_base_path='',
                 analysis_base_path='',
                 map_meta_data=None, main_meta_data='',
                 run_test=False):
        self.temp_base_path = temp_base_path
        self.log_base_path = log_base_path
        self.analysis_base_path = analysis_base_path or log_base_path

        if not os.path.exists(self.analysis_base_path):
            os.mkdir(self.analysis_base_path)

        if not os.path.exists(self.analysis_base_path + self.PREDICTION_PATH):
            os.mkdir(self.analysis_base_path + self.PREDICTION_PATH)

        model_path = log_base_path + self.MODEL_PATH

        sample_model_filename = model_path + '/' + os.listdir(model_path)[0]

        with h5py.File(sample_model_filename, 'r') as f:
            config = f.attrs['deoxys_config']
            config = load_json_config(config)

        self.dataset_filename = config['dataset_params']['config']['filename']
        self.data_reader = load_data(config['dataset_params'])

        self.temp_prediction_path = temp_base_path + self.PREDICTION_PATH
        predicted_files = os.listdir(self.temp_prediction_path)

        self.epochs = [int(filename[-6:-3]) for filename in predicted_files]

        if map_meta_data:
            if type(map_meta_data) == str:
                self.map_meta_data = map_meta_data.split(',')
            else:
                self.map_meta_data = map_meta_data
        else:
            self.map_meta_data = ['patient_idx', 'slice_idx']

        if main_meta_data:
            self.main_meta_data = main_meta_data
        else:
            self.main_meta_data = self.map_meta_data[0]

        self.run_test = run_test

    def map_2d_meta_data(self):
        print('mapping 2d meta data')
        if not self.run_test:
            map_folder = self.log_base_path + self.SINGLE_MAP_PATH

            if not os.path.exists(map_folder):
                os.makedirs(map_folder)
            map_filename = map_folder + self.SINGLE_MAP_NAME

            for epoch in self.epochs:
                H5MetaDataMapping(
                    ref_file=self.dataset_filename,
                    save_file=map_filename.format(epoch=epoch),
                    folds=self.data_reader.val_folds,
                    fold_prefix='',
                    dataset_names=self.map_meta_data).post_process()
        else:
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            map_filename = test_folder + self.TEST_SINGLE_MAP_NAME
            H5MetaDataMapping(
                ref_file=self.dataset_filename,
                save_file=map_filename,
                folds=self.data_reader.test_folds,
                fold_prefix='',
                dataset_names=self.map_meta_data).post_process()

        return self

    def calculate_fscore_single(self):
        print('calculating dice score per items in val set')
        if not self.run_test:
            predicted_path = self.temp_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME
            map_folder = self.log_base_path + self.SINGLE_MAP_PATH
            map_filename = map_folder + self.SINGLE_MAP_NAME
            for epoch in self.epochs:
                H5CalculateFScore(
                    predicted_path.format(epoch=epoch),
                    map_filename.format(epoch=epoch)
                ).post_process()
        else:
            predicted_path = self.temp_base_path + \
                self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            map_filename = test_folder + self.TEST_SINGLE_MAP_NAME

            H5CalculateFScore(
                predicted_path,
                map_filename
            ).post_process()

        return self

    def calculate_fscore_single_3d(self):
        self.calculate_fscore_single()
        if not self.run_test:
            map_folder = self.log_base_path + self.SINGLE_MAP_PATH

            main_log_folder = self.log_base_path + self.MAP_PATH
            os.rename(map_folder, main_log_folder)
        else:
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            map_filename = test_folder + self.TEST_SINGLE_MAP_NAME

            main_result_file_name = test_folder + self.TEST_MAP_NAME

            os.rename(map_filename, main_result_file_name)

    def merge_2d_slice(self):
        print('merge 2d slice to 3d images')
        if not self.run_test:
            predicted_path = self.temp_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME
            map_folder = self.log_base_path + self.SINGLE_MAP_PATH
            map_filename = map_folder + self.SINGLE_MAP_NAME

            merge_path = self.log_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME

            main_log_folder = self.log_base_path + self.MAP_PATH

            if not os.path.exists(main_log_folder):
                os.makedirs(main_log_folder)
            main_log_filename = main_log_folder + self.MAP_NAME

            for epoch in self.epochs:
                H5Merge2dSlice(
                    predicted_path.format(epoch=epoch),
                    map_filename.format(epoch=epoch),
                    self.main_meta_data,
                    merge_path.format(epoch=epoch),
                    main_log_filename.format(epoch=epoch)
                ).post_process()
        else:
            predicted_path = self.temp_base_path + \
                self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            map_filename = test_folder + self.TEST_SINGLE_MAP_NAME
            merge_path = test_folder + self.PREDICT_TEST_NAME
            main_result_file_name = test_folder + self.TEST_MAP_NAME

            H5Merge2dSlice(
                predicted_path,
                map_filename,
                self.main_meta_data,
                merge_path,
                main_result_file_name
            ).post_process()

        return self

    def merge_3d_patches(self):
        print('merge 3d patches to 3d images')
        if not self.run_test:
            predicted_path = self.temp_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME
            # map_folder = self.log_base_path + self.SINGLE_MAP_PATH
            # map_filename = map_folder + self.SINGLE_MAP_NAME

            merge_path = self.analysis_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME

            main_log_folder = self.log_base_path + self.MAP_PATH

            if not os.path.exists(main_log_folder):
                os.makedirs(main_log_folder)
            main_log_filename = main_log_folder + self.MAP_NAME

            for epoch in self.epochs:
                H5MergePatches(
                    ref_file=self.dataset_filename,
                    predicted_file=predicted_path.format(epoch=epoch),
                    map_column=self.main_meta_data,
                    merge_file=merge_path.format(epoch=epoch),
                    save_file=main_log_filename.format(epoch=epoch),
                    patch_size=self.data_reader.patch_size,
                    overlap=self.data_reader.overlap,
                    folds=self.data_reader.val_folds,
                    fold_prefix='',
                    original_input_dataset=self.data_reader.x_name,
                    original_target_dataset=self.data_reader.y_name,
                ).post_process()
        else:
            predicted_path = self.temp_base_path + \
                self.TEST_OUTPUT_PATH + self.PREDICT_TEST_NAME
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            merge_path = test_folder + self.PREDICT_TEST_NAME
            main_result_file_name = test_folder + self.TEST_MAP_NAME

            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            H5MergePatches(
                ref_file=self.dataset_filename,
                predicted_file=predicted_path,
                map_column=self.main_meta_data,
                merge_file=merge_path,
                save_file=main_result_file_name,
                patch_size=self.data_reader.patch_size,
                overlap=self.data_reader.overlap,
                folds=self.data_reader.val_folds,
                fold_prefix='',
                original_input_dataset=self.data_reader.x_name,
                original_target_dataset=self.data_reader.y_name,
            ).post_process()

        return self

    def calculate_fscore(self):
        print('calculating dice score per 3d image')
        if not self.run_test:
            merge_path = self.analysis_base_path + \
                self.PREDICTION_PATH + self.PREDICTION_NAME

            main_log_folder = self.log_base_path + self.MAP_PATH
            main_log_filename = main_log_folder + self.MAP_NAME

            for epoch in self.epochs:
                H5CalculateFScore(
                    merge_path.format(epoch=epoch),
                    main_log_filename.format(epoch=epoch),
                    map_file=main_log_filename.format(epoch=epoch),
                    map_column=self.main_meta_data
                ).post_process()
        else:
            test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
            merge_path = test_folder + self.PREDICT_TEST_NAME
            main_result_file_name = test_folder + self.TEST_MAP_NAME

            H5CalculateFScore(
                merge_path,
                main_result_file_name,
                map_file=main_result_file_name,
                map_column=self.main_meta_data
            ).post_process()

        return self

    def get_best_model(self, monitor='', keep_best_only=True):
        print('finding best model')

        epochs = self.epochs

        res_df = pd.DataFrame(epochs, columns=['epochs'])

        results = []
        results_path = self.log_base_path + self.MAP_PATH + self.MAP_NAME

        for epoch in epochs:
            df = pd.read_csv(results_path.format(epoch=epoch))
            if not monitor:
                monitor = df.columns[-1]

            results.append(df[monitor].mean())

        res_df[monitor] = results
        best_epoch = epochs[res_df[monitor].argmax()]

        res_df.to_csv(self.log_base_path + '/log_new.csv', index=False)

        if keep_best_only:
            for epoch in epochs:
                if epoch != best_epoch:
                    os.remove(self.analysis_base_path + self.PREDICTION_PATH +
                              self.PREDICTION_NAME.format(epoch=epoch))
                elif self.log_base_path != self.analysis_base_path:
                    # move the best prediction to main folder
                    shutil.copy(
                        self.analysis_base_path + self.PREDICTION_PATH +
                        self.PREDICTION_NAME.format(epoch=epoch),
                        self.log_base_path + self.PREDICTION_PATH +
                        self.PREDICTION_NAME.format(epoch=epoch))

                    os.remove(self.analysis_base_path + self.PREDICTION_PATH +
                              self.PREDICTION_NAME.format(epoch=epoch))
        elif self.log_base_path != self.analysis_base_path:
            # Copy the best prediction to the main folder
            shutil.copy(self.analysis_base_path + self.PREDICTION_PATH +
                        self.PREDICTION_NAME.format(epoch=best_epoch),
                        self.log_base_path + self.PREDICTION_PATH +
                        self.PREDICTION_NAME.format(epoch=best_epoch))

        return self.log_base_path + self.MODEL_PATH + \
            self.MODEL_NAME.format(epoch=best_epoch)

    def get_best_performance_images(self, monitor='', best_num=2, worst_num=2):
        epochs = self.epochs
        results_path = self.log_base_path + self.MAP_PATH + self.MAP_NAME

        results = []
        for epoch in epochs:
            # only plot things in prediction
            if os.path.exists(self.log_base_path + self.PREDICTION_PATH +
                              self.PREDICTION_NAME.format(epoch=epoch)):
                df = pd.read_csv(results_path.format(epoch=epoch))

                if not monitor:
                    monitor = df.columns[-1]
                largest_indice = df[monitor].nlargest(best_num, keep='all')
                smallest_indice = df[monitor].nlargest(
                    worst_num, keep='all')

                indice = list(largest_indice.index) + \
                    list(smallest_indice.index)

            results.append(
                {'file_name': self.PREDICTION_NAME.format(epoch=epoch),
                 'ids': df.values[indice][:, 0],
                 'values': df.values[indice][:, 1]})

        return results

    def get_best_performance_images_test_set(
            self, monitor='', best_num=2, worst_num=2):

        test_folder = self.log_base_path + self.TEST_OUTPUT_PATH
        main_result_file_name = test_folder + self.TEST_MAP_NAME

        df = pd.read_csv(main_result_file_name)

        if not monitor:
            monitor = df.columns[-1]
        largest_indice = df[monitor].nlargest(best_num, keep='all')
        smallest_indice = df[monitor].nlargest(
            worst_num, keep='all')

        indice = list(largest_indice.index) + \
            list(smallest_indice.index)

        return {'ids': df.values[indice][:, 0],
                'values': df.values[indice][:, 1]}
