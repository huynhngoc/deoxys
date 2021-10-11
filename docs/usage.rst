=====
Usage
=====

To use deoxys in a project::

    import deoxys

Table of contents
=================

.. contents::


Quickstart
==========
Deep learning for auto-segmentation tasks
---------------------------------------------

TBD

Other deep learning tasks
---------------------------------------------
TBD

Configure and run experiment
============================

The dataset files
-----------------

The HDF5 file format
^^^^^^^^^^^^^^^^^^^^
Current, the *deoxys* framework only support files in HDF5 format for ease of compression and data transfer. You can read more about the format `here <https://portal.hdfgroup.org/display/HDF5/HDF5>`__.

If you do not want to know everything about HDF5 format in depth, here is an important quote from the documentation.

    HDF5 files are organized in a hierarchical structure, with two primary structures: groups and datasets.

    * HDF5 group: a grouping structure containing instances of zero or more groups or datasets, together with supporting metadata.
    * HDF5 dataset: a multidimensional array of data elements, together with supporting metadata.

The *HDF5 dataset* contains the data, while the *HDF5 group* contains the *HDF datasets*.

To check the content of your HDF5 file, you can use the following python script
::

    import h5py

    def print_detail(file_name):
        with h5py.File(filename, 'r') as f:
            for group in f.keys():
                print(group)
                for ds_name in f[group].keys():
                    print('--', ds_name, f[group][ds_name].shape)

    print_detail(insert_path_to_your_dataset_file_here)


Structure requirements
^^^^^^^^^^^^^^^^^^^^^^^
Your dataset file must satisfy the following requirements to be compatible with the deoxys framework.

#. There are exactly two levels of hierarchy in your file.
#. The top level structures of your hdf5 file must always be *HDF5 groups*, containing two or more *HDF5 datasets*
#. The *HDF5 dataset*'s names must be the same in every group. Please check the examples in the next part
#. The dimensions of the *HDF5 datasets* with the same name in different *HDF5 groups* should match, except for the first dimension. (Check the example)

The train - val - test structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    my_dataset.h5 # your dataset file, in HDF5 format
    ├─ train # the group name
    |   ├── input  # name of the dataset containing the input data - shape of (1500, 128, 128, 3)
    |   ├── target # name of the dataset containing the label data - shape of (1500, 128, 128, 1)
    |   ├── meta1  # name of the dataset containing the meta data - shape of (1500, )
    |   ├── meta2  #
    |   └── (other meta data)
    ├─ val # the group name
    |   ├── input  # name of the dataset containing the input data - shape of (200, 128, 128, 3)
    |   ├── target # name of the dataset containing the label data - shape of (200, 128, 128, 1)
    |   ├── meta1  # name of the dataset containing the meta data - shape of (200, )
    |   ├── meta2  #
    |   └── (other meta data)
    └─ test # the group name
        ├── input  # name of the dataset containing the input data - shape of (500, 128, 128, 3)
        ├── target # name of the dataset containing the label data - shape of (500, 128, 128, 1)
        ├── meta1  # name of the dataset containing the meta data - shape of (500, )
        ├── meta2  #
        └── (other meta data)

The kfold validation structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    my_kfold_dataset.h5
    ├─ fold_0 # the group name
    |   ├── input  # name of the dataset containing the input data - shape of (200, 128, 128, 3)
    |   ├── target # name of the dataset containing the label data - shape of (200, 128, 128, 1)
    |   ├── meta1  # name of the dataset containing the meta data - shape of (200, )
    |   ├── meta2  #
    |   └── (other meta data)
    ├─ fold_1 # the group name
    |   ├── input  # name of the dataset containing the input data - shape of (200, 128, 128, 3)
    |   ├── target # name of the dataset containing the label data - shape of (200, 128, 128, 1)
    |   ├── meta1  # name of the dataset containing the meta data - shape of (200, )
    |   ├── meta2  #
    |   └── (other meta data)
    ├─ fold_2 # the group name
    |   ├── input  # name of the dataset containing the input data - shape of (200, 128, 128, 3)
    |   ├── target # name of the dataset containing the label data - shape of (200, 128, 128, 1)
    |   ├── meta1  # name of the dataset containing the meta data - shape of (200, )
    |   ├── meta2  #
    |   └── (other meta data)
    ├─ fold_3 # the group name
    |   ├── input  # name of the dataset containing the input data - shape of (200, 128, 128, 3)
    |   ├── target # name of the dataset containing the label data - shape of (200, 128, 128, 1)
    |   ├── meta1  # name of the dataset containing the meta data - shape of (200, )
    |   ├── meta2  #
    |   └── (other meta data)
    ├─ fold_4 # the group name
    |   ├── input  # name of the dataset containing the input data - shape of (200, 128, 128, 3)
    |   ├── target # name of the dataset containing the label data - shape of (200, 128, 128, 1)
    |   ├── meta1  # name of the dataset containing the meta data - shape of (200, )
    |   ├── meta2  #
    |   └── (other meta data)
    ├─ fold_5 # the group name
    |   ├── input  # name of the dataset containing the input data - shape of (200, 128, 128, 3)
    |   ├── target # name of the dataset containing the label data - shape of (200, 128, 128, 1)
    |   ├── meta1  # name of the dataset containing the meta data - shape of (200, )
    |   ├── meta2  #
    |   └── (other meta data)
    ├─ fold_6 # the group name
    |   ├── input  # name of the dataset containing the input data - shape of (200, 128, 128, 3)
    |   ├── target # name of the dataset containing the label data - shape of (200, 128, 128, 1)
    |   ├── meta1  # name of the dataset containing the meta data - shape of (200, )
    |   ├── meta2  #
    |   └── (other meta data)
    └─ fold_7 # the group name
        ├── input  # name of the dataset containing the input data - shape of (200, 128, 128, 3)
        ├── target # name of the dataset containing the label data - shape of (200, 128, 128, 1)
        ├── meta1  # name of the dataset containing the meta data - shape of (200, )
        ├── meta2  #
        └── (other meta data)


Create your own dataset
^^^^^^^^^^^^^^^^^^^^^^^
In the case you are not provided with a prepared dataset file, or you want to customize your dataset, here is an example python script to create your own dataset
::

    import h5py

    # First gather your data as np.array
    # get_train_data, get_val_data and get_test_data are just example code for you to understand to process
    # meta_data can be the patient IDs, the slide IDs or any other information about the data
    train_X, train_y, train_pids = get_train_data()
    val_X, val_y, val_pids = get_val_data()
    test_X, test_y, test_pids = get_test_data()

    # Next get the shape of your data
    dim1, dim2, num_channel = train_X.shape[1:]

    # Finally create your file
    with h5py.File(filename, 'a') as f:
        train_group = f.create_group('train')
        train_group.create_dataset('x', data=train_X, dtype='f4',
                                    chunks=(1, img_dim1, img_dim2, num_channel),
                                    compression='lzf')
        train_group.create_dataset('y', data=train_y, dtype='f4',
                                    chunks=(1, img_dim1, img_dim2, 1),
                                    compression='lzf')
        train_group.create_dataset('patient_idx', data=train_pids)

        val_group = f.create_group('val')
        val_group.create_dataset('x', data=val_X, dtype='f4',
                                    chunks=(1, img_dim1, img_dim2, num_channel),
                                    compression='lzf')
        val_group.create_dataset('y', data=val_y, dtype='f4',
                                    chunks=(1, img_dim1, img_dim2, 1),
                                    compression='lzf')
        val_group.create_dataset('patient_idx', data=val_pids)

        test_group = f.create_group('test')
        test_group.create_dataset('x', data=test_X, dtype='f4',
                                    chunks=(1, img_dim1, img_dim2, num_channel),
                                    compression='lzf')
        test_group.create_dataset('y', data=test_y, dtype='f4',
                                    chunks=(1, img_dim1, img_dim2, 1),
                                    compression='lzf')
        test_group.create_dataset('patient_idx', data=test_pids)

In the case you want to create a kfold structure
::

    import h5py

    # First define a function to gather your data and split your data into folds
    def get_fold(index):
        # process your data here
        return X, y, pids

    # Either hard-code these values or use the first fold to get these values
    dim1, dim2, num_channel = get_fold(0)[0][1:]

    # Loop through your data and create your dataset
    for i in range(num_folds):
        with h5py.File(filename, 'a') as f:
            group = f.create_group(f'fold_{i}')
            data_x, data_y, pids = get_fold(i)
            group.create_dataset('x', data=data_x, dtype='f4',
                                    chunks=(1, img_dim1, img_dim2, num_channel),
                                    compression='lzf')
            group.create_dataset('y', data=data_y, dtype='f4',
                                    chunks=(1, img_dim1, img_dim2, 1),
                                    compression='lzf')
            group.create_dataset('patient_idx', data=pids)

The configurable JSON file
---------------------------
The basic configurable JSON object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
All of the objects in the JSON configuration follows this structure:

.. code-block:: JSON

    {
        "class_name": "ClassName0",
        "config": {
            "param1": "value1",
            "param2": "value2"
        }
    }


The above configuration tells the configuration loader to create an instance of `ClassName0`, using `params` in the config as arguments in the constructor function.::
    request_object = ClassName0(param1=value1, param2=value2)


Class names can be found in https://deoxys.readthedocs.io/en/latest/modules.html and https://keras.io/api/

The JSON configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The configuration file should contains the following 5 objects: `dataset_params`, `train_params`, `input_params`, `model_params`, and `architecture`

.. code-block:: JSON

    {
        "dataset_params": {
        },
        "train_params": {
        },
        "input_params": {
        },
        "model_params": {
        },
        "architecture": {
        }
    }



* ``dataset_params``: contains the configuration for the datareader object, (check the list of DataReaders `here <https://deoxys.readthedocs.io/en/latest/data.html#module-deoxys.data.data_reader>`__. It is recommended that you use the `H5Reader`
* ``input_params``: put the required parameters for the `Input layer <https://keras.io/api/layers/core_layers/input/>`__ here, usually, the shape of the input image
* ``model_params``: put the required parameters for the ``compile`` function of the `model <https://keras.io/api/models/model_training_apis/>`__ in here. Most of the time, you only need to define:

    * the ``optimizer``: either str or JSON object, check the list of `Optimizers <https://keras.io/api/optimizers/#core-optimizer-api>`_
    * the ``loss`` function: either str or JSON object, check the list of Loss functions, in `keras <https://keras.io/api/losses/#available-losses>`__ and in in `deoxys <https://deoxys.readthedocs.io/en/latest/model.html#module-deoxys.model.losses>`__
    * the ``metrics`` list: list of str or JSON objects, check the list of Metrics, in `keras <https://keras.io/api/metrics/#available-metrics>`__ and in `deoxys <https://deoxys.readthedocs.io/en/latest/model.html#module-deoxys.model.metrics>`__
* ``train_params``: put the parameters for the `fit` function of the Model in here. Most of the time, you only need to define the list of ``callbacks``, check the list callbacks in `keras <https://keras.io/api/callbacks/#available-callbacks>`__ and in `deoxys <https://deoxys.readthedocs.io/en/latest/model.html#module-deoxys.model.callbacks>`__.

    Note that number of epoch, x and y params, as well as callbacks for logging the performance and save models/prediction are already handled while you run your experiment. You should use callbacks relating to stopping the model (EarlyStopping) or changing the learning rate (ReduceLROnPlateau) here.
* ``architecture``: configure your architecture here. You should create the architecture using the helper functions. Then modify the resulting JSON (For example, adding more layers to the base architecture).

You can look at the example configuration (`config/2d_unet_CT_W_PET.json`) to understand how it works.

Configure the H5Reader JSON object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
General information
"""""""""""""""""""
First, put the class name and the config object into the `dataset_params` object

.. code-block:: JSON

    {
        "dataset_params": {
            "class_name": "H5Reader",
            "config": {}
        },
    }


Next, define the basic information:

* ``filename``: path to the dataset file, either relative path or absolute path
* ``x_name``: name of the HDF5 dataset acts as the inputs.
* ``y_name``: name of the HDF5 dataset acts as the labels.
* ``batch_size``: the size of the training batch
*  ``batch_cache``: number of batches to be ready in your RAM
*  ``shuffle``: should be true

.. code-block:: JSON

    {
        "dataset_params": {
            "class_name": "H5Reader",
            "config": {
                "filename": "../../full_dataset_singleclass.h5",
                "x_name": "input",
                "y_name": "target",
                "batch_size": 2,
                "batch_cache": 1,
                "shuffle": true,
            }
        },
    }

Split into train, validation and test
""""""""""""""""""""""""""""""""""""""
Depending on the structure of your data, set fold_prefix, train_folds, val_folds, and test_folds accordingly.

If your dataset file is in train, val, test structure

.. code-block:: JSON

    {
        "dataset_params": {
            "class_name": "H5Reader",
            "config": {
                "filename": "../../full_dataset_singleclass.h5",
                "x_name": "input",
                "y_name": "target",
                "batch_size": 2,
                "batch_cache": 1,
                "shuffle": true,
                "fold_prefix": "",
                "train_folds": [
                    "train"
                ],
                "val_folds": [
                    "val"
                ],
                "test_folds": [
                    "test"
                ],
            }
        },
    }


If your dataset file supports cross-validation:

* First determine the prefix of each fold, usually `fold`
* Next, determine which fold to be in the trains/validation or test

In the case your dataset file contains 7 folds, and you want to put the last 2 folds as test dataset, while the remaining folds are used for cross-validation:

.. code-block:: JSON

    {
        "dataset_params": {
            "class_name": "H5Reader",
            "config": {
                "filename": "../../full_dataset_singleclass.h5",
                "x_name": "input",
                "y_name": "target",
                "batch_size": 2,
                "batch_cache": 1,
                "shuffle": true,
                "fold_prefix": "fold",
                "train_folds": [
                    0, 1, 2, 3, 4
                ],
                "val_folds": [
                    5
                ],
                "test_folds": [
                    6, 7
                ],
            }
        },
    }

Alternatively, if you want to validate on a different fold. Note that the `test_folds` list won't change.

.. code-block:: JSON

    {
        "dataset_params": {
            "class_name": "H5Reader",
            "config": {
                "filename": "../../full_dataset_singleclass.h5",
                "x_name": "input",
                "y_name": "target",
                "batch_size": 2,
                "batch_cache": 1,
                "shuffle": true,
                "fold_prefix": "fold",
                "train_folds": [
                    0, 1, 2, 3, 5
                ],
                "val_folds": [
                    4
                ],
                "test_folds": [
                    6, 7
                ],
            }
        },
    }

Put the preprocessors in place
""""""""""""""""""""""""""""""
Next, put the list of necessary preprocessors. The preprocessors will apply in the order of the list. Check the list of preprocessors in `here <https://deoxys.readthedocs.io/en/latest/data.html#module-deoxys.data.preprocessor>`_.

For example, if you want apply windowing to the CT channel of your PET/CT images (which is the first channel) with `width=200`, `center=70`, then normalize the CT channel within the range between `[-100, 100]` and the PET channel within the range between `[0, 25]`.

.. code-block:: JSON

    {
        "dataset_params": {
            "class_name": "H5Reader",
            "config": {
                "filename": "../../full_dataset_singleclass.h5",
                "x_name": "input",
                "y_name": "target",
                "batch_size": 2,
                "batch_cache": 1,
                "shuffle": true,
                "fold_prefix": "",
                "train_folds": [
                    "train"
                ],
                "val_folds": [
                    "val"
                ],
                "test_folds": [
                    "test"
                ],
                "preprocessors": [
                    {
                        "class_name": "HounsfieldWindowingPreprocessor",
                        "config": {
                            "window_center": 70,
                            "window_width": 200,
                            "channel": 0
                        }
                    },
                    {
                        "class_name": "ImageNormalizerPreprocessor",
                        "config": {
                            "vmin": [
                                -100,
                                0
                            ],
                            "vmax": [
                                100,
                                25
                            ]
                        }
                    }
                ],
            }
        },
    }

**Tips for choosing the vmin and vmax values**:

* If you leave the vmin and vmax empty (no configuration for vmin and vmax), or ``"vmin":null`` and ``"vmax":null``, the ``ImageNormalizerPreprocessor`` will automatically normalize the images based on the minimum and maximum intensity values of each channel.
* If you are working on PET/CT images, and you applies windowing it is suggest that you use the vmin, vmax values for the CT channel half the window width (in the case ``window_width=200``, ``vmin``, ``vmax`` should be `-100` and `100` respectively), and set vmin, vmax for PET channel to 0 and 25 (we will treat any numbers larger than 25 as 25)

In another example, you want to remove the second channel in your image, then normalize the image

.. code-block:: JSON

    {
        "dataset_params": {
            "class_name": "H5Reader",
            "config": {
                "filename": "../../full_dataset_singleclass.h5",
                "x_name": "input",
                "y_name": "target",
                "batch_size": 2,
                "batch_cache": 1,
                "shuffle": true,
                "fold_prefix": "",
                "train_folds": [
                    "train"
                ],
                "val_folds": [
                    "val"
                ],
                "test_folds": [
                    "test"
                ],
                "preprocessors": [
                    {
                        "class_name": "ChannelRemoval",
                        "config": {
                            "channel": 1
                        }
                    },
                    {
                        "class_name": "ImageNormalizerPreprocessor",
                        "config": {}
                    }
                ],
            }
        },
    }

Configure image augmentation
""""""""""""""""""""""""""""
If you do not want image augmentation in your dataset, simply put an empty list to the `augmentations` object. Now the datareader is ready.

.. code-block:: JSON

    {
        "dataset_params": {
            "class_name": "H5Reader",
            "config": {
                "filename": "../../full_dataset_singleclass.h5",
                "x_name": "input",
                "y_name": "target",
                "batch_size": 2,
                "batch_cache": 1,
                "shuffle": true,
                "fold_prefix": "",
                "train_folds": [
                    "train"
                ],
                "val_folds": [
                    "val"
                ],
                "test_folds": [
                    "test"
                ],
                "preprocessors": [
                    {
                        "class_name": "ImageNormalizerPreprocessor",
                        "config": {}
                    }
                ],
                "augmentations": []
            }
        },
    }


You can follow the documentation `<https://deoxys.readthedocs.io/en/latest/data.html#deoxys.data.preprocessor.ImageAugmentation2D>`_ to configure different augmentation options that can apply to your images.

.. code-block:: JSON

    {
        "dataset_params": {
            "class_name": "H5Reader",
            "config": {
                "filename": "../../full_dataset_singleclass.h5",
                "x_name": "input",
                "y_name": "target",
                "batch_size": 2,
                "batch_cache": 1,
                "shuffle": true,
                "fold_prefix": "",
                "train_folds": [
                    "train"
                ],
                "val_folds": [
                    "val"
                ],
                "test_folds": [
                    "test"
                ],
                "preprocessors": [
                    {
                        "class_name": "ImageNormalizerPreprocessor",
                        "config": {}
                    }
                ],
                "augmentations": [{
                    "class_name": "ImageAugmentation2D",
                    "config": {
                        "rotation_range": 90,
                        "rotation_chance ": 0.5,
                        "zoom_range": [
                            0.8,
                            1.2
                        ],
                        "shift_range": [
                            10,
                            10
                        ],
                        "flip_axis": 0,
                        "brightness_range": [
                            0.8,
                            1.2
                        ],
                        "contrast_range": [
                            0.7,
                            1.3
                        ],
                        "noise_variance": 0.05,
                        "noise_channel": 1,
                        "blur_range": [
                            0.5,
                            1.5
                        ],
                        "blur_channel": 1
                    }
                }]
            }
        },
    }


Configure `input_params`
^^^^^^^^^^^^^^^^^^^^^^^^
You should put the shape of your images **after** preprocessing in here.

.. code-block:: JSON

    {
        "dataset_params": {
            "class_name": "ReaderClassName",
            "config": {
                "param1": "value1",
                "param2": "value2"
            }
        },
        "input_params": {
            "shape": [
                191,
                265,
                2
            ]
        },
    }

Check the content of your hdf5 file to get the exact shape. **Note: remember to remove the number of items (the first number).**


Running your experiments
--------------------------------------------
When you have your dataset files and configuration files ready, you can run the experiments
::

    from deoxys.experiment import Experiment

    if __name__ == '__main__':

        Experiment(
            log_base_path='path_to_log_folder'
        ).from_full_config(
            'path_to_config_file'
        ).run_experiment(
            train_history_log=True,
            model_checkpoint_period=1, # interval of epochs to make a copy of the model
            prediction_checkpoint_period=1, # interval of epochs to save validation results
            epochs=50 # number of epochs to run
        ).plot_performance().plot_prediction(
            masked_images=[i for i in range(42)] # plot 42 first images in validation set
        )


You can also parameterize the log folder, config file path and number of epochs.
::

    # filename: experiment.py

    from deoxys.experiment import Experiment
    import argparse

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("config_file")
        parser.add_argument("log_folder")

        parser.add_argument("--epochs", default=50, type=int)
        parser.add_argument("--model_checkpoint_period", default=5, type=int)
        parser.add_argument("--prediction_checkpoint_period", default=5, type=int)


        args, unknown = parser.parse_known_args()

        print('training from configuration', args.config_file,
            'and saving log files to', args.log_folder)

        exp = Experiment(
            log_base_path=args.log_folder
        ).from_full_config(
            args.config_file
        ).run_experiment(
            train_history_log=True,
            model_checkpoint_period=args.model_checkpoint_period,
            prediction_checkpoint_period=args.prediction_checkpoint_period,
            epochs=args.epochs,
        ).plot_performance().plot_prediction(
            masked_images=[i for i in range(42)] # plot 42 first images in validation set
        )


Run the file in terminal

.. code-block:: bash

    python experiment.py path_to_config_file path_to_log_folder --epochs 50 --model_checkpoint_period 10 --prediction_checkpoint_period 10


This is an easy way to test your deep learning models with different configurations, or different image modalities, or different preprocessing methods.


Running your experiment pipeline
--------------------------------------------
If you are running the experiments a virtual machine or HPC system, it's best to use the experiment pipeline with post-processing.
The log folder will then contain the Dice score of each patient in the validation set in each training interval (``prediction_checkpoint_period``)

::

    # filename: experiment.py
    # some naming convention to setup the post-processing type
    # if log folder contains '2d', 2d post-processors will run after the experiments
    # if log folder contains '3d', 3d post-processors will run after the experiments
    # if log folder contains 'patch', 3d patches post-processors will run after the experiments
    # the meta data in this experiment is the patient id
    # anything in the temp folder and analyis folder are safe to delete after the experiment finished

    from deoxys.experiment import ExperimentPipeline
    import argparse

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("config_file")
        parser.add_argument("log_folder")
        parser.add_argument("--temp_folder", default='', type=str)
        parser.add_argument("--analysis_folder",
                            default='', type=str)
        parser.add_argument("--epochs", default=200, type=int)
        parser.add_argument("--model_checkpoint_period", default=5, type=int)
        parser.add_argument("--prediction_checkpoint_period", default=5, type=int)
        parser.add_argument("--meta", default='patient_idx', type=str)
        parser.add_argument("--monitor", default='', type=str)

        args, unknown = parser.parse_known_args()

        if 'patch' in args.log_folder:
            analysis_folder = args.analysis_folder
        else:
            analysis_folder = ''

        if '2d' in args.log_folder:
            meta = args.meta
        else:
            meta = args.meta.split(',')[0]

        print('training from configuration', args.config_file,
            'and saving log files to', args.log_folder)
        print('Unprocesssed prediciton are saved to', args.temp_folder)
        if analysis_folder:
            print('Intermediate processed files for merging patches are saved to',
                analysis_folder)

        exp = ExperimentPipeline(
            log_base_path=args.log_folder,
            temp_base_path=args.temp_folder
        ).from_full_config(
            args.config_file
        ).run_experiment(
            train_history_log=True,
            model_checkpoint_period=args.model_checkpoint_period,
            prediction_checkpoint_period=args.prediction_checkpoint_period,
            epochs=args.epochs,
        ).apply_post_processors(
            recipe='auto',
            analysis_base_path=analysis_folder,
            map_meta_data=meta,
        ).plot_performance().plot_prediction(
            masked_images=[], best_num=2, worst_num=2
        ).load_best_model(monitor=args.monitor)
        if analysis_folder:
            exp.plot_prediction(best_num=2, worst_num=2)


Run the file using the parameters

.. code-block:: bash

    python experiment.py path_to_config_file path_to_log_folder --epochs 200 --model_checkpoint_period 10 --prediction_checkpoint_period 10 --temp_folder path_to_temp_folder --analysis_folder path_to_analysis_folder
