"""
Example of running experiment with different setting in different interval
of epochs.
The model is loaded from 'examples/json/oxford-pet-config.json'.
The outputs are in '../../oxford_perf/logs'.
In this experiment, it first runs 10 epochs, saves model and predicts
validation data at epoch 5 and 10.
After that, it runs 10 more epochs. The logs are appended to previous log file.
Saving model and prediction of validation happen at epoch 12, 14, 16, 18, 20.
Finally, performance plots are created and prediction images in both contour
and separated form are created.
"""

from deoxys.experiment import Experiment
from deoxys.utils import read_file


if __name__ == '__main__':

    config = read_file('examples/json/oxford-pet-config.json')

    exp = Experiment(
        log_base_path='../../oxford_perf/logs',
        best_model_monitors=['val_loss', 'val_accuracy',
                             'val_binary_fbeta', 'val_fbeta']
    ).from_full_config(
        config
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=5,
        prediction_checkpoint_period=5,
        epochs=30
    ).run_experiment(
        model_checkpoint_period=2,
        prediction_checkpoint_period=2,
        epochs=10,
        initial_epoch=30
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
