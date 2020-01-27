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

from deoxys.experiment import ExperimentDB
from deoxys.utils import read_file
from deoxys.database import MongoDBClient
from bson.objectid import ObjectId

if __name__ == '__main__':

    experiment_id = ObjectId('5e2e0fd8db58a9f1a10dfe44')
    print(type(experiment_id))
    dbclient = MongoDBClient('deoxys', 'localhost', 27017)

    exp = ExperimentDB(
        dbclient=dbclient, experiment=experiment_id,
        log_base_path='../../oxford_perf/logs_db',
        best_model_monitors=['val_loss', 'val_accuracy',
                             'val_binary_fbeta', 'val_fbeta']
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=2,
        prediction_checkpoint_period=2,
        epochs=30
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

    print(exp.best_model())
