from deoxys.experiment import Experiment
from deoxys.utils import read_file

config = read_file('examples/json/oxford-pet-config.json')

history = Experiment().from_full_config(
    config).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        eval_checkpoint_period=1,
        prediction_checkpoint_period=1,
        plot_performance=True,
        masked_images=[0, 1, 2, 3],
        log_base_path='../../oxford_perf/logs',
        epochs=1
)
print(history)


# "callbacks": [
#             {
#                 "class_name": "CSVLogger",
#                 "config": {
#                     "filename": "../../oxford.csv"
#                 }
#             },
#             {
#                 "class_name": "DeoxysModelCheckpoint",
#                 "config": {
#                     "monitor": "val_loss",
#                     "filepath": "../../oxford.{epoch:02d}-{accuracy:.5f}.h5",
#                     "period": 1
#                 }
#             },
#             {
#                 "class_name": "EvaluationCheckpoint",
#                 "config": {
#                     "filename": "../../oxford_test.csv",
#                     "period": 1
#                 }
#             },
#             {
#                 "class_name": "PredictionCheckpoint",
#                 "config": {
#                     "filepath": "../../oxford_prediction.{epoch:02d}.h5",
#                     "use_original": false
#                 }
#             }
#         ]
