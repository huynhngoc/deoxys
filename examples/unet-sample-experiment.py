from deoxys.experiment import Experiment
from deoxys.utils import read_file

config = read_file('examples/json/unet-sample-config.json')

history = Experiment().from_full_config(
    config).run_experiment(
        train_history_log=True,
        model_checkpoint_period=1,
        eval_checkpoint_period=1,
        prediction_checkpoint_period=1,
        plot_performance=True,
        masked_images=[i for i in range(42)],
        log_base_path='../../hn_perf/logs',
        epochs=3
)
print(history)

# "callbacks": [
#     {
#         "class_name": "CSVLogger",
#         "config": {
#             "filename": "../../unet_log.csv"
#         }
#     },
#     {
#         "class_name": "ModelCheckpoint",
#         "config": {
#             "monitor": "val_loss",
#             "filepath": "../../unet.{epoch:02d}-{binary_fbeta[0]:.5f}.h5",
#             "period": 1
#         }
#     }
# ]
