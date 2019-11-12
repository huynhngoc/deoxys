from deoxys.experiment import Experiment
from deoxys.utils import read_file

config = read_file('examples/json/sequential-config.json')

history = Experiment().from_full_config(
    config).run_experiment(test_checkpoint=1, test_file='../../mnist')
print(history)
