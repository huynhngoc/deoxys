from deoxys.experiment import Experiment
from deoxys.utils import read_file

config = read_file('examples/json/oxford-pet-config.json')

history = Experiment().from_full_config(
    config).run_experiment()
print(history)
