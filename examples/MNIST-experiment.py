from deoxys.experiment import Experiment
from deoxys.utils import read_file


if __name__ == '__main__':

    config = read_file('examples/json/sequential-config.json')

    Experiment(log_base_path='../../mnist/logs').from_full_config(
        config).run_experiment()
