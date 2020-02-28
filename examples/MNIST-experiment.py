"""
To use comet.ml, uncomment this line
"""
###############################################
# from comet_ml import Experiment as CometEx
###############################################
from deoxys.experiment import Experiment
from deoxys.utils import read_file

if __name__ == '__main__':
    """
    To use comet.ml, uncomment this line
    """
    ######################################################################
    # # Create an experiment with your api key
    # experiment = CometEx(api_key="YOUR_API_KEY",
    #                      project_name="YOUR_PROJECT_NAME",
    #                      workspace="YOUR_WORKSPACE")
    ######################################################################
    config = read_file('examples/json/sequential-config.json')

    Experiment(log_base_path='../../mnist/logs').from_full_config(
        config).run_experiment()
