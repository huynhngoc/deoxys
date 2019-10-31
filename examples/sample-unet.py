from deoxys.data import HDF5Reader
from deoxys.model import model_from_config
from deoxys.utils import read_file
from deoxys.utils import load_json_config

if __name__ == '__main__':
    # load model config
    config = read_file('examples/json/unet-config.json')
    config, = load_json_config(config)

    dr = HDF5Reader('../../full_dataset_single.h5',
                    batch_size=32, x_name='input', y_name='target')

    ar = config['architecture']
    input_params = {'shape': (191, 265, 2)}
    model = model_from_config(ar, input_params,
                              model_params=config['model_params'],
                              dataset_params=dr)
    model.fit_train(verbose=2)
