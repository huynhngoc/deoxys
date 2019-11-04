from deoxys.data import HDF5Reader
from deoxys.model import model_from_full_config
from deoxys.utils import read_file
from deoxys.utils import load_json_config

if __name__ == '__main__':
    # load model config
    config = read_file('examples/json/unet-sample-config.json')

    dr = HDF5Reader('../../full_dataset_single.h5',
                    batch_size=8, x_name='input', y_name='target',
                    batch_cache=4)

    model = model_from_full_config(config)
    model.model.summary()
    model.fit_train(verbose=1)
    model.save('model.h5')
