from deoxys.data import HDF5Reader
from deoxys.model import model_from_full_config
from deoxys.utils import read_file, write_file
from deoxys.utils import load_json_config

if __name__ == '__main__':
    # load model config
    config = read_file('examples/json/unet-sample-config.json')

    model = model_from_full_config(config)
    model.model.summary()
    model.fit_train(verbose=1, epochs=2)
    print("After 2 epochs")
    model.save('model2.h5')
    score = model.evaluate_test(verbose=1)
    print(score)
    write_file(str(score), '2.txt')

    model.fit_train(verbose=1, epochs=3, initial_epoch=3)
    print("After 3 epochs")
    model.save('model3.h5')
    score = model.evaluate_test(verbose=1)
    print(score)
    write_file(str(score), '3.txt')

    model.fit_train(verbose=1, epochs=4, initial_epoch=4)
    print("After 3 epochs")
    model.save('model4.h5')
    score = model.evaluate_test(verbose=1)
    print(score)
    write_file(str(score), '4.txt')

    model.fit_train(verbose=1, epochs=5, initial_epocj=5)
    score = model.evaluate_test(verbose=1)
    print("After 5 epochs")
    model.save('model5.h5')
    score = model.evaluate_test(verbose=1)
    print(score)
    write_file(str(score), '5.txt')
