from deoxys.model import load_model, model_from_full_config
from deoxys.utils import read_file, write_file


if __name__ == '__main__':
    config = read_file('examples/json/unet-sample-config.json')
    model = model_from_full_config(config, weights_file='model2.h5')

    model.fit_train(verbose=1, epochs=3, initial_epoch=2)
    print("After 3 epochs")
    model.save('model3.h5')
    score = model.evaluate_test(verbose=1)
    print(score)
    write_file(str(score), '3.txt')
