from deoxys.model import Model
from deoxys.utils import read_file
from tensorflow.keras.datasets import mnist


if __name__ == "__main__":
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    input_shape = (28, 28)

    # Scale data in 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Prepare the model
    config = read_file('examples/json/sequential-config.json')
    model = Model.from_config(config, shape=input_shape)

    model.fit(x_train, y_train, epochs=5)

    score = model.evaluate(x_test,  y_test, verbose=2)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
