from deoxys.model import Model
from deoxys.utils import read_file
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


if __name__ == "__main__":
    # Pre-config
    batch_size = 128
    num_classes = 10
    epochs = 12

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Scale data in 1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # y
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Prepare the model
    config = read_file('examples/json/sequential-config.json')
    model = Model.from_config(config, shape=input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
