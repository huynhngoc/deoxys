from ..utils import is_keras_standalone

if is_keras_standalone():
    from keras.activations import *
else:
    from tensorflow.keras.activations import *
