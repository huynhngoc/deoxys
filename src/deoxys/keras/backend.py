from ..utils import is_keras_standalone

if is_keras_standalone():
    from keras.backend import *
else:
    from tensorflow.keras.backend import *
