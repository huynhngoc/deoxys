from ..utils import is_keras_standalone

if is_keras_standalone():
    from keras.optimizers import *
else:
    from tensorflow.keras.optimizers import *
