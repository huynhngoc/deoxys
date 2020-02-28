from ..utils import is_keras_standalone

if is_keras_standalone():
    from keras.callbacks import *
else:
    from tensorflow.keras.callbacks import *
