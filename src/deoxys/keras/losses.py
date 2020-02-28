from ..utils import is_keras_standalone

if is_keras_standalone():
    from keras.losses import *
else:
    from tensorflow.keras.losses import *
