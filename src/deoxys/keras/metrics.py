from ..utils import is_keras_standalone

if is_keras_standalone():
    from keras.metrics import *
else:
    from tensorflow.keras.metrics import *
