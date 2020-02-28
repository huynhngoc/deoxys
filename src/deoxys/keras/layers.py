from ..utils import is_keras_standalone

if is_keras_standalone():
    from keras.layers import *
else:
    from tensorflow.keras.layers import *
