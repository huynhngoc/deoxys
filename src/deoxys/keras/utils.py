from ..utils import is_keras_standalone

if is_keras_standalone():
    from keras.utils import *
else:
    from tensorflow.keras.utils import *
