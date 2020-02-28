from ..utils import is_keras_standalone

if is_keras_standalone():
    from keras.models import *
else:
    from tensorflow.keras.models import *
