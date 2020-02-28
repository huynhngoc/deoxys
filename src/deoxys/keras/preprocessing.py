from ..utils import is_keras_standalone

if is_keras_standalone():
    from keras.preprocessing.image import *
else:
    from tensorflow.keras.preprocessing.image import *
