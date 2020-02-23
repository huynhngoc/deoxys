import os

mode = 'TENSORFLOW'
if 'KERAS_MODE' in os.environ:
    mode = os.environ.get('KERAS_MODE')
if mode.upper() == 'ALONE':
    from keras.backend import *
elif mode.upper() == 'TENSORFLOW':
    from tensorflow.keras.backend import *
