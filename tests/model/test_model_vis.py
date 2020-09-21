import pytest
from deoxys.model import load_model, Model
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image


@pytest.fixture(scope='module', autouse=True)
def vgg():
    vgg = vgg16.VGG16(weights='imagenet', include_top=True)

    yield vgg


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def load_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = vgg16.preprocess_input(x)

    return x


def test_activation_map(vgg):
    # load the vgg into the deoxys model
    deo = Model(vgg)

    # load an real image
    img = load_image('tests/img/cat.jpg')

    # View activation map, 1st filters
    layer_list = [layer.name for layer in vgg.layers[1:4]]

    for i, layer in enumerate(layer_list):
        outs = deprocess_image(
            deo.activation_map(layer, img)[0][..., 0])


def test_backprop(vgg):
    # load the vgg into the deoxys model
    deo = Model(vgg)

    # load an real image
    img = load_image('tests/img/cat.jpg')

    layer_list = [layer.name for layer in vgg.layers[1:4]]

    for i, layer in enumerate(layer_list):
        outs = deprocess_image(deo.backprop(layer, img)[0])

    deo.backprop(layer, img, mode='one')
    deo.backprop(layer, img, mode='all')


def test_deconvnet(vgg):

    # load the vgg into the deoxys model
    deo = Model(vgg)

    # load an real image
    img = load_image('tests/img/cat.jpg')

    layer_list = [layer.name for layer in vgg.layers[1:4]]

    for i, layer in enumerate(layer_list):
        deprocess_image(deo.deconv(layer, img, mode='min')[0])


def test_guided_backprop(vgg):
    # load the vgg into the deoxys model
    deo = Model(vgg)

    # load an real image
    img = load_image('tests/img/cat.jpg')

    layer_list = [layer.name for layer in vgg.layers[1:4]]

    for i, layer in enumerate(layer_list):
        outs = deprocess_image(deo.guided_backprop(layer, img, mode='mean')[0])


def test_activation_maximization(vgg):
    deo = Model(vgg)

    layer_list = [layer.name for layer in vgg.layers[1:4]]

    for i, layer in enumerate(layer_list):
        outs = deprocess_image(deo.activation_maximization(
            layer, epochs=5, step_size=2)[0])


def test_max_filter(vgg):
    deo = Model(vgg)

    img = load_image('tests/img/cat.jpg')

    layer_list = [layer.name for layer in vgg.layers[1:4]]

    for i, layer in enumerate(layer_list):
        deo.max_filter(layer, img)
