from deoxys.model import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
from deoxys.utils import is_keras_standalone
if is_keras_standalone():
    from keras.applications import vgg16
    from keras.applications.vgg16 import preprocess_input, \
        decode_predictions
    from keras.preprocessing import image
else:
    from tensorflow.keras.applications import vgg16
    from tensorflow.keras.applications.vgg16 import preprocess_input, \
        decode_predictions
    from tensorflow.keras.preprocessing import image


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
    x = preprocess_input(x)

    return x


if __name__ == '__main__':
    vgg = vgg16.VGG16(weights='imagenet', include_top=True)
    vgg.summary()

    # load the vgg into the deoxys model
    deo = Model(vgg)

    # load an real image
    img = load_image('../../test_img/cat.jpg')

    # predict what is in that image
    preds = deo.predict(img)
    predicted_class = preds.argmax(axis=1)[0]

    print("predicted top1 class:", predicted_class)
    print('Predicted:', decode_predictions(preds, top=1)[0])

    # View activation map, 1st filters
    layer_list = [layer.name for layer in vgg.layers[1:19]]
    nrow, ncol = 4, 5

    # fig, axes = plt.subplots(nrow, ncol)
    # for ax in axes.flatten():
    #     ax.axis('off')
    # for i, layer in enumerate(layer_list):
    #     outs = deprocess_image(
    #         deo.activation_map_for_image(layer, img)[0][..., 0])
    #     axes[i//5, i % 5].imshow(outs)
    #     axes[i//5, i % 5].set_title(layer)
    #     # axes[i//5, i % 5].axis('off')
    # plt.suptitle('Activation Map')
    # plt.show()
    # input('Press ENTER to continue...')

    # fig, axes = plt.subplots(nrow, ncol)
    # for ax in axes.flatten():
    #     ax.axis('off')
    # for i, layer in enumerate(layer_list):
    #     outs = deprocess_image(deo.backprop(layer, img)[0])
    #     axes[i//5, i % 5].imshow(outs)
    #     axes[i//5, i % 5].set_title(layer)
    # plt.suptitle('Backprop Map')
    # plt.show()
    # input('Press ENTER to continue...')

    # fig, axes = plt.subplots(nrow, ncol)
    # for ax in axes.flatten():
    #     ax.axis('off')
    # for i, layer in enumerate(layer_list):
    #     outs = deprocess_image(deo.deconv(layer, img)[0])
    #     axes[i//5, i % 5].imshow(outs)
    #     axes[i//5, i % 5].set_title(layer)
    # plt.suptitle('Deconvnet Map')
    # plt.show()
    # input('Press ENTER to continue...')

    fig, axes = plt.subplots(nrow, ncol)
    for ax in axes.flatten():
        ax.axis('off')
    for i, layer in enumerate(layer_list):
        outs = deprocess_image(deo.guided_backprop(layer, img)[0])
        axes[i//5, i % 5].imshow(outs)
        axes[i//5, i % 5].set_title(layer)
    plt.suptitle('Guided Backprop Map')
    plt.show()
    input('Press ENTER to continue...')

    # fig, axes = plt.subplots(nrow, ncol)
    # for ax in axes.flatten():
    #     ax.axis('off')
    # for i, layer in enumerate(layer_list):
    #     outs = deprocess_image(deo.gradient_map(
    #         layer, epochs=5, step_size=2)[0])
    #     axes[i//5, i % 5].imshow(outs)
    #     axes[i//5, i % 5].set_title(layer)
    # plt.suptitle('Gradients Map')
    # plt.show()
    # input('Press ENTER to continue...')

    # fig, axes = plt.subplots(1, 3)
    # for ax in axes.flatten():
    #     ax.axis('off')
    # epoch = 0
    # name_list = ['cat', 'banana',  'cup']
    # index_list = [281, 954, 968]
    # img_list = []
    # gradient_generators = deo.gradient_map_generator(
    #     'predictions', filter_index=index_list)
    # while True:
    #     if epoch == 0:
    #         # initialize
    #         plt.suptitle('Gradients Map - step={}'.format(epoch))
    #         gradients = next(gradient_generators)
    #         for i, gradient in enumerate(gradients):
    #             outs = deprocess_image(gradient[0])
    #             ax = axes[i].imshow(outs)
    #             axes[i].set_title(name_list[i])
    #             img_list.append(ax)
    #     else:
    #         gradients = next(gradient_generators)
    #         if epoch % 5 == 0:
    #             for i, gradient in enumerate(gradients):
    #                 print(i)
    #                 print(gradient.shape)
    #                 img_list[i].set_data(deprocess_image(gradient[0]))

    #             plt.suptitle('Gradients Map - step={}'.format(epoch))
    #             plt.pause(1e-3)

    #     epoch += 1
    #     if epoch % 500 == 0:
    #         if input('Press ENTER to continue...') == 'exit':
    #             break
    # plt.show()
