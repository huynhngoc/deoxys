from deoxys.keras import backend as K
from deoxys.model import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image


if __name__ == '__main__':

    # load model
    deo = load_model(
        '../../hn_perf/exps/model.014.h5')

    # get data reader
    dr = deo.data_reader

    # get input images, targets, predictions
    imgs = []
    targets = []

    datagen = dr.val_generator.generate()

    indexes = [10, 205, 230, 390, 834]
    k = len(indexes)

    for i, (x, y) in enumerate(datagen):
        for index in indexes:
            if i == index // 4:
                imgs.append(x[index % 4])
                targets.append(y[index % 4])
        if len(imgs) == k:
            break
    imgs = np.array(imgs)

    print('Predict ....')
    preds = deo.predict(imgs)

    layer_list = [
        layer for layer in deo.layers if 'conv2d' in layer or 'batch' in layer]

    filter_nums = [
        deo.layers[layer].output_shape[-1]
        for layer in deo.layers if 'conv2d' in layer or 'batch' in layer
    ]

    for layer in layer_list:
        activations = deo.activation_map_for_image(layer, imgs)

        filters = activations.shape[-1]

        for i in range(k):
            ncols = 8
            nrows = int(filters / 8)

            plt.figure(figsize=(8*5, nrows*5))

            for filter_idx in range(filters):
                plt.subplot(nrows, ncols, filter_idx + 1)
                plt.imshow(activations[i][..., filter_idx], 'gray')
                plt.title('Filter {}'.format(filter_idx + 1))
                plt.axis('off')

            plt.suptitle('Image Index {} - Layer {}'.format(indexes[i], layer))

            plt.savefig(
                '../../results/hn_vis/activation_map_{}_layer_{}.png'.format(
                    indexes[i], layer))
            plt.close('all')

        if imgs.shape[1:3] == activations.shape[1:3]:
            for i in range(k):
                ncols = 8
                nrows = int(filters / 8)

                plt.figure(figsize=(8*5, nrows*5))

                for filter_idx in range(filters):
                    plt.subplot(nrows, ncols, filter_idx + 1)
                    plt.imshow(imgs[i][..., 0], 'gray')
                    plt.contour(targets[i][..., 0], 1,
                                levels=1, colors='yellow')
                    plt.contour(preds[i][..., 0], 1, levels=1, colors='red')
                    plt.imshow(activations[i]
                               [..., filter_idx], 'jet', alpha=0.3)
                    plt.title('Filter {}'.format(filter_idx + 1))
                    plt.axis('off')

                plt.suptitle(
                    'Image Index {} - Layer {}'.format(indexes[i], layer))

                plt.savefig(
                    '../../results/hn_vis/activation_map_{}_layer_{}_overlay.png'.format(
                        indexes[i], layer))

                plt.close('all')
