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

        max_activation = np.max(activations, axis=-1)

        plt.figure(figsize=(25, 10))

        for i in range(k):
            nrow = 2

            max_active = max_activation[i]
            original_size = imgs.shape[1:3]
            if max_active.shape == original_size:
                nrow = 3

            # plot original images with predictions
            plt.subplot(nrow, k, i+1)
            plt.imshow(imgs[i][..., 0], 'gray')
            plt.contour(targets[i][..., 0], 1, levels=1, colors='yellow')
            plt.contour(preds[i][..., 0], 1, levels=1, colors='red')
            plt.title('Index {}'.format(indexes[i]))
            plt.axis('off')

            # plot max activation
            plt.subplot(nrow, k, k+i+1)
            plt.imshow(max_active, 'gray')
            plt.title('Max outputs')
            plt.axis('off')

            if nrow == 3:
                # plot max_output overlay original images
                plt.subplot(nrow, k, 2*k+i+1)
                plt.imshow(imgs[i][..., 0], 'gray')
                plt.imshow(max_active, 'jet', alpha=0.3)
                plt.title('Max outputs on images')
                plt.axis('off')

            plt.suptitle('Layer {}'.format(layer))

        # plt.show()
        plt.savefig(
            '../../results/hn_vis/activation_map_max_layer_{}.png'.format(layer))
        plt.close('all')
