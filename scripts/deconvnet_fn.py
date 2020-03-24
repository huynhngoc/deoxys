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

    nrow, ncol = 5, k

    def loss_fn(output):
        # tp = K.cast(np.array(targets), K.floatx())
        return K.sum(output * K.cast(np.array(targets), K.floatx()))

    for layer in layer_list[-1:]:
        deconvnets = deo.deconv(layer, imgs, mode='custom', loss_fn=loss_fn)
        plt.figure(figsize=(5*ncol, 5*nrow))

        for i in range(k):
            plt.subplot(nrow, ncol, i+1)
            plt.imshow(imgs[i][..., 0], 'gray')
            plt.contour(targets[i][..., 0], 1,
                        levels=1, colors='yellow')
            plt.contour(preds[i][..., 0], 1, levels=1, colors='red')
            plt.title('Image Index {}'.format(indexes[i]))
            plt.axis('off')

            plt.subplot(nrow, ncol, k + i+1)
            plt.imshow(deconvnets[i][..., 0], 'gray')
            plt.title('Deconvnet CT')
            plt.axis('off')

            plt.subplot(nrow, ncol, 2*k + i+1)
            plt.imshow(deconvnets[i][..., 1], 'gray')
            plt.title('Deconvnet PET')
            plt.axis('off')

            plt.subplot(nrow, ncol, 3*k + i+1)
            plt.imshow(imgs[i][..., 0], 'gray')
            plt.imshow(deconvnets[i][..., 0], 'jet', alpha=0.3)
            plt.title('Deconvnet CT overlay')
            plt.axis('off')

            plt.subplot(nrow, ncol, 4*k + i+1)
            plt.imshow(imgs[i][..., 0], 'gray')
            plt.imshow(deconvnets[i][..., 1], 'jet', alpha=0.3)
            plt.title('Deconvnet PET overlay')
            plt.axis('off')

        plt.suptitle('Deconvnet True Positive at Layer {}'.format(layer))
        # plt.show()
        plt.savefig(
            '../../results/hn_vis/deconvnet_tp_layer_{}'.format(layer))

        plt.close('all')


def loss_fn(output):
    return K.sum(output * (K.constant(np.array(targets)) - output))


for layer in layer_list[-1:]:
    deconvnets = deo.deconv(layer, imgs, mode='custom', loss_fn=loss_fn)
    plt.figure(figsize=(5*ncol, 5*nrow))

    for i in range(k):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(imgs[i][..., 0], 'gray')
        plt.contour(targets[i][..., 0], 1,
                    levels=1, colors='yellow')
        plt.contour(preds[i][..., 0], 1, levels=1, colors='red')
        plt.title('Image Index {}'.format(indexes[i]))
        plt.axis('off')

        plt.subplot(nrow, ncol, k + i+1)
        plt.imshow(deconvnets[i][..., 0], 'gray')
        plt.title('Deconvnet CT')
        plt.axis('off')

        plt.subplot(nrow, ncol, 2*k + i+1)
        plt.imshow(deconvnets[i][..., 1], 'gray')
        plt.title('Deconvnet PET')
        plt.axis('off')

        plt.subplot(nrow, ncol, 3*k + i+1)
        plt.imshow(imgs[i][..., 0], 'gray')
        plt.imshow(deconvnets[i][..., 0], 'jet', alpha=0.3)
        plt.title('Deconvnet CT overlay')
        plt.axis('off')

        plt.subplot(nrow, ncol, 4*k + i+1)
        plt.imshow(imgs[i][..., 0], 'gray')
        plt.imshow(deconvnets[i][..., 1], 'jet', alpha=0.3)
        plt.title('Deconvnet PET overlay')
        plt.axis('off')

    plt.suptitle('Deconvnet False Positive at Layer {}'.format(layer))
    # plt.show()
    plt.savefig(
        '../../results/hn_vis/deconvnet_fp_layer_{}'.format(layer))

    plt.close('all')
