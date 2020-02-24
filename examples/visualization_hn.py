from deoxys.model import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
import h5py


deo = load_model(
    # '../../oxford_perf/logs_db/5e2e2349a356a4893813c8f7/model/model.030.h5')
    # '../../oxford_perf/log_db_ex2/5e3e87d2b26396ccbca9c3e7/model/model.030.h5')
    # '../../hn_perf/logs_saved_9epochs/model/model.006.h5')
    '../../hn_perf/exps/model.014.h5')
# '../../mnist/logs/model/model.012.h5')

if __name__ == '__main__':
    deo.model.summary()

    dr = deo.data_reader

    imgs = []
    targets = []

    datagen = dr.val_generator.generate()
    # datagen.__next__()
    # datagen.__next__()
    # x, y = datagen.__next__()

    indexes = [10, 205, 230, 390, 834]
    k = 5

    for i, (x, y) in enumerate(datagen):
        for index in indexes:
            if i == index // 4:
                imgs.append(x[index % 4])
                targets.append(y[index % 4])
        if len(imgs) == k:
            break
    imgs = np.array(imgs)
    print(imgs.shape)
    print('Predict ....')
    preds = deo.predict(imgs)

    nrow, ncol = 9, 5
    plt.figure(figsize=(25, 45))

    print('Plotting ...')
    # cur_pos = 1
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(imgs[i][..., 0], 'gray')
    #     plt.contour(targets[i][..., 0], 1, levels=1, colors='yellow')
    #     plt.contour(preds[i][..., 0], 1, levels=1, colors='red')
    #     plt.axis('off')

    # guided = deo.guided_backprop('conv2d', imgs)

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 0], 'gray')
    #     plt.title('conv2d CT')
    #     plt.axis('off')

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 1], 'gray')
    #     plt.title('conv2d PET')
    #     plt.axis('off')

    # guided = deo.guided_backprop('conv2d_9', imgs)

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 0], 'gray')
    #     plt.title('conv2d_9 CT')
    #     plt.axis('off')

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 1], 'gray')
    #     plt.title('conv2d_9 PET')
    #     plt.axis('off')

    # guided = deo.guided_backprop('conv2d_16', imgs)

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 0], 'gray')
    #     plt.title('conv2d_16 CT')
    #     plt.axis('off')

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 1], 'gray')
    #     plt.title('conv2d_16 PET')
    #     plt.axis('off')

    # guided = deo.guided_backprop('conv2d_17', imgs)

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 0], 'gray')
    #     plt.title('conv2d_17 CT')
    #     plt.axis('off')

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 1], 'gray')
    #     plt.title('conv2d_17 PET')
    #     plt.axis('off')

    # plt.savefig('../../hn_perf/exps/guided.png')
    # plt.show()

    # cur_pos = 1
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(imgs[i][..., 0], 'gray')
    #     plt.contour(targets[i][..., 0], 1, levels=1, colors='yellow')
    #     plt.contour(preds[i][..., 0], 1, levels=1, colors='red')
    #     plt.axis('off')

    # guided = deo.backprop('conv2d', imgs)

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 0], 'gray')
    #     plt.title('conv2d CT')
    #     plt.axis('off')

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 1], 'gray')
    #     plt.title('conv2d PET')
    #     plt.axis('off')

    # guided = deo.backprop('conv2d_9', imgs)

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 0], 'gray')
    #     plt.title('conv2d_9 CT')
    #     plt.axis('off')

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 1], 'gray')
    #     plt.title('conv2d_9 PET')
    #     plt.axis('off')

    # guided = deo.backprop('conv2d_16', imgs)

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 0], 'gray')
    #     plt.title('conv2d_16 CT')
    #     plt.axis('off')

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 1], 'gray')
    #     plt.title('conv2d_16 PET')
    #     plt.axis('off')

    # guided = deo.backprop('conv2d_17', imgs)

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 0], 'gray')
    #     plt.title('conv2d_17 CT')
    #     plt.axis('off')

    # cur_pos += k
    # for i in range(k):
    #     plt.subplot(nrow, ncol, cur_pos + i)
    #     plt.imshow(guided[i][..., 1], 'gray')
    #     plt.title('conv2d_17 PET')
    #     plt.axis('off')

    # plt.savefig('../../hn_perf/exps/backprop.png')
    # plt.show()

    cur_pos = 1
    for i in range(k):
        plt.subplot(nrow, ncol, cur_pos + i)
        plt.imshow(imgs[i][..., 0], 'gray')
        plt.contour(targets[i][..., 0], 1, levels=1, colors='yellow')
        plt.contour(preds[i][..., 0], 1, levels=1, colors='red')
        plt.axis('off')

    guided = deo.deconv('conv2d', imgs)

    cur_pos += k
    for i in range(k):
        plt.subplot(nrow, ncol, cur_pos + i)
        plt.imshow(guided[i][..., 0], 'gray')
        plt.title('conv2d CT')
        plt.axis('off')

    cur_pos += k
    for i in range(k):
        plt.subplot(nrow, ncol, cur_pos + i)
        plt.imshow(guided[i][..., 1], 'gray')
        plt.title('conv2d PET')
        plt.axis('off')

    guided = deo.deconv('conv2d_9', imgs)

    cur_pos += k
    for i in range(k):
        plt.subplot(nrow, ncol, cur_pos + i)
        plt.imshow(guided[i][..., 0], 'gray')
        plt.title('conv2d_9 CT')
        plt.axis('off')

    cur_pos += k
    for i in range(k):
        plt.subplot(nrow, ncol, cur_pos + i)
        plt.imshow(guided[i][..., 1], 'gray')
        plt.title('conv2d_9 PET')
        plt.axis('off')

    guided = deo.deconv('conv2d_16', imgs)

    cur_pos += k
    for i in range(k):
        plt.subplot(nrow, ncol, cur_pos + i)
        plt.imshow(guided[i][..., 0], 'gray')
        plt.title('conv2d_16 CT')
        plt.axis('off')

    cur_pos += k
    for i in range(k):
        plt.subplot(nrow, ncol, cur_pos + i)
        plt.imshow(guided[i][..., 1], 'gray')
        plt.title('conv2d_16 PET')
        plt.axis('off')

    guided = deo.deconv('conv2d_17', imgs)

    cur_pos += k
    for i in range(k):
        plt.subplot(nrow, ncol, cur_pos + i)
        plt.imshow(guided[i][..., 0], 'gray')
        plt.title('conv2d_17 CT')
        plt.axis('off')

    cur_pos += k
    for i in range(k):
        plt.subplot(nrow, ncol, cur_pos + i)
        plt.imshow(guided[i][..., 1], 'gray')
        plt.title('conv2d_17 PET')
        plt.axis('off')

    plt.savefig('../../hn_perf/exps/deconvnet.png')
    plt.show()
