import pandas as pd
import matplotlib.pyplot as plt


def plot_log_performance_from_csv(filepath, output_path):
    # Load data to file
    df = pd.read_csv(filepath, index_col='epoch')

    # Convert str-like array into float numbers
    object_keys = [key for key, dtype in df.dtypes.items()
                   if dtype == 'object']
    for key in object_keys:
        df[key] = df[key].map(_arraystr2float, na_action='ignore')

    # Plot all data
    _plot_data(df, 'All parameters', df.columns, output_path + '/all.png')

    # Plot train data
    train_keys = [key for key in df.columns if 'val' not in key]
    _plot_data(df, 'Train Performance', train_keys, output_path + '/train.png')

    # Plot val data
    val_keys = [key for key in df.columns if 'val' in key]
    _plot_data(df, 'Validation Performance',
               val_keys, output_path + '/val.png')

    # Plot compare train and val
    compare = [[key, 'val_' + key] for key in train_keys]
    for keys in compare:
        _plot_data(df, '{} Performance'.format(
            keys[0]), keys, output_path + '/{}.png'.format(keys[0]))


def plot_evaluation_performance_from_csv(filepath, output_path):
    # Load data to file
    df = pd.read_csv(filepath, index_col='epoch')

    # Convert str-like array into float numbers
    object_keys = [key for key, dtype in df.dtypes.items()
                   if dtype == 'object']
    for key in object_keys:
        df[key] = df[key].map(_arraystr2float, na_action='ignore')

    # Plot evaluation
    _plot_data(df, 'Evaluation Performance', df.columns,
               output_path + '/evaluation.png')


def mask_prediction(output_path, image, true_mask, pred_mask,
                    title='Predicted',
                    mask_levels=None, channel=None):
    if not mask_levels:
        mask_levels = 1
    if not channel:
        if (len(image.shape) == 2
                or (len(image.shape) == 3 and image.shape[2] == 3)):
            image_data = image
        else:
            image_data = image[..., 0]
    else:
        image_data = image[..., channel]

    true_mask_data = true_mask
    pred_mask_data = pred_mask

    if len(true_mask_data.shape) == 3:
        true_mask_data = true_mask[..., 0]
        pred_mask_data = pred_mask[..., 0]

    plt.figure()
    plt.imshow(image_data)
    true_con = plt.contour(
        true_mask_data, 1, levels=mask_levels, color='yellow')
    pred_con = plt.contour(
        pred_mask_data, 1, levels=mask_levels, color='red')

    plt.title(title)
    plt.legend([true_con.collections[0],
                pred_con.collections[0]], ['True', 'Predicted'])
    plt.savefig(output_path)
    plt.close('all')


def _plot_data(dataframe, name, columns, filename):
    ax = dataframe[columns].plot()
    epoch_num = dataframe.shape[0]
    if epoch_num < 20:
        ax.set_xticks(dataframe.index)
    else:
        tick_distance = epoch_num // 10
        ax.set_xticks([tick for i, tick in enumerate(
            dataframe.index) if i % tick_distance == 0])
    ax.set_title(name)
    plt.savefig(filename)


def _arraystr2float(val):
    if '"[' in val and ']"' in val:
        array = val[2: -2].split(',')
        if len(array) == 1:
            try:
                return float(array[0])
            except Exception:
                return array[0]
        else:
            return array
    else:
        return val
