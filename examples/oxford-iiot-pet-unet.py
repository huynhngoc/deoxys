import tensorflow as tf
import tensorflow_datasets as tfds
from deoxys.model import model_from_full_config
from deoxys.utils import read_file


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask


def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


if __name__ == "__main__":
    # Load data
    dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True)

    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    # Preprocess data
    train = dataset['train'].map(
        load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset['test'].map(load_image_test)

    train_dataset = train.cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    # Prepare the model
    config = read_file('examples/json/unet-config.json')
    model = model_from_full_config(config)

    EPOCHS = 20
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits[
        'test'
    ].num_examples//BATCH_SIZE//VAL_SUBSPLITS

    model.fit(train_dataset, epochs=EPOCHS,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_steps=VALIDATION_STEPS,
              validation_data=test_dataset,
              callbacks=None)
