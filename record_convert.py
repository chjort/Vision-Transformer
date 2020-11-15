import glob
import os

import tensorflow as tf

from chambers.augmentations import resize
from chambers.data.loader import SequentialImageDataset
from chambers.data.tf_record import serialize_tensor_example


def write_dataset_to_tfrecord(path, dataset, label_names):
    os.makedirs(path, exist_ok=True)
    previous_label = None
    current_writer = None
    for x, y in dataset:
        current_label = label_names[y.numpy()]

        if current_label != previous_label:
            if current_writer is not None:
                current_writer.close()

            save_path = os.path.join(path, "{}.tfrecord".format(current_label))
            current_writer = tf.io.TFRecordWriter(save_path)

        serialized_sample = serialize_tensor_example(x, y)
        current_writer.write(serialized_sample)
        previous_label = current_label


def preprocess(x):
    x = x[..., 0:1]
    return tf.cast(x, tf.uint8)


train_path = "/home/crr/datasets/omniglot/train"
test_path = "/home/crr/datasets/omniglot/test"

train_class_dirs = glob.glob(train_path + "/*/")
train_labels = list(range(len(train_class_dirs)))
train_label_names = {
    label: os.path.basename(cls.rstrip("/"))
    for cls, label in zip(train_class_dirs, train_labels)
}

test_class_dirs = glob.glob(test_path + "/*/")
test_labels = list(range(len(train_class_dirs), len(train_class_dirs) + len(test_class_dirs)))
test_label_names = {
    label: os.path.basename(cls.rstrip("/"))
    for cls, label in zip(test_class_dirs, test_labels)
}

n_train = len(train_class_dirs)
n_test = len(test_class_dirs)

# %%
INPUT_SHAPE = (84, 84, 1)

train_dataset = SequentialImageDataset(class_dirs=train_class_dirs,
                                       labels=train_labels,
                                       shuffle=False,
                                       reshuffle_iteration=False,
                                       repeats=None,
                                       seed=42)
train_dataset.map_image(resize, INPUT_SHAPE[0], INPUT_SHAPE[1])
train_dataset.map_image(preprocess)

test_dataset = SequentialImageDataset(class_dirs=test_class_dirs,
                                      labels=test_labels,
                                      shuffle=False,
                                      reshuffle_iteration=False,
                                      repeats=None,
                                      seed=42)
test_dataset.map_image(resize, INPUT_SHAPE[0], INPUT_SHAPE[1])
test_dataset.map_image(preprocess)

train_dataset = train_dataset.dataset
test_dataset = test_dataset.dataset

# %%
train_out = "/home/crr/datasets/omniglot/train_records"
test_out = "/home/crr/datasets/omniglot/test_records"
os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

write_dataset_to_tfrecord(train_out, train_dataset, train_label_names)
write_dataset_to_tfrecord(test_out, test_dataset, test_label_names)
