import glob
import os

import tensorflow as tf
import tensorflow.keras.backend as K
from einops import rearrange
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2

from chambers.augmentations import resize
from chambers.data.loader import InterleaveOneshotDataset
from chambers.models.transformer import VisionTransformerOS


train_path = "/home/crr/datasets/omniglot/train"
test_path = "/home/crr/datasets/omniglot/test"

train_class_dirs = glob.glob(train_path + "/*/")
train_labels = list(range(len(train_class_dirs)))

test_class_dirs = glob.glob(test_path + "/*/")
test_labels = list(range(len(train_class_dirs), len(train_class_dirs) + len(test_class_dirs)))

n_train = len(train_class_dirs)
n_test = len(test_class_dirs)

# strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

# %%
INPUT_SHAPE = (84, 84, 1)

def flatten_batch(x, y):
    x1 = rearrange(x[0], "b n h w c -> (b n) h w c")
    x2 = rearrange(x[1], "b n h w c -> (b n) h w c")
    y = tf.reshape(y, [-1])

    return (x1, x2), y


def preprocess(x, y):
    x1, x2 = x[0], x[1]

    x1 = resize(x1, INPUT_SHAPE[0], INPUT_SHAPE[1])
    x2 = resize(x2, INPUT_SHAPE[0], INPUT_SHAPE[1])

    x1 = x1[..., 0:1]
    x2 = x2[..., 0:1]

    return (x1, x2), y


n = 2
n_pairs = 16 * strategy.num_replicas_in_sync
train_dataset = InterleaveOneshotDataset(class_dirs=train_class_dirs,
                                         labels=train_labels,
                                         n=n,
                                         sample_n_random=True,
                                         shuffle=True,
                                         reshuffle_iteration=True,
                                         repeats=-1,
                                         seed=42)
train_dataset.map(preprocess)
train_dataset.batch(n_pairs)
train_dataset.map(flatten_batch)
train_dataset.prefetch(-1)

test_dataset = InterleaveOneshotDataset(class_dirs=test_class_dirs,
                                        labels=test_labels,
                                        n=n,
                                        sample_n_random=True,
                                        shuffle=True,
                                        reshuffle_iteration=False,
                                        repeats=-1,
                                        seed=42)
test_dataset.map(preprocess)
test_dataset.batch(n_pairs)
test_dataset.map(flatten_batch)
test_dataset.prefetch(-1)

train_dataset = train_dataset.dataset
test_dataset = test_dataset.dataset

