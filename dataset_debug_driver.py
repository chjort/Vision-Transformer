import glob

import tensorflow as tf
from einops import rearrange

from chambers.augmentations import resize
from chambers.data.loader import InterleaveOneshotDataset, InterleaveImageDataset, InterleaveImagesDatasetTensor, \
    InterleaveOneshotDatasetTensor
from chambers.data.utils import time_dataset

train_path = "/home/crr/datasets/omniglot/train"
test_path = "/home/crr/datasets/omniglot/test"

train_class_dirs = glob.glob(train_path + "/*/")[:4]
train_labels = list(range(len(train_class_dirs)))

test_class_dirs = glob.glob(test_path + "/*/")
test_labels = list(range(len(train_class_dirs), len(train_class_dirs) + len(test_class_dirs)))

n_train = len(train_class_dirs)
n_test = len(test_class_dirs)


# %%
def fn(n):
    files = tf.repeat(n, 2)
    return tf.data.Dataset.from_tensors(files)


cyc_len = 2
b_len = 1
a = [1, 2, 3, 4]
d = tf.data.Dataset.from_tensor_slices(a).shuffle(buffer_size=4, reshuffle_each_iteration=True, seed=42).repeat()
d = d.interleave(fn, cyc_len, b_len)
d = d.batch(cyc_len * b_len)

it = iter(d)
x = next(it)
x

# %%
n_class = 4
per_class = 6
td = InterleaveImageDataset(class_dirs=train_class_dirs,
                            labels=train_labels,
                            class_cycle_length=n_class,
                            n_per_class=per_class,
                            sample_n_random=True,
                            repeats=-1,
                            shuffle=True,
                            reshuffle_iteration=False,
                            buffer_size=None,
                            seed=42
                            )
td.batch(n_class, drop_remainder=True)
time_dataset(td.dataset, 2)

tdt = InterleaveImagesDatasetTensor(class_dirs=train_class_dirs,
                                    labels=train_labels,
                                    class_cycle_length=n_class,
                                    n_per_class=per_class,
                                    sample_n_random=True,
                                    repeats=-1,
                                    shuffle=True,
                                    reshuffle_iteration=False,
                                    buffer_size=None,
                                    seed=42
                                    )
tdt.batch(n_class, drop_remainder=True)
time_dataset(tdt.dataset, 2)

# it = iter(td.dataset)

# %%
import matplotlib.pyplot as plt

x, y = next(it)
x.shape
y

rearrange(y, "n k -> k n")

fig, axes = plt.subplots(len(x), 1, figsize=(5, 8))
for i in range(len(x)):
    axes[i].imshow(x[i], cmap="gray")
    axes[i].set_ylabel(str(y[i].numpy()), rotation=0)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.tight_layout()
plt.show()

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
# n_pairs = 2
train_dataset = InterleaveOneshotDatasetTensor(class_dirs=train_class_dirs,
                                               labels=train_labels,
                                               n=n,
                                               sample_n_random=True,
                                               shuffle=True,
                                               reshuffle_iteration=False,
                                               repeats=-1,
                                               seed=42)

train_dataset.map(preprocess)
train_dataset.batch(n_pairs)
train_dataset.map(flatten_batch)

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

train_dataset = train_dataset.dataset
test_dataset = test_dataset.dataset

# %%
import matplotlib.pyplot as plt

it = iter(train_dataset)

c = 0
# %%
(x1, x2), y = next(it)
print(x1.shape, x2.shape, y.shape)

batch_size = n * 2 * n_pairs

nplot = tf.minimum(8, x1.shape[0]).numpy()
br = iter(range(0, batch_size, nplot))
c += 1

# %%
start = next(br)
fig, axes = plt.subplots(nplot, 2)
for i, idx in enumerate(range(start, start + nplot)):
    axes[i][0].imshow(x1[idx], cmap="gray")
    axes[i][0].set_ylabel(str(y[idx].numpy()), rotation=0)
    axes[i][1].imshow(x2[idx], cmap="gray")
    axes[i][0].set_xticks([])
    axes[i][1].set_xticks([])
    axes[i][0].set_yticks([])
    axes[i][1].set_yticks([])
plt.tight_layout()
plt.show()
