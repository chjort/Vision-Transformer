import glob
import os

import tensorflow as tf
import tensorflow_addons as tfa
from einops import rearrange, repeat

from chambers.data.loader import InterleaveTFRecordOneshotDataset, InterleaveImageDataset
from chambers.models.transformer import VisionTransformerOS

# %%
k = 5
q = 1
n = 2


def arrange_oneshot_kway(x, y):
    Q = x[:, :q]
    S = x[:, q:]

    Q = repeat(Q, "k q h w c -> (k kn q) h w c", kn=k * n)
    S = rearrange(S, "k n h w c -> (k n) h w c")
    S = repeat(S, "kn h w c -> (k kn) h w c", k=k)

    return Q  # , S


TEST_PATH = "/home/crr/datasets/omniglot/test"
class_dirs = glob.glob(os.path.join(TEST_PATH, "*/"))
labels = list(range(len(class_dirs)))
td = InterleaveImageDataset(class_dirs=class_dirs,
                            labels=labels,
                            class_cycle_length=k,
                            n_per_class=n + q,
                            block_bound=True,
                            sample_n_random=True,
                            shuffle=True,
                            seed=42
                            )
td.batch(n + q, drop_remainder=True)
td.batch(k, drop_remainder=True)
td.map(arrange_oneshot_kway)
td.window(k * n)
td.dataset

# %%
it = iter(td.dataset)
Q, S = next(it)

itq = iter(Q)
its = iter(S)

bq = next(itq)
bs = next(its)

bq.shape

Q.shape
S.shape

# %%
x, y = next(iter(td.dataset))
x.shape
y

Q = x[:, :q]
S = x[:, q:]

Q = repeat(Q, "k q h w c -> (k kn q) h w c", kn=k * n)
S = rearrange(S, "k n h w c -> (k n) h w c")
S = repeat(S, "kn h w c -> (k kn) h w c", k=k)

# Q = y[:, :q]
# S = y[:, q:]
#
# Q = repeat(Q, "k q -> (k kn q)", kn=k*n)
# S = rearrange(S, "k n -> (k n)")
# S = repeat(S, "kn -> (k kn)", k=k)

Q.shape
S.shape


# %%
def flatten_batch(x, y):
    x1 = rearrange(x[0], "b n h w c -> (b n) h w c")
    x2 = rearrange(x[1], "b n h w c -> (b n) h w c")
    y = tf.reshape(y, [-1])

    return (x1, x2), y


def preprocess(x, y):
    x1, x2 = x[0], x[1]

    x1 = x1[..., 0:1]
    x2 = x2[..., 0:1]

    return (x1, x2), y


# strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

# data parameters
TEST_PATH = "/home/crr/datasets/omniglot/test_records"

# loading data
test_records = glob.glob(os.path.join(TEST_PATH, "*.tfrecord"))

N_PER_PAIR = 2  # increasing this value causes overfitting
N_PAIRS_PER_DEVICE = 64
N_PAIRS = N_PAIRS_PER_DEVICE * strategy.num_replicas_in_sync  # scale batch size with this.
test_dataset = InterleaveTFRecordOneshotDataset(records=test_records,
                                                n=N_PER_PAIR,
                                                sample_n_random=False,
                                                shuffle=False,
                                                reshuffle_iteration=False,
                                                repeats=None,
                                                seed=42)
test_dataset.map(preprocess)
test_dataset.batch(N_PAIRS)
test_dataset.map(flatten_batch)
test_dataset.prefetch(-1)

test_dataset = test_dataset.dataset

# %%

INPUT_SHAPE = (84, 84, 1)
PATCH_SIZE = 7
PATCH_DIM = 128
N_ENCODER_LAYERS = 8
NUM_HEADS = 8
FF_DIM = 512
LR = 0.00006

with strategy.scope():
    model = VisionTransformerOS(input_shape=INPUT_SHAPE,
                                patch_size=PATCH_SIZE,
                                patch_dim=PATCH_DIM,
                                n_encoder_layers=N_ENCODER_LAYERS,
                                n_heads=NUM_HEADS,
                                ff_dim=FF_DIM,
                                dropout_rate=0.1
                                )

    optimizer = tfa.optimizers.AdamW(learning_rate=LR,
                                     weight_decay=1e-4,
                                     beta_1=0.9,
                                     beta_2=0.98,  # 0.999
                                     epsilon=1e-8,
                                     amsgrad=False)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC()])

    # %%
    model.load_weights("outputs/vitos_b2-98_drop01_b256_e200_p7_2/weights_0.8121.h5")

model.summary()

# %%
model.evaluate(test_dataset)
