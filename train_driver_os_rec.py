"""TODOS:
* Set seed and test its effect
* Use learning rate scheduler with 10% linear warmup steps.
* Increase batch size to be very big 30k-100k
* Try LAMB optimizer
* Increase epochs until clear overfitting
* Use exponential decay learning rate scheduler with 10% linear warmup steps.
* Regularize if still overfitting
* Increase model size until not overfitting (double descent theory)
"""

from chambers.utils import set_random_seed

set_random_seed(42)

import glob
import os

import tensorflow as tf
import tensorflow_addons as tfa
from einops import rearrange

from chambers.data.loader import InterleaveTFRecordOneshotDataset
from chambers.models.transformer import VisionTransformerOS
from chambers.utils import get_model_memory_usage


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


strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

# data parameters
TRAIN_PATH = "/home/crr/datasets/omniglot/train_records"
TEST_PATH = "/home/crr/datasets/omniglot/test_records"

# loading data
train_records = glob.glob(os.path.join(TRAIN_PATH, "*.tfrecord"))
test_records = glob.glob(os.path.join(TEST_PATH, "*.tfrecord"))

N_PER_PAIR = 2  # increasing this value causes overfitting
N_PAIRS_PER_DEVICE = 256
N_PAIRS = N_PAIRS_PER_DEVICE * strategy.num_replicas_in_sync  # scale batch size with this.
train_dataset = InterleaveTFRecordOneshotDataset(records=train_records,
                                                 n=N_PER_PAIR,
                                                 sample_n_random=True,
                                                 shuffle=True,
                                                 reshuffle_iteration=True,
                                                 repeats=-1,
                                                 seed=42)
train_dataset.map(preprocess)
train_dataset.batch(N_PAIRS)
train_dataset.map(flatten_batch)
train_dataset.prefetch(-1)

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

train_dataset = train_dataset.dataset
test_dataset = test_dataset.dataset

# %%
EPOCHS = 200
STEPS_PER_EPOCH = 200 // strategy.num_replicas_in_sync
LR = 0.00006

INPUT_SHAPE = (84, 84, 1)
PATCH_SIZE = 12
PATCH_DIM = 128
N_ENCODER_LAYERS = 8
NUM_HEADS = 8
FF_DIM = 512

with strategy.scope():
    model = VisionTransformerOS(input_shape=INPUT_SHAPE,
                                patch_size=PATCH_SIZE,
                                patch_dim=PATCH_DIM,
                                n_encoder_layers=N_ENCODER_LAYERS,
                                n_heads=NUM_HEADS,
                                ff_dim=FF_DIM,
                                dropout_rate=0.1
                                )

    # LR = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LR,
    #                                                     decay_steps=int(0.66 * EPOCHS) * STEPS_PER_EPOCH,
    #                                                     decay_rate=0.1,
    #                                                     staircase=True)

    optimizer = tfa.optimizers.AdamW(learning_rate=LR,
                                     weight_decay=1e-4,
                                     beta_1=0.9,
                                     beta_2=0.98,  # 0.999
                                     epsilon=1e-8,
                                     amsgrad=False)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics="accuracy")

model.summary()

batch_size = 2 * N_PER_PAIR * N_PAIRS_PER_DEVICE
batch_mem_gb = get_model_memory_usage(batch_size, model)
print("Batch size of {} in memory: {}GB".format(batch_size, batch_mem_gb))

# %%
output_dir = "outputs/vitos_b2-98_drop01_b1024_e200_seed"
os.makedirs(output_dir, exist_ok=True)
hist = model.fit(train_dataset,
                 epochs=EPOCHS,
                 steps_per_epoch=STEPS_PER_EPOCH,
                 validation_data=test_dataset,
                 callbacks=[
                     tf.keras.callbacks.CSVLogger(os.path.join(output_dir, "log.csv")),
                     tf.keras.callbacks.TensorBoard(os.path.join(output_dir, "tb_logs"),
                                                    profile_batch=0)
                 ]
                 )
