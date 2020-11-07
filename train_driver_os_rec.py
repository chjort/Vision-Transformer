import glob
import os

import tensorflow as tf
import tensorflow_addons as tfa
from einops import rearrange

from chambers.data.loader import InterleaveTFRecordOneshotDataset
from chambers.models.transformer import VisionTransformerOS

train_path = "/home/crr/datasets/omniglot/train_records"
test_path = "/home/crr/datasets/omniglot/test_records"

train_records = glob.glob(os.path.join(train_path, "*.tfrecord"))
test_records = glob.glob(os.path.join(test_path, "*.tfrecord"))

strategy = tf.distribute.MirroredStrategy()


# strategy = tf.distribute.OneDeviceStrategy("/gpu:0")


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


n = 2  # increasing this value causes overfitting
n_pairs = 128 * strategy.num_replicas_in_sync  # scale batch size with this.
train_dataset = InterleaveTFRecordOneshotDataset(records=train_records,
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

test_dataset = InterleaveTFRecordOneshotDataset(records=test_records,
                                                n=n,
                                                sample_n_random=False,
                                                shuffle=False,
                                                reshuffle_iteration=False,
                                                repeats=None,
                                                seed=42)
test_dataset.map(preprocess)
test_dataset.batch(n_pairs)
test_dataset.map(flatten_batch)
test_dataset.prefetch(-1)

train_dataset = train_dataset.dataset
test_dataset = test_dataset.dataset

# %%
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

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
    #                                                              decay_steps=100,
    #                                                              decay_rate=0.1,
    #                                                              staircase=True)

    LR = 0.00006
    optimizer = tfa.optimizers.AdamW(lr=LR,
                                     weight_decay=1e-4,
                                     beta_1=0.9,
                                     beta_2=0.999,  # 0.98
                                     epsilon=1e-8,
                                     amsgrad=False)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics="accuracy")

model.summary()

# %%
EPOCHS = 100
steps_per_epoch = 200 // strategy.num_replicas_in_sync

output_dir = "outputs/vitos_no_tricks_adamw_drop01"
os.makedirs(output_dir, exist_ok=True)
hist = model.fit(train_dataset,
                 epochs=EPOCHS,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=test_dataset,
                 callbacks=[
                     tf.keras.callbacks.CSVLogger(os.path.join(output_dir, "log.csv")),
                     tf.keras.callbacks.TensorBoard(os.path.join(output_dir, "tb_logs"),
                                                    profile_batch=0)
                 ]
                 )
