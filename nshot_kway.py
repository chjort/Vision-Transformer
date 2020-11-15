import glob
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from einops import rearrange, repeat

from chambers.data.loader import InterleaveTFRecordDataset
from chambers.data.tf_record import batch_deserialize_tensor_example_uint8
from chambers.models.transformer import VisionTransformerOS


class Nshot_Kway(tf.keras.metrics.Metric):
    def __init__(self, name="nshot_kway"):
        super(Nshot_Kway, self).__init__(name=name)
        self.n_episodes = self.add_weight(name="n_trials", initializer="zeros")
        self.n_correct = self.add_weight(name="n_correct_trials", initializer="zeros")

    def update_state(self, y_true, y_pred, **kwargs):
        self.n_episodes.assign_add(1)

        n_true = tf.reduce_sum(y_true)

        is_correct = tf.reduce_all(tf.equal(
            tf.sort(tf.math.top_k(tf.squeeze(y_true), n_true, sorted=False)[1]),
            tf.sort(tf.math.top_k(tf.squeeze(y_pred), n_true, sorted=False)[1])
        ))
        # tf.reduce_any(tf.equal(
        #     tf.math.top_k(tf.squeeze(y_true), n, sorted=True)[1],
        #     tf.math.top_k(tf.squeeze(y_pred), 1, sorted=True)[1]
        # ))

        is_correct = tf.cast(is_correct, tf.float32)
        self.n_correct.assign_add(is_correct)

    def result(self):
        return self.n_correct / self.n_episodes


strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

# %%
k = 5
q = 1
n = 1


def arrange_nshot_kway(x, y):
    Q = x[:, :q]
    S = x[:, q:]

    Qy = y[:, :q]
    Sy = y[:, q:]

    Q = repeat(Q, "k q h w c -> (k kn q) h w c", kn=k * n)
    S = rearrange(S, "k n h w c -> (k n) h w c")
    S = repeat(S, "kn h w c -> (k kn) h w c", k=k)

    Y = tf.reshape(tf.equal(Qy, tf.reshape(Sy, [-1])), [-1])
    Y = tf.cast(Y, tf.int32)

    return (Q, S), Y


TEST_PATH = "/home/crr/datasets/omniglot/test_records"
test_records = glob.glob(os.path.join(TEST_PATH, "*.tfrecord"))
td = InterleaveTFRecordDataset(records=test_records,
                               record_cycle_length=k,
                               n_per_record=n + q,
                               block_bound=True,
                               sample_n_random=True,
                               shuffle=True,
                               seed=42
                               )
td.batch(n + q, drop_remainder=True)
td.map(batch_deserialize_tensor_example_uint8)
td.batch(k, drop_remainder=True)
td.map(arrange_nshot_kway)
td.unbatch()
td.batch(k * n)

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
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                           tf.keras.metrics.Precision()
                           # Nshot_Kway(),
                           # tf.keras.metrics.TopKCategoricalAccuracy()
                           ])

    # %%
    model.load_weights("outputs/vitos_b2-98_drop01_b256_e200_p7_2/weights_0.8121.h5")

model.summary()

# %%
model.evaluate(td.dataset)

# it = iter(td.dataset)
# acc = Nshot_Kway()

# %%
# (Q, S), Y = next(it)
# yhat = model([Q, S])
# acc.update_state(Y, yhat)
# acc.n_episodes
# acc.n_correct
#
# # %%
# fig, ax = plt.subplots(k * n, 2, figsize=(6, 13))
#
# for i in range(k * n):
#     ax[i, 0].imshow(Q[i])
#     ax[i, 1].imshow(S[i])
#     ax[i, 0].axis("off")
#     ax[i, 1].axis("off")
#     ax[i, 0].annotate(str(Y.numpy()[i]), xy=(0, 0), xytext=(-20, 45))
#     ax[i, 1].annotate(str(yhat.numpy()[i]), xy=(0, 0), xytext=(95, 45))
#
# plt.suptitle(acc.result().numpy())
# plt.tight_layout()
# plt.show()
