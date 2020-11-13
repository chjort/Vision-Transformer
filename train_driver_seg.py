import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from einops import rearrange

from chambers.models.transformer import VisionTransformerSeg


def rgb_to_onehot(m, class_labels):
    return tf.cast(tf.equal(m, class_labels), tf.int32)


def onehot_to_rgb(onehot, class_labels):
    tf_class_labels = tf.constant(class_labels, dtype=tf.uint8)
    class_indices = tf.argmax(onehot, axis=-1)
    class_indices = tf.reshape(class_indices, [-1])
    rgb_image = tf.gather(tf_class_labels, class_indices)
    rgb_image = tf.reshape(rgb_image, [tf.shape(onehot)[0], tf.shape(onehot)[1], 1])
    return rgb_image


# strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

# %%
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
NUM_CLASSES = info.features["label"].num_classes
MASK_LABELS = [1, 2, 3]
INPUT_SHAPE = (128, 128, 3)


def preprocess(sample):
    x = sample["image"]
    m = sample["segmentation_mask"]
    y = sample["label"]

    x = tf.image.resize(x, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    x = x / 255.

    m = tf.cast(tf.image.resize(m, (INPUT_SHAPE[0], INPUT_SHAPE[1])), tf.int32)
    m = rgb_to_onehot(m, MASK_LABELS)

    y = tf.one_hot(y, NUM_CLASSES)

    return x, m, y


train_td = dataset["train"]
train_td = train_td.map(preprocess)
train_td = train_td.batch(16)

# %%
# it = iter(train_td)
# x, m, y = next(it)
# x.shape, m.shape
#
#
# patch_size = 8
# xr = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
# xr.shape
#
# h = int(INPUT_SHAPE[0] / patch_size)
# w = int(INPUT_SHAPE[1] / patch_size)
# xh = rearrange(xr, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', p1=patch_size, p2=patch_size, h=h, w=w)
# xh.shape
#
# import matplotlib.pyplot as plt
#
# plt.imshow(x[0])
# plt.show()
# plt.imshow(m[0, ..., 0])
# plt.show()
# plt.imshow(xh[0])
# plt.show()

# %%
EPOCHS = 200
STEPS_PER_EPOCH = 200 // strategy.num_replicas_in_sync
LR = 0.00006

PATCH_SIZE = 8
PATCH_DIM = 128
N_ENCODER_LAYERS = 8
NUM_HEADS = 8
FF_DIM = 512

with strategy.scope():
    model = VisionTransformerSeg(input_shape=INPUT_SHAPE,
                                 n_classes=NUM_CLASSES,
                                 n_mask_classes=len(MASK_LABELS),
                                 patch_size=PATCH_SIZE,
                                 patch_dim=PATCH_DIM,
                                 n_encoder_layers=N_ENCODER_LAYERS,
                                 n_heads=NUM_HEADS,
                                 ff_dim=FF_DIM,
                                 dropout_rate=0.0
                                 )

    optimizer = tfa.optimizers.AdamW(learning_rate=LR,
                                     weight_decay=1e-4,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-8,
                                     amsgrad=False)
    # model.compile(optimizer=optimizer,
    #               loss=tf.keras.losses.BinaryCrossentropy(),
    #               metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC()])

model.summary()

#%%
it = iter(train_td)
x, m, y = next(it)
x.shape, m.shape

cls, mask = model(x)
cls.shape, mask.shape

# %%
# hist = model.fit(train_dataset,
#                  epochs=EPOCHS,
#                  steps_per_epoch=STEPS_PER_EPOCH,
#                  validation_data=test_dataset,
#                  )
