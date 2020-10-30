import glob
import tensorflow as tf
from tensorflow.keras import datasets

from chambers.models.transformer import VisionTransformer
from chambers.data.loader import InterleaveImageDataset

# %%
train_path = "/home/crr/datasets/omniglot/train"
test_path = "/home/crr/datasets/omniglot/test"

train_class_dirs = glob.glob(train_path + "/*/")
train_labels = list(range(len(train_class_dirs)))

test_class_dirs = glob.glob(test_path + "/*/")
test_labels = list(range(len(train_class_dirs), len(train_class_dirs) + len(test_class_dirs)))

# %%
# TODO: Binary Match/Non-match dataset
# class_cycle_length = 16
# images_per_class = 5
# train_td = InterleaveImageDataset(image_class_dirs=train_class_dirs,
#                                   labels=train_labels,
#                                   class_cycle_length=class_cycle_length,
#                                   images_per_class=images_per_class,
#                                   sample_random=True,
#                                   shuffle=True)
# train_td.batch(class_cycle_length * images_per_class)
#
# x, y = next(iter(train_td.dataset))


# %%
train_x = tf.data.Dataset.from_tensor_slices(train_images, )
train_y = tf.data.Dataset.from_tensor_slices(train_labels)
train_dataset = tf.data.Dataset.zip((train_x, train_y))
test_x = tf.data.Dataset.from_tensor_slices(test_images)
test_y = tf.data.Dataset.from_tensor_slices(test_labels)
test_dataset = tf.data.Dataset.zip((test_x, test_y))

# %%
IMG_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
PATCH_SIZE = 4
PATCH_DIM = 64
N_ENCODER_LAYERS = 3
NUM_HEADS = 4
FF_DIM = 128

model = VisionTransformer(input_shape=IMG_SHAPE,
                          n_classes=NUM_CLASSES,
                          patch_size=PATCH_SIZE,
                          patch_dim=PATCH_DIM,
                          n_encoder_layers=N_ENCODER_LAYERS,
                          n_heads=NUM_HEADS,
                          ff_dim=FF_DIM)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics="accuracy")

model.summary()

# %%
BATCH_SIZE = 64
EPOCHS = 10

hist = model.fit(train_dataset.batch(BATCH_SIZE).repeat(),
                 epochs=EPOCHS,
                 steps_per_epoch=n_train // BATCH_SIZE,
                 validation_data=test_dataset.batch(BATCH_SIZE),
                 validation_steps=n_test // BATCH_SIZE,
                 )

# %%
x, y = next(iter(test_dataset.batch(BATCH_SIZE)))

yhat = model(x)
yhat = tf.argmax(tf.nn.softmax(yhat, axis=-1), axis=-1)[:, 0]
yhat
y
