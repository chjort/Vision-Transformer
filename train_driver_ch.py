import tensorflow as tf
from tensorflow.keras import datasets

from chambers.models.transformer import VisionTransformer

# %%
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
n_train = train_images.shape[0]
n_test = test_images.shape[0]

# %%
train_images = tf.cast(train_images.reshape((-1, 28, 28, 1)), dtype=tf.float32)
test_images = tf.cast(test_images.reshape((-1, 28, 28, 1)), dtype=tf.float32)
train_images, test_images = train_images / 255.0, test_images / 255.0

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
