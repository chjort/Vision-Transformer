import tensorflow as tf
import tensorflow_addons as tfa
from einops.layers.tensorflow import Rearrange
from tensorflow.keras import datasets

from chambers.layers.embedding import ConcatEmbedding, LearnedEmbedding1D
from chambers.layers.transformer import Encoder

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
N_PATCHES = (IMG_SHAPE[0] // PATCH_SIZE) * (IMG_SHAPE[1] // PATCH_SIZE)
N_ENCODER_LAYERS = 3
NUM_HEADS = 4
FF_DIM = 128

inputs = tf.keras.layers.Input(IMG_SHAPE)
x = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=PATCH_SIZE, p2=PATCH_SIZE)(inputs)
n_patches = x.shape[1]

x = tf.keras.layers.Dense(PATCH_DIM)(x)
x = ConcatEmbedding(1, PATCH_DIM,
                    side="left",
                    axis=1,
                    initializer=tf.keras.initializers.RandomNormal(),
                    name="add_cls_token")(x)
x = LearnedEmbedding1D(x.shape[1], PATCH_DIM,
                       initializer=tf.keras.initializers.RandomNormal(),
                       name="pos_embedding")(x)
x = Encoder(embed_dim=PATCH_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            num_layers=N_ENCODER_LAYERS,
            dropout_rate=0.0)(x)
x = tf.keras.layers.Cropping1D((0, n_patches))(x)

x = tf.keras.Sequential([
    tf.keras.layers.Dense(FF_DIM, activation=tfa.activations.gelu),
    tf.keras.layers.Dense(NUM_CLASSES)],
    name="mlp_head")(x)

model = tf.keras.models.Model(inputs, x)

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

x.shape

yhat = model(x)
yhat = tf.argmax(tf.nn.softmax(yhat, axis=-1), axis=-1)[:, 0]
yhat
y
