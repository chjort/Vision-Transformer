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


def get_siamese_model(input_shape):
    """
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer="normal", kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer="normal",
                     bias_initializer="normal", kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer="normal",
                     bias_initializer="normal", kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer="normal",
                     bias_initializer="normal", kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer="normal", bias_initializer="normal"))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer="normal")(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


# %%
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
def flatten_batch(x, y):
    x1 = rearrange(x[0], "b n h w c -> (b n) h w c")
    x2 = rearrange(x[1], "b n h w c -> (b n) h w c")
    y = tf.reshape(y, [-1])
    return (x1, x2), y


input_shape = (84, 84, 1)

n = 2
n_pairs = 16 * strategy.num_replicas_in_sync
train_dataset = InterleaveOneshotDataset(train_class_dirs, train_labels, n, sample_random=True, shuffle=True,
                                         reshuffle_iteration=False, repeats=-1)
train_dataset.map_images(resize, input_shape[0], input_shape[1])
train_dataset.map_images(lambda x: x[..., 0:1])
train_dataset.map(lambda x1, x2, y: ((x1, x2), y))
train_dataset.batch(n_pairs)
train_dataset.map(flatten_batch)

test_dataset = InterleaveOneshotDataset(test_class_dirs, test_labels, n, sample_random=True, shuffle=True,
                                        reshuffle_iteration=False, repeats=-1)
test_dataset.map_images(resize, input_shape[0], input_shape[1])
test_dataset.map_images(lambda x: x[..., 0:1])
test_dataset.map(lambda x1, x2, y: ((x1, x2), y))
test_dataset.batch(n_pairs)
test_dataset.map(flatten_batch)

train_dataset = train_dataset.dataset
test_dataset = test_dataset.dataset

# %%
# import matplotlib.pyplot as plt
#
# it = iter(train_dataset)
# (x1, x2), y = next(it)
# print(x1.shape, x2.shape, y.shape)
#
# batch_size = n * 2 * n_pairs
# fig, axes = plt.subplots(batch_size, 2)
# for i in range(batch_size):
#     axes[i][0].imshow(x1[i], cmap="gray")
#     axes[i][1].imshow(x2[i], cmap="gray")
#     axes[i][0].axis("off")
#     axes[i][1].axis("off")
# plt.tight_layout()
# plt.show()

# %%
PATCH_SIZE = 12
# PATCH_SIZE = 7
PATCH_DIM = 128
# PATCH_DIM = 256
N_ENCODER_LAYERS = 8
NUM_HEADS = 8
FF_DIM = 512

with strategy.scope():
    model = VisionTransformerOS(input_shape=input_shape,
                                patch_size=PATCH_SIZE,
                                patch_dim=PATCH_DIM,
                                n_encoder_layers=N_ENCODER_LAYERS,
                                n_heads=NUM_HEADS,
                                ff_dim=FF_DIM)
    # model = get_siamese_model(input_shape)

# LR = 0.0001
LR = 0.00006
model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR),
              loss="binary_crossentropy",
              metrics="accuracy")

model.summary()

# %%
EPOCHS = 100
steps_per_epoch = 200 // strategy.num_replicas_in_sync
validation_steps = 50 // strategy.num_replicas_in_sync

output_dir = "outputs/vitos1"
os.makedirs(output_dir, exist_ok=True)
hist = model.fit(train_dataset,
                 epochs=EPOCHS,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=test_dataset.take(validation_steps),
                 callbacks=[
                     tf.keras.callbacks.CSVLogger(os.path.join(output_dir, "log.csv")),
                     tf.keras.callbacks.ModelCheckpoint(os.path.join(output_dir, "checkpoints", "model.ckpt"))
                 ]
                 )

# %%
# it = iter(test_dataset)
# (x1, x2), y = next(it)
#
# yhat = model([x1, x2])
# yhat = tf.argmax(tf.nn.softmax(yhat, axis=-1), axis=-1)[:, 0]
# yhat
# y
