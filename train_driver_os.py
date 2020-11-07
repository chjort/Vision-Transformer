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
# INPUT_SHAPE = (105, 105, 1)
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
n_pairs = 64 * strategy.num_replicas_in_sync
train_dataset = InterleaveOneshotDataset(class_dirs=train_class_dirs,
                                         labels=train_labels,
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
test_dataset.prefetch(-1)

train_dataset = train_dataset.dataset
test_dataset = test_dataset.dataset

# %%
# PATCH_SIZE = 15
PATCH_SIZE = 12
# PATCH_SIZE = 7
PATCH_DIM = 128
# PATCH_DIM = 256
N_ENCODER_LAYERS = 8
NUM_HEADS = 8
FF_DIM = 512

with strategy.scope():
    model = VisionTransformerOS(input_shape=INPUT_SHAPE,
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
                 ]
                 )

"""
200/200 [==============================] - 20s 102ms/step - loss: 0.6929 - accuracy: 0.5091 - val_loss: 0.6910 - val_accuracy: 0.5206
Epoch 2/100
200/200 [==============================] - 19s 93ms/step - loss: 0.6916 - accuracy: 0.5213 - val_loss: 0.6945 - val_accuracy: 0.5303
Epoch 3/100
200/200 [==============================] - 19s 93ms/step - loss: 0.6915 - accuracy: 0.5236 - val_loss: 0.6905 - val_accuracy: 0.5200
"""
