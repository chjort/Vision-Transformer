import tensorflow as tf
from tensorflow.keras import datasets

from model import ViT
from trainer import Trainer, TrainerConfig

# %%
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# %%
# train_images = tf.cast(train_images.reshape((-1, 3, 32, 32)), dtype=tf.float32)
# test_images = tf.cast(test_images.reshape((-1, 3, 32, 32)), dtype=tf.float32)
train_images = tf.cast(train_images.reshape((-1, 1, 28, 28)), dtype=tf.float32)
test_images = tf.cast(test_images.reshape((-1, 1, 28, 28)), dtype=tf.float32)
train_images, test_images = train_images / 255.0, test_images / 255.0

# %%
train_x = tf.data.Dataset.from_tensor_slices(train_images, )
train_y = tf.data.Dataset.from_tensor_slices(train_labels)
train_dataset = tf.data.Dataset.zip((train_x, train_y))
test_x = tf.data.Dataset.from_tensor_slices(test_images)
test_y = tf.data.Dataset.from_tensor_slices(test_labels)
test_dataset = tf.data.Dataset.zip((test_x, test_y))

# %%
tconf = TrainerConfig(max_epochs=10, batch_size=64, learning_rate=1e-3)

# %%
# sample model config.
# model_config = {"image_size": 32,
#                 "patch_size": 4,
#                 "num_classes": 10,
#                 "dim": 64,
#                 "depth": 3,
#                 "heads": 4,
#                 "mlp_dim": 128}
model_config = {"image_size": 28,
                "patch_size": 4,
                "num_classes": 10,
                "dim": 64,
                "depth": 1,
                "heads": 4,
                "mlp_dim": 128}

# %%
trainer = Trainer(ViT, model_config, train_dataset, len(train_images), test_dataset, len(test_images), tconf)

# %%
trainer.train()
