import glob

import tensorflow as tf

from chambers.data.loader import InterleaveOneshotDataset
from chambers.models.transformer import VisionTransformerOS
from chambers.augmentations import resize

# %%
train_path = "/home/crr/datasets/omniglot/train"
test_path = "/home/crr/datasets/omniglot/test"

train_class_dirs = glob.glob(train_path + "/*/")
train_labels = list(range(len(train_class_dirs)))

test_class_dirs = glob.glob(test_path + "/*/")
test_labels = list(range(len(train_class_dirs), len(train_class_dirs) + len(test_class_dirs)))

n_train = len(train_class_dirs)
n_test = len(test_class_dirs)

# %%
n = 4
train_dataset = InterleaveOneshotDataset(train_class_dirs, train_labels, n, sample_random=True, shuffle=True,
                                         reshuffle_iteration=True, repeats=-1)
train_dataset.map_images(resize, 32, 32)

test_dataset = InterleaveOneshotDataset(test_class_dirs, test_labels, n, sample_random=True, shuffle=True,
                                        reshuffle_iteration=True)
test_dataset.map_images(resize, 32, 32)

train_dataset = train_dataset.dataset
test_dataset = test_dataset.dataset

# %%
# import matplotlib.pyplot as plt
# it = iter(train_dataset)
# x1, x2, y = next(it)
#
# fig, axes = plt.subplots(n * 2, 2)
# for i in range(n * 2):
#     axes[i][0].imshow(x1[i])
#     axes[i][1].imshow(x2[i])
#     axes[i][0].axis("off")
#     axes[i][1].axis("off")
# plt.tight_layout()
# plt.show()

# %%
IMG_SHAPE = (32, 32, 1)
PATCH_SIZE = 4
PATCH_DIM = 64
N_ENCODER_LAYERS = 3
NUM_HEADS = 4
FF_DIM = 128

model = VisionTransformerOS(input_shape=IMG_SHAPE,
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
EPOCHS = 10
hist = model.fit(train_dataset,
                 epochs=EPOCHS,
                 steps_per_epoch=n_train,
                 validation_data=test_dataset,
                 validation_steps=n_test,
                 )

# %%
x1, x2, y = next(iter(test_dataset))

yhat = model([x1, x2])
yhat = tf.argmax(tf.nn.softmax(yhat, axis=-1), axis=-1)[:, 0]
yhat
y
