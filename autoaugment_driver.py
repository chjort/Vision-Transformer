import tensorflow as tf
from chambers.augmentations.autoaugment import distort_image_with_autoaugment, distort_image_with_randaugment
import matplotlib.pyplot as plt

img_bytes = tf.io.read_file("/home/ch/datasets/left.jpg")
img = tf.image.decode_image(img_bytes, channels=3)

# img_a = distort_image_with_autoaugment(img, augmentation_name="v0")
img_a = distort_image_with_randaugment(img, num_layers=3, magnitude=15)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img)
ax2.imshow(img_a)
plt.show()
