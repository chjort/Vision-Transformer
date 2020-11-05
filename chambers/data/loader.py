from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from einops import rearrange

from .base_datasets import TensorSliceDataset
from .loader_functions import match_img_files, read_and_decode, match_img_files_triplet
from .mixins import ImageLabelMixin
from ..augmentations import resize


class _TransformSliceDataset(TensorSliceDataset):
    def __init__(self, inputs, repeats=None, shuffle=False, buffer_size=None, reshuffle_iteration=True, seed=None):
        """
            :param inputs: inputs to sample from
            :param repeats: number of times to repeat dataset. Set repeats=-1 to repeat indefinitely.
            :param shuffle: If True the order of classes will be shuffled
            :param buffer_size: Size of the shuffle buffer
            :param reshuffle_iteration: If True, will reshuffle dataset each iteration. Otherwise shuffle only once.
            :param seed: seed for the shuffle
        """

        super().__init__(inputs)
        self.repeats = repeats
        self.do_shuffle = shuffle
        self.buffer_size = buffer_size
        self.reshuffle_iteration = reshuffle_iteration
        self.seed = seed

        if repeats is not None and repeats == -1:
            self.repeat()
        elif repeats is not None and repeats > 0:
            self.repeat(repeats)

        if self.buffer_size is None:
            input_len = self._get_input_len(inputs)
            self.buffer_size = input_len

        if self.do_shuffle:
            self.shuffle(buffer_size=self.buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_iteration)

    @staticmethod
    def _get_input_len(inputs):
        input_ndims = np.ndim(inputs)
        if input_ndims == 1:
            input_len = len(inputs)
        elif input_ndims > 1:
            input_len = len(inputs[0])
        else:
            raise ValueError("Input with 0 dimensions has no length.")

        return input_len


class InterleaveDataset(_TransformSliceDataset, ABC):
    """
        Constructs a tensorflow.data.Dataset which samples inputs by interleaving according to 'self.interleave_map'.
        For more detailed documentation see: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave

        The attribute 'self.dataset' is the tensorflow.data.Dataset producing outputs
    """

    def __init__(self, inputs, cycle_length=-1, block_length=1, repeats=None, shuffle=False,
                 reshuffle_iteration=True, buffer_size=None, seed=None):
        """
            :param inputs: inputs to sample from
            :param cycle_length: number of inputs to iterate through before going to the next 'cycle_length' inputs.
            :param block_length: number of items to iterate through per input before moving to the next input in the
            cycle length.
            :param repeats: number of times to repeat dataset. Set repeats=-1 to repeat indefinitely.
            :param shuffle: If True the order of classes will be shuffled
            :param buffer_size: Size of the shuffle buffer
            :param seed: seed for the shuffle
            """

        super().__init__(inputs, repeats=repeats, shuffle=shuffle, reshuffle_iteration=reshuffle_iteration,
                         buffer_size=buffer_size, seed=seed)

        input_len = self._get_input_len(inputs)
        if cycle_length > input_len:
            raise ValueError(
                "Cycle length {} higher than input length {}. Cycle length must be lower than length of input.".format(
                    cycle_length, input_len)
            )

        self.cycle_length = cycle_length
        self.block_length = block_length

        self.interleave(self.interleave_fn, cycle_length=self.cycle_length,
                        block_length=self.block_length)

    @abstractmethod
    def interleave_fn(self, *args, **kwargs):
        pass


class InterleaveImageDataset(InterleaveDataset, ImageLabelMixin):
    """
        Constructs a tensorflow.data.Dataset which loads images by interleaving through input folders.

        The attribute 'self.dataset' is the tensorflow.data.Dataset producing outputs of (images, labels)
    """

    def __init__(self, image_class_dirs: list, labels: list, class_cycle_length, images_per_class,
                 sample_random=False, repeats=None, shuffle=False, reshuffle_iteration=True, buffer_size=None,
                 seed=None):
        """
        :param image_class_dirs: list of class directories containing image files
        :param labels: list of labels for each class
        :param class_cycle_length: number of classes per cycle
        :param images_per_class: number of images per class in a cycle
        :param sample_random: Boolean. If true, will uniformly sample the images per class at random
        """
        self.sample_random = sample_random
        super().__init__((image_class_dirs, labels), cycle_length=class_cycle_length, block_length=images_per_class,
                         repeats=repeats, shuffle=shuffle, reshuffle_iteration=reshuffle_iteration,
                         buffer_size=buffer_size, seed=seed)

        self.map_image(read_and_decode)

    def get_dir_images(self, input_dir):
        return match_img_files(input_dir)

    def random_upsample(self, x, n):
        n_x = tf.shape(x)[0]
        diff = n - n_x
        random_indices = tf.random.uniform(shape=[diff], minval=0, maxval=n_x, dtype=tf.int32, seed=self.seed)
        extra_samples = tf.gather(x, random_indices)
        x = tf.concat([x, extra_samples], axis=0)
        return x

    @tf.function
    def random_sample(self, x, n, replace=True):
        n_x = tf.shape(x)[0]

        if replace:
            random_indices = tf.random.uniform(shape=[n], minval=0, maxval=n_x, dtype=tf.int32, seed=self.seed)
            x = tf.gather(x, random_indices)
        else:
            tf.assert_less(n, n_x + 1,
                           message="`n` must be less than or equal to length of x, when `replace` is False.")
            uniform_dist = tf.random.uniform(shape=[n_x], minval=0, maxval=1, seed=self.seed)
            _, random_indices = tf.math.top_k(uniform_dist, k=n)
            x = tf.gather(x, random_indices)

        return x

    def block_iter(self, files, label):
        n_files = tf.shape(files)[0]

        if n_files < self.block_length:
            files = self.random_upsample(files, self.block_length)

        if self.sample_random:
            files = self.random_sample(files, self.block_length, replace=False)

        n_files = tf.shape(files)[0]
        labels = tf.tile([label], [n_files])
        return tf.data.Dataset.from_tensor_slices((files, labels))

    @tf.function
    def interleave_fn(self, input_dir, label):
        class_images = self.get_dir_images(input_dir)
        return self.block_iter(class_images, label)


class InterleaveOneshotDataset(InterleaveImageDataset):
    def __init__(self, image_class_dirs: list, labels: list, n: int, sample_random=False, repeats=None,
                 shuffle=False, reshuffle_iteration=True, buffer_size=None, seed=None):
        """
        :param image_class_dirs: list of class directories containing image files
        :param labels: list of labels for each class
        :param sample_random: Boolean. If true, will uniformly sample the images per class at random
        """
        assert ( n >= 2 and n % 2 == 0), "n must be an even number and at least 2."
        super(InterleaveOneshotDataset, self).__init__(image_class_dirs=image_class_dirs,
                                                       labels=labels,
                                                       class_cycle_length=2,
                                                       images_per_class=n,
                                                       sample_random=sample_random,
                                                       repeats=repeats,
                                                       shuffle=shuffle,
                                                       reshuffle_iteration=reshuffle_iteration,
                                                       buffer_size=buffer_size,
                                                       seed=seed)
        self.n = n
        self.batch(self.cycle_length * self.block_length, drop_remainder=True)
        self.map(self.arrange_oneshot)
        self.map(self.split_to_x1_x2_y)

    def map_images(self, func, *args, **kwargs):
        def fn(x1, x2, labels):
            return func(x1, *args, **kwargs), func(x2, *args, **kwargs), labels

        self.map(fn)

    def arrange_oneshot(self, x, y):
        pos = rearrange(x, "(n k) h w c -> n k h w c", k=self.cycle_length, n=self.block_length)
        neg = rearrange(x, "(k n) h w c -> n k h w c", k=self.cycle_length, n=self.block_length)
        x = tf.concat([pos, neg], axis=0)
        y = tf.concat([tf.ones(self.block_length), tf.zeros(self.block_length)], axis=0)
        y = tf.cast(y, tf.int32)
        return x, y

    def split_to_x1_x2_y(self, x, y):
        x1, x2 = tf.split(x, 2, axis=1)
        x1 = tf.squeeze(x1, 1)
        x2 = tf.squeeze(x2, 1)
        return x1, x2, y


class InterleaveTripletDataset(InterleaveImageDataset, ImageLabelMixin):
    def triplet_block_iter(self, anch, pos, neg, label):
        pos = tf.concat([anch, pos], axis=0)

        n_pos = tf.shape(pos)[0]
        n_neg = tf.shape(neg)[0]

        n_pos_block = tf.cast(tf.math.floor(self.block_length / 2), tf.int32)
        n_neg_block = tf.cast(tf.math.ceil(self.block_length / 2), tf.int32)

        if n_pos < n_pos_block:
            pos = self.random_upsample(pos, n_pos_block)
            n_pos = tf.shape(pos)[0]
        elif n_pos > n_pos_block:
            pos = self.random_sample(pos, n_pos_block, replace=False)
            n_pos = tf.shape(pos)[0]

        if n_neg < n_neg_block:
            neg = self.random_upsample(neg, n_neg_block)
            n_neg = tf.shape(neg)[0]
        elif n_neg > n_neg_block:
            neg = self.random_sample(neg, n_neg_block, replace=False)
            n_neg = tf.shape(neg)[0]

        if self.sample_random:
            pos = tf.random.shuffle(pos, seed=self.seed)
            neg = tf.random.shuffle(neg, seed=self.seed)

        triplet_images = tf.concat([pos, neg], axis=0)

        pos_labels = tf.tile([label], [n_pos])
        neg_labels = tf.tile([-1], [n_neg])
        labels = tf.concat([pos_labels, neg_labels], axis=0)

        return tf.data.Dataset.from_tensor_slices((triplet_images, labels))

    @tf.function
    def interleave_fn(self, input_dir, label):
        anch, pos, neg = match_img_files_triplet(input_dir)
        return self.triplet_block_iter(anch, pos, neg, label)


class InterleaveImageTripletDataset(InterleaveTripletDataset, ImageLabelMixin):
    @tf.function
    def interleave_fn(self, input_dir, label):
        img_files = match_img_files(input_dir)

        if tf.shape(img_files)[0] == 0:  # If no images found in folder, assume it is triplet folder
            anch, pos, neg = match_img_files_triplet(input_dir)
            return self.triplet_block_iter(anch, pos, neg, label)
        else:
            return self.block_iter(img_files, label)


class InterleaveTFRecordDataset(InterleaveDataset, ImageLabelMixin):
    """
        Constructs a tensorflow.data.Dataset which loads examples from TF Record files by interleaving through
        the input record files.
    """

    def __init__(self, records: list, record_cycle_length, samples_per_record,
                 sample_random=False, repeats=None, shuffle=False, reshuffle_iteration=True, buffer_size=None, seed=None
                 ):
        """
        :param records: list of TF Record files
        :param record_cycle_length: number of records per cycle
        :param samples_per_record: number of examples per record in a cycle
        :param sample_random: Boolean. If true, will uniformly sample the examples per record at random
        """
        self.sample_random = sample_random
        super().__init__(records, cycle_length=record_cycle_length, block_length=samples_per_record,
                         repeats=repeats, shuffle=shuffle, reshuffle_iteration=reshuffle_iteration,
                         buffer_size=buffer_size, seed=seed)

    @tf.function
    def interleave_fn(self, record):
        td_rec = tf.data.TFRecordDataset(record)
        if self.sample_random:
            td_rec = td_rec.shuffle(buffer_size=100)
            td_rec = td_rec.repeat()
            td_rec = td_rec.take(self.block_length)
        return td_rec


class SequentialImageDataset(_TransformSliceDataset, ImageLabelMixin):
    """
        Constructs a tensorflow.data.Dataset which sequentially loads images from input folders.
    """

    def __init__(self, image_class_dirs: list, labels: list, repeats=None, shuffle=False, reshuffle_iteration=True,
                 buffer_size=None, seed=None):
        """
        :param image_class_dirs: list of class directories containing image files
        :param labels: list of labels for each class
        :param class_cycle_length: number of classes per cycle
        :param images_per_class_cycle: number of images per class in a cycle
        :param sample_random: Boolean. If true, will uniformly sample the images per class at random
        """
        super().__init__((image_class_dirs, labels), repeats=repeats, shuffle=shuffle,
                         reshuffle_iteration=reshuffle_iteration, buffer_size=buffer_size,
                         seed=seed)

        self.flat_map(self.flat_map_fn)
        self.map_image(read_and_decode)

    def make_iter(self, files, label):
        n_files = tf.shape(files)[0]
        y = tf.tile([label], [n_files])
        return tf.data.Dataset.from_tensor_slices((files, y))

    def flat_map_fn(self, input_dir, label):
        files = match_img_files(input_dir)
        return self.make_iter(files, label)


class SequentialTripletDataset(SequentialImageDataset, ImageLabelMixin):
    def triplet_make_iter(self, anch, pos, neg, label):
        pos = tf.concat([anch, pos], axis=0)
        n_pos = tf.shape(pos)[0]
        n_neg = tf.shape(neg)[0]

        triplet_images = tf.concat([pos, neg], axis=0)

        pos_labels = tf.tile([label], [n_pos])
        neg_labels = tf.tile([-1], [n_neg])
        labels = tf.concat([pos_labels, neg_labels], axis=0)

        return tf.data.Dataset.from_tensor_slices((triplet_images, labels))

    def flat_map_fn(self, input_dir, label):
        anch, pos, neg = match_img_files_triplet(input_dir)
        return self.triplet_make_iter(anch, pos, neg, label)


class SequentialImageTripletDataset(SequentialTripletDataset, ImageLabelMixin):
    @tf.function
    def flat_map_fn(self, input_dir, label):
        img_files = match_img_files(input_dir)
        if tf.shape(img_files)[0] == 0:  # If no images found in folder, assume it is triplet folder
            anch, pos, neg = match_img_files_triplet(input_dir)
            return self.triplet_make_iter(anch, pos, neg, label)
        else:
            return self.make_iter(img_files, label)


class SequentialTFRecordDataset(_TransformSliceDataset, ImageLabelMixin):
    """
        Constructs a tensorflow.data.Dataset which sequentially loads examples from TF Record files.
    """

    def __init__(self, records: list, repeats=None, shuffle=False, reshuffle_iteration=True, buffer_size=None,
                 seed=None):
        """
        :param image_class_dirs: list of class directories containing image files
        :param labels: list of labels for each class
        :param class_cycle_length: number of classes per cycle
        :param images_per_class_cycle: number of images per class in a cycle
        :param sample_random: Boolean. If true, will uniformly sample the images per class at random
        """
        super().__init__(records, repeats=repeats, shuffle=shuffle, reshuffle_iteration=reshuffle_iteration,
                         buffer_size=buffer_size,
                         seed=seed)

        self.dataset = tf.data.TFRecordDataset(records)


class EpisodeImageDataset(InterleaveImageDataset, ImageLabelMixin):
    """
        Constructs a tensorflow.data.Dataset which generates batches of 'n-shot, k-way episodes'. Each batch consists of a
        dictionary containing a support set and a query set, and a one-hot encoded tensor as the labels
    """

    def __init__(self, image_class_dirs: list, labels: list, n, k, q, resize_shape=(224, 224)):
        """
        :param image_class_dirs:
        :param labels:
        :param n: Number of images per class in the support set
        :param k: Number of classes in the support set
        :param q: Number of images per class in the query set
        :param resize_shape: Tuple of (height, width) to resize image to
        """

        super().__init__(image_class_dirs, labels, class_cycle_length=k, images_per_class=n + q,
                         sample_random=True, shuffle=True)
        self.n = n
        self.k = k
        self.q = q
        self.map_image(resize, *resize_shape)
        self.batch(self.n + self.q, drop_remainder=True)
        self.batch(self.k, drop_remainder=True)
        self.map(self._get_support_query_y)

    @tf.function
    def _get_support_query_y(self, images):
        """
        Returns a function that splits an interleave batch into a support set and a query set according
        to 'n', 'k' and 'q'.
        :return: Dictionary containing support set and query set and the one-hot encoded labels
        """

        S = images[:, :self.n, ...]
        Q = images[:, -self.q:, ...]
        Y = tf.tile(tf.reshape(tf.range(self.k), [-1, 1]), [1, self.q])
        Y = tf.one_hot(Y, self.k)

        return {"support": S, "query": Q}, Y
