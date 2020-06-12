import abc
import logging
import magic
import os
import random
import numpy as np

import tensorflow as tf


class DataPipeline(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, path, batch_size=32,
                 buffer_size=16,
                 shuffle=True,
                 fliplr=True,
                 flipud=False,
                 rotate=False,
                 brightness=True,
                 contrast=True,
                 hue=True,
                 satu=True,
                 nthreads=4,
                 num_epochs=None):
        self.path = path
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.nthreads = nthreads
        self.num_epochs = num_epochs

        # Data augmentation
        self.fliplr = fliplr
        self.flipud = flipud
        self.rotate = rotate

        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.satu = satu

        self.samples = self._produce_one_sample()

    @abc.abstractmethod
    def _produce_one_sample(self):
        pass

    def _augment_data(self, img, nchan=3):
        """Flip, crop and rotate samples randomly."""

        with tf.name_scope('data_augmentation'):
            if self.fliplr:
                img = tf.image.random_flip_left_right(img, seed=1234)
            if self.flipud:
                img = tf.image.random_flip_up_down(img, seed=3456)
            if self.rotate:
                angle = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32, seed=4567)
                img = tf.case([(tf.equal(angle, 1), lambda: tf.image.rot90(img, k=1)),
                               (tf.equal(angle, 2), lambda: tf.image.rot90(img, k=2)),
                               (tf.equal(angle, 3), lambda: tf.image.rot90(img, k=3))],
                              lambda: img)

            if self.brightness:
                # 随机设置图片的亮度
                img = tf.image.random_brightness(img, max_delta=30)
            if self.contrast:
                # 随机设置图片的对比度
                img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
            if self.hue:
                # 随机设置图片的色度
                img = tf.image.random_hue(img, max_delta=0.3)
            if self.satu:
                # 随机设置图片的饱和度
                img = tf.image.random_saturation(img, lower=0.2, upper=1.8)

            img.set_shape([None, None, nchan])

            return img


class ImageFileDataPipeline(DataPipeline):
    def __init__(self, path, height, width, random_crop, batch_size=32,
                 buffer_size=16,
                 shuffle=True,
                 fliplr=True,
                 flipud=False,
                 rotate=False,
                 brightness=True,
                 contrast=True,
                 hue=True,
                 satu=True,
                 nthreads=4,
                 num_epochs=None):
        self.height = height
        self.width = width
        self.random_crop = random_crop
        super(ImageFileDataPipeline, self).__init__(path, batch_size, buffer_size, shuffle, fliplr, flipud, rotate,
                                                    brightness, contrast, hue, satu, nthreads, num_epochs)

    def _produce_one_sample(self):
        dirname = os.path.dirname(self.path)
        with open(self.path, 'r') as fid:
            flist = [l.strip() for l in fid]

        last_file = dirname + "/" + flist[-1].split(' ')[0]
        last_label = int(flist[-1].split(' ')[1])

        if self.shuffle:
            random.shuffle(flist)

        input_files = [dirname + "/" + f.split(' ')[0] for f in flist]
        labels = [int(f.split(' ')[1]) for f in flist]

        if '16-bit' in magic.from_file(last_file):
            input_dtype = tf.uint16
            input_wl = 65535.0
        else:
            input_wl = 255.0
            input_dtype = tf.uint8

        is_input_jpg = os.path.splitext(last_file)[-1] == '.jpg'

        dataset = tf.data.Dataset.from_tensor_slices((input_files, labels))
        if self.num_epochs:
            dataset = dataset.repeat(self.num_epochs)
        else:
            dataset = dataset.repeat()

        if self.shuffle:
            dataset = dataset.shuffle(3)

        def _parse_function(input_file, label):
            input_file = tf.read_file(input_file)
            # 解码出内存队列的图片内容
            if is_input_jpg:
                im_input = tf.image.decode_jpeg(input_file, channels=3)
            else:
                im_input = tf.image.decode_png(input_file, dtype=input_dtype, channels=3)

            img = tf.to_float(self._augment_data(im_input, 3)) / input_wl

            with tf.name_scope('crop'):
                # 得到图片的长、宽、通道数
                shape = tf.shape(img)
                new_height = tf.to_int32(self.height)
                new_width = tf.to_int32(self.width)
                height_ok = tf.assert_less_equal(new_height, shape[0])
                width_ok = tf.assert_less_equal(new_width, shape[1])
                with tf.control_dependencies([height_ok, width_ok]):
                    if self.random_crop:
                        # 随机裁剪图片
                        img = tf.random_crop(
                            img, tf.stack([new_height, new_width, 3]))
                    else:
                        height_offset = tf.to_int32((shape[0] - new_height) / 2)
                        width_offset = tf.to_int32((shape[1] - new_width) / 2)
                        img = tf.image.crop_to_bounding_box(
                            img, height_offset, width_offset,
                            new_height, new_width)

            return img, label

        self.nsamples = len(flist)

        # prefetch可以用多线程填充缓冲区
        dataset = dataset.map(_parse_function, num_parallel_calls=self.nthreads).prefetch(self.buffer_size).batch(
            self.batch_size)

        iterator = dataset.make_one_shot_iterator()

        image_input, image_label = iterator.get_next()

        # 对输入和输出的图片做归一化
        sample = dict()
        sample['size'] = len(flist)
        sample['image_input'] = image_input
        sample['image_label'] = image_label
        sample['num_cls'] = int(last_label) + 1

        return sample
