# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'

import numpy as np
import cv2
import random
import threading


def random_intensity_change_3D(img, max_change):
    img = img.astype(np.float32)
    for j in range(3):
        delta = random.randint(-max_change, max_change)
        img[:, :, :, j] += delta
    img[img < 0] = 0
    img[img > 255] = 255
    return img


def random_rotate_with_mask(image, mask, max_angle):
    cols = image.shape[1]
    rows = image.shape[0]

    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    dst_msk = cv2.warpAffine(mask, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return dst, dst_msk


def random_rotate(image, max_angle):
    cols = image.shape[1]
    rows = image.shape[0]

    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return dst


def get_random_mirror(image):
    # all possible mirroring and flips
    # (in total there are only 8 possible configurations)
    # image must be square for correct output
    mirror = random.randint(0, 1)
    if mirror == 1:
        # flipud
        image = image[::-1, :, :]
    angle = random.randint(0, 3)
    if angle != 0:
        image = np.rot90(image, k=angle)
    return image


class threadsafe_iter(object):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
      self.it = it
      self.lock = threading.Lock()

    def __iter__(self):
      return self

    def __next__(self):
      with self.lock:
          return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
      return threadsafe_iter(f(*a, **kw))
    return g