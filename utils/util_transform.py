import random
import tensorflow as tf
import math


# random_shear(11)
def random_shear(intensity_set):
    def inner(t_image):
        intensity1 = random.randint(-intensity_set, intensity_set)
        x_intensity = math.tan(intensity1*math.pi/180)
        intensity2 = random.randint(-intensity_set, intensity_set)
        y_intensity = math.tan(intensity2*math.pi/180)
        forward_transform = [[1.0,x_intensity,0],
                             [y_intensity,1.0,0],
                             [0,0,1.0]]
        t = tf.contrib.image.matrices_to_flat_transforms(tf.linalg.inv(forward_transform))
        # please notice that forward_transform must be a float matrix,
        # e.g. [[2.0,0,0],[0,1.0,0],[0,0,1]] will work
        # but [[2,0,0],[0,1,0],[0,0,1]] will not
        imgOut = tf.contrib.image.transform(t_image, t, interpolation="BILINEAR", name=None)
        return imgOut

    return inner


def _rand_select(xs, seed=None):
    xs_list = list(xs)
    rand_n = tf.random_uniform((), 0, len(xs_list), "int32", seed=seed)
    return tf.constant(xs_list)[rand_n]


def random_crop(batch_num, height, width, channel_num):
    """Ensures the specified spatial shape by either padding or cropping.
    Meant to be used as a last transform for architectures insisting on a specific
    spatial shape of their inputs.
    """
    def inner(t_image):
        return tf.random_crop(t_image, [batch_num, height, width, channel_num])
    return inner

