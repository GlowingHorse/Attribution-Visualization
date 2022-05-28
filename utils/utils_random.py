import tensorflow as tf
import numpy as np


color_corr_decomp_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")
normed_color_decomp_sqrt = np.max(np.linalg.norm(color_corr_decomp_sqrt, axis=0))


def from_linear_to_gamma(a):
    return a ** (1.0 / 2.2)


def from_gamma_to_linear(a):
    return a ** 2.2


def _linear_decorelate_color(t):
    # check that inner dimension is 3?
    flatten_t = tf.reshape(t, [-1, 3])
    normed_color_corr = color_corr_decomp_sqrt / normed_color_decomp_sqrt
    flatten_t = tf.matmul(flatten_t, normed_color_corr.T)
    return tf.reshape(flatten_t, tf.shape(t))


def clip_to_valid_rgb(t, decorrelate=False, flag_from_gamma_to_linear=False):
    if decorrelate:
        t = _linear_decorelate_color(t)
    t = tf.clip_by_value(t, 0, 1)
    if flag_from_gamma_to_linear:
        t = from_gamma_to_linear(t)
    return t


def to_valid_rgb(t, decorrelate=False, flag_from_gamma_to_linear=False):
    if decorrelate:
        t = _linear_decorelate_color(t)
    t = tf.nn.sigmoid(t)
    if flag_from_gamma_to_linear:
        t = from_gamma_to_linear(t)
    return t


def image_sample(shape, decorrelate=True, flag_from_gamma_to_linear=False, sd=None, decay_power=1):
    raw_spatial = gen_freq_img(shape, sd, decay_power)
    return to_valid_rgb(raw_spatial, decorrelate=decorrelate, flag_from_gamma_to_linear=flag_from_gamma_to_linear)


def gen_freq_img(shape, sd=None, decay_power=None, random_seed=0):
    b, h, w, ch = shape
    sd = 0.01 if sd is None else sd

    imgs = []
    for _ in range(b):
        if random_seed > 0:
            np.random.seed(random_seed)
        sampled_fourier = _rfft2d_freqs(h, w)
        fh, fw = sampled_fourier.shape
        spectrum_tensor_var = tf.random_normal([2, ch, fh, fw], dtype="float32") * sd
        spectrum = tf.complex(spectrum_tensor_var[0], spectrum_tensor_var[1])

        scale_factor = 1.0 / np.maximum(1.0 / max(h, w), sampled_fourier) ** decay_power
        scale_factor = np.sqrt(w * h) * scale_factor
        scaled_spectrum = spectrum * scale_factor
        # scaled_spectrum_value = scaled_spectrum.eval()
        # scaled_spectrum_value = np.absolute(scaled_spectrum_value)

        img = tf.spectral.irfft2d(scaled_spectrum)
        img = img[:ch, :h, :w]
        img = tf.transpose(img, [1, 2, 0])
        imgs.append(img)
    return tf.stack(imgs) / 4.


def fft_image(shape, sd=None, decay_power=1, random_seed=0):
    b, h, w, ch = shape
    sd = sd or 0.01

    imgs = []
    for _ in range(b):
        if random_seed > 0:
            np.random.seed(random_seed)
        sampled_fourier = _rfft2d_freqs(h, w)
        fh, fw = sampled_fourier.shape

        spectrum_tensor_np = np.random.randn(2, ch, fh, fw).astype("float32") * sd
        spectrum_tensor_var = tf.Variable(spectrum_tensor_np)
        spectrum = tf.complex(spectrum_tensor_var[0], spectrum_tensor_var[1])

        scale_factor = 1.0 / np.maximum(1.0 / max(h, w), sampled_fourier) ** decay_power
        # Scale the spectrum by the square-root of the number of pixels
        # to get a unitary transformation. This allows to use similar
        # leanring rates to pixel-wise optimisation.
        scale_factor = np.sqrt(w*h) * scale_factor
        scaled_spectrum = spectrum * scale_factor
        img = tf.spectral.irfft2d(scaled_spectrum)
        # in case of odd input dimension we cut off the additional pixel
        # we get from irfft2d length computation
        img = img[:ch, :h, :w]
        img = tf.transpose(img, [1, 2, 0])
        imgs.append(img)
    stacked_imgs = tf.stack(imgs) / 4.0
    return stacked_imgs


def _rfft2d_freqs(h, w):
    """Compute 2d spectrum frequences."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[:w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[:w // 2 + 1]
    fx_square = np.multiply(fx, fx)
    fy_square = np.multiply(fy, fy)
    struct_fft2d_freq = np.sqrt(fx_square + fy_square)
    return struct_fft2d_freq
