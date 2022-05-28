import numpy as np
import tensorflow as tf
import utils.utils_random as utils_random
from utils.utils_random import image_sample
from math import ceil


def list_move_left(A, a):
  for i in range(a):
    A.insert(len(A), A[0])
    A.remove(A[0])
  return A


def list_move_right(A, a):
  for i in range(a):
    A.insert(0, A.pop())
  return A


def mask_tensor_smooth(mask_tensor, mask_gen):
  unfold_mask_tensor = tf.extract_image_patches(mask_tensor, ksizes=mask_gen.kernel_size,
                                                strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')

  blurred_mask_tensor_mid = tf.multiply(unfold_mask_tensor, mask_gen.unfold_kernel)

  # blurred_mask_tensor = tf.reduce_sum(blurred_mask_tensor_mid *
  #                                     tf.nn.softmax(blurred_mask_tensor_mid * mask_gen.temperature_param, -1),
  #                                     axis=-1, keepdims=True)
  # blurred_mask_tensor = tf.clip_by_value(blurred_mask_tensor, 0, 1)

  blurred_mask_tensor = tf.reduce_sum(blurred_mask_tensor_mid, axis=-1, keepdims=True)
  # blurred_mask_tensor = tf.nn.sigmoid(blurred_mask_tensor)
  return blurred_mask_tensor


def img_tensor_smooth(img_tensor, num_levels, max_blur=3., ceil_size=3):
  EPSILON_SINGLE = tf.constant(1.19209290E-07, tf.float32)
  blur_img_pyramid_list = []
  for sigma_temp in np.linspace(start=0.0, stop=1.0, num=num_levels, endpoint=True):
    if sigma_temp == 1.0:
      blur_img_pyramid_list.append(img_tensor)
    else:
      sigma = (1 - sigma_temp) * max_blur
      width = ceil(ceil_size * sigma)
      sigma = tf.constant(sigma, tf.float32)

      filt = tf.range(-width - 1, width + 1, dtype=tf.float32)
      filt_temp = tf.math.exp(-(tf.pow(filt, 2) / (2 * tf.pow(tf.cast(sigma + EPSILON_SINGLE, tf.float32), 2))))
      filt_norm2d = tf.pow(tf.reduce_sum(filt_temp), 2)
      gaussian_kernel = tf.tensordot(filt_temp, filt_temp, axes=0) / filt_norm2d
      gaussian_kernel = tf.expand_dims(gaussian_kernel, axis=-1)
      gaussian_kernel = tf.expand_dims(tf.tile(gaussian_kernel, (1, 1, 3)), axis=-1)

      # img_tensor尺寸应为四维
      blurred_img_tensor = tf.nn.depthwise_conv2d(img_tensor, gaussian_kernel, [1, 1, 1, 1], 'SAME')
      blur_img_pyramid_list.append(blurred_img_tensor)
  blur_img_pyramid = tf.stack(blur_img_pyramid_list, 0)
  blur_img_pyramid = tf.transpose(blur_img_pyramid, (1, 0, 2, 3, 4))
  return blur_img_pyramid


def img_tensor_noise(img_tensor, num_levels, flag_decorrelate=True, flag_from_gamma_to_linear=False):
  img_noise_pyramid_list = []

  # sigma_arr = np.linspace(start=0.19, stop=0.2, num=num_levels, endpoint=True)[::-1]
  # decay_power_arr = np.linspace(start=1.3, stop=1.5, num=num_levels, endpoint=True)[::-1]

  sigma_arr = np.linspace(start=0.17, stop=0.2, num=num_levels, endpoint=True)[::-1]
  decay_power_arr = np.linspace(start=1.3, stop=1.5, num=num_levels, endpoint=True)[::-1]

  for i_param in range(num_levels):
    sigma = sigma_arr[i_param]
    decay_power = decay_power_arr[i_param]  # decay_power_arr[i_param]
    noise_img_tensor = image_sample([1, img_tensor.shape[1].value, img_tensor.shape[2].value, 3],
                                    decorrelate=flag_decorrelate,
                                    flag_from_gamma_to_linear=flag_from_gamma_to_linear,
                                    sd=sigma, decay_power=decay_power)
    img_noise_pyramid_list.append(noise_img_tensor)

  # for i_param in range(num_levels):
  #   sigma = sigma_arr[i_param]
  #   decay_power = decay_power_arr[i_param]
  #   noise_img_tensor = image_sample([1, img_tensor.shape[1].value, img_tensor.shape[2].value, 3],
  #                                   decorrelate=flag_decorrelate,
  #                                   flag_from_gamma_to_linear=flag_from_gamma_to_linear,
  #                                   sd=sigma, decay_power=decay_power)
  #   img_noise_pyramid_list.append(noise_img_tensor)

  # for sigma_temp in np.linspace(start=0.0, stop=0.99, num=num_levels, endpoint=True):
  #   # if sigma_temp == 1.0:
  #   #   img_noise_pyramid_list.append(tf.zeros_like(img_tensor))
  #   # else:
  #   sigma = 1 - sigma_temp
  #   noise_img_tensor = image_sample([1, img_tensor.shape[1].value, img_tensor.shape[2].value, 3],
  #                                   decorrelate=flag_decorrelate,
  #                                   flag_from_gamma_to_linear=flag_from_gamma_to_linear,
  #                                   sd=sigma, decay_power=1.5)
  #   img_noise_pyramid_list.append(noise_img_tensor)

  noise_img_pyramid = tf.concat(img_noise_pyramid_list, 0)
  return noise_img_pyramid


def img_tensor_blur_noise(img_tensor, num_levels):
  EPSILON_SINGLE = tf.constant(1.19209290E-07, tf.float32)
  max_blur = 3
  img_blur_noise_pyramid_list = []
  for sigma_temp in np.linspace(start=0.0, stop=1.0, num=num_levels, endpoint=True):
    if sigma_temp == 1.0:
      img_blur_noise_pyramid_list.append(img_tensor)
    else:
      sigma = (1 - sigma_temp) * max_blur
      width = ceil(2 * sigma)
      sigma = tf.constant(sigma, tf.float32)

      filt = tf.range(-width - 1, width + 1, dtype=tf.float32)
      filt_temp = tf.math.exp(-(tf.pow(filt, 2) / (2 * tf.pow(tf.cast(sigma + EPSILON_SINGLE, tf.float32), 2))))
      filt_norm2d = tf.pow(tf.reduce_sum(filt_temp), 2)
      gaussian_kernel = tf.tensordot(filt_temp, filt_temp, axes=0) / filt_norm2d
      gaussian_kernel = tf.expand_dims(gaussian_kernel, axis=-1)
      gaussian_kernel = tf.expand_dims(tf.tile(gaussian_kernel, (1, 1, 3)), axis=-1)

      blurred_img_tensor = tf.nn.depthwise_conv2d(img_tensor, gaussian_kernel, [1, 1, 1, 1], 'SAME')

      noise_img_tensor = image_sample([1, img_tensor.shape[1].value,
                                       img_tensor.shape[2].value, 3], sd=1 - sigma_temp, decay_power=1.5)
      img_blur_noise_tensor = blurred_img_tensor + noise_img_tensor
      # img_blur_noise_tensor = tf.nn.sigmoid(img_blur_noise_tensor)

      img_blur_noise_pyramid_list.append(img_blur_noise_tensor)
  img_blur_noise_pyramid = tf.concat(img_blur_noise_pyramid_list, 0)
  return img_blur_noise_pyramid


def torch_gather(tobe_gathered_tensor, torch_version_indices, gather_axis):
  # if pytorch gather indices are
  # [[[0, 10, 20], [0, 10, 20], [0, 10, 20]],
  #  [[0, 10, 20], [0, 10, 20], [0, 10, 20]]]
  # tf nd_gather needs to be
  # [[0,0,0], [0,0,10], [0,0,20], [0,1,0], [0,1,10], [0,1,20], [0,2,0], [0,2,10], [0,2,20],
  #  [1,0,0], [1,0,10], [1,0,20], [1,1,0], [1,1,10], [1,1,20], [1,2,0], [1,2,10], [1,2,20]]

  # create a tensor containing indices of each element
  all_indices = tf.where(tf.fill(torch_version_indices.shape, True))
  gather_locations = tf.reshape(torch_version_indices, [torch_version_indices.shape.num_elements()])

  # splice in our pytorch style index at the correct axis
  gather_indices = []
  for axis in range(len(torch_version_indices.shape)):
    if axis == gather_axis:
      gather_indices.append(gather_locations)
    else:
      gather_indices.append(all_indices[:, axis])

  gather_indices = tf.stack(gather_indices, axis=-1)
  gathered = tf.gather_nd(tobe_gathered_tensor, gather_indices)
  reshaped = tf.reshape(gathered, torch_version_indices.shape)
  return reshaped


def blur_background_composed(blur_img_pyramid_inner, num_levels, t_alpha_inner):
  batch_size = t_alpha_inner.shape[0]
  t_alpha_temp = tf.reshape(t_alpha_inner, [batch_size, 1, *t_alpha_inner.shape[1:]])
  t_alpha_temp = t_alpha_temp * (num_levels - 1)
  t_alpha_k = tf.floor(t_alpha_temp)
  t_alpha_temp = t_alpha_temp - t_alpha_k
  t_alpha_k = tf.cast(t_alpha_k, tf.int64)

  if len(blur_img_pyramid_inner.get_shape().as_list()) == 4:
    blur_img_pyramid_inner = tf.expand_dims(blur_img_pyramid_inner, 0)
  else:
    blur_img_pyramid_inner = blur_img_pyramid_inner

  t_alpha_k = tf.tile(t_alpha_k, [1, 1, 1, 1, 3])
  t_composed_0 = torch_gather(blur_img_pyramid_inner, t_alpha_k,
                              gather_axis=1)

  t_alpha_k_plusone = tf.clip_by_value(t_alpha_k + 1,
                                       clip_value_min=0,
                                       clip_value_max=num_levels - 1)
  t_composed_1 = torch_gather(blur_img_pyramid_inner, t_alpha_k_plusone, gather_axis=1)
  t_composed_inner = tf.squeeze(((1 - t_alpha_temp) * t_composed_0 +
                                 t_alpha_temp * t_composed_1), axis=1)
  return t_composed_inner


def noise_background_composed(noise_img_pyramid_inner, img_tensor, num_levels, t_alpha_inner):
  batch_size = t_alpha_inner.shape[0]
  t_alpha_temp = tf.reshape(t_alpha_inner, [batch_size, 1, *t_alpha_inner.shape[1:]])
  t_alpha_temp = t_alpha_temp * (num_levels - 1)
  t_alpha_k = tf.floor(t_alpha_temp)
  t_alpha_temp = t_alpha_temp - t_alpha_k
  t_alpha_k = tf.cast(t_alpha_k, tf.int64)

  if len(noise_img_pyramid_inner.get_shape().as_list()) == 4:
    blur_img_pyramid_inner = tf.expand_dims(noise_img_pyramid_inner, 0)
  else:
    blur_img_pyramid_inner = noise_img_pyramid_inner

  t_alpha_k = tf.tile(t_alpha_k, [1, 1, 1, 1, 3])
  t_composed_0 = torch_gather(blur_img_pyramid_inner, t_alpha_k,
                              gather_axis=1)

  t_alpha_k_plusone = tf.clip_by_value(t_alpha_k + 1,
                                       clip_value_min=0,
                                       clip_value_max=num_levels - 1)
  t_composed_1 = torch_gather(blur_img_pyramid_inner, t_alpha_k_plusone, gather_axis=1)
  t_composed_inner = tf.squeeze(((1 - t_alpha_temp) * t_composed_0 +
                                 t_alpha_temp * t_composed_1), axis=1)
  t_composed_inner = t_composed_inner * (1 - t_alpha_inner) + img_tensor * t_alpha_inner
  return t_composed_inner


def gen_composed_img(t_rgb, t_alpha_ori, flag_opt_vis,
                     constant_param, mask_gen):

  if flag_opt_vis == "ConstBack":
    t_alpha = t_alpha_ori
    t_composed = constant_param * (1.0 - t_alpha) + t_rgb * t_alpha

  elif flag_opt_vis == "Noise":
    t_alpha = t_alpha_ori
    t_composed = constant_param * (1.0 - t_alpha) + t_rgb * t_alpha

  elif flag_opt_vis == "Normal":
    t_alpha = t_alpha_ori
    t_composed = constant_param * (1.0 - t_alpha) + t_rgb * t_alpha

  elif flag_opt_vis == "BlurMaskNoise":
    # t_alpha = mask_tensor_smooth(t_alpha_ori, max_blur=0.1, ceil_size=8, temperature_param=30)
    t_alpha = mask_tensor_smooth(t_alpha_ori, mask_gen)
    t_composed = constant_param * (1.0 - t_alpha) + t_rgb * t_alpha

  elif flag_opt_vis == "NoisePyramid":
    t_alpha = t_alpha_ori
    t_composed = noise_background_composed(constant_param, t_rgb, 8, t_alpha)

  elif flag_opt_vis == "BlurImage":
    num_levels_outer = 6
    t_alpha = t_alpha_ori
    blur_t_rgb_pyramid_tensor = img_tensor_smooth(t_rgb, num_levels_outer, max_blur=10, ceil_size=2)
    t_composed = blur_background_composed(blur_t_rgb_pyramid_tensor, num_levels_outer, t_alpha)
  return t_composed, t_alpha
