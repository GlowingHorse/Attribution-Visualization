
from __future__ import absolute_import, division, print_function
from future.standard_library import install_aliases
install_aliases()
from builtins import range

import numpy as np
import tensorflow as tf
import logging

from lucid.optvis import objectives, param, transform

from utils.utils_composed_img import gen_composed_img
from utils.utils_random import image_sample, to_valid_rgb
from math import ceil
import cv2 as cv


# create logger with module name, e.g. lucid.misc.io.reading
log = logging.getLogger(__name__)


def list_move_left(A, a):
  for i in range(a):
    A.insert(len(A), A[0])
    A.remove(A[0])
  return A


def list_move_right(A, a):
  for i in range(a):
    A.insert(0, A.pop())
  return A


class MaskGenerator:
  def __init__(self, max_blur, ceil_size, temperature_param):
    EPSILON_SINGLE = tf.constant(1.19209290E-07, tf.float32)
    width = ceil(ceil_size * max_blur)
    sigma = tf.constant(max_blur, tf.float32)

    filt = tf.range(-width, width + 1, dtype=tf.float32)
    filt_temp = tf.math.exp(-(tf.pow(filt, 2) / (2 * tf.pow(tf.cast(sigma + EPSILON_SINGLE, tf.float32), 2))))

    filt_norm2d = tf.pow(tf.reduce_sum(filt_temp), 2)
    gaussian_kernel = tf.tensordot(filt_temp, filt_temp, axes=0) / filt_norm2d

    # gaussian_kernel = tf.tensordot(filt_temp, filt_temp, axes=0)

    gaussian_kernel = tf.expand_dims(gaussian_kernel, axis=-1)
    gaussian_kernel = tf.expand_dims(tf.tile(gaussian_kernel, (1, 1, 1)), axis=-1)

    kernel_size = gaussian_kernel.shape.as_list()
    self.kernel_size = list_move_right(kernel_size, 1)

    unfold_kernel = tf.reshape(gaussian_kernel,
                               [1, 1, kernel_size[0] * kernel_size[1] * kernel_size[2], kernel_size[3]])
    unfold_kernel = tf.transpose(unfold_kernel, [0, 3, 1, 2])
    self.unfold_kernel = tf.tile(unfold_kernel, [1, 1, 1, 1])
    self.temperature_param = temperature_param


def render_vis(model, objective_f, param_f=None, optimizer=None,
               transforms=None, thresholds=(512,), print_objectives=None,
               verbose=True, use_fixed_seed=False,
               transparent_flag=True, draw_loss_flag=False, flag_opt_vis=None,
               init_learning_rate=0.5, attrs=None, lambda_param=None,
               batch_size=None, masksize=1):
  with tf.Graph().as_default() as graph, tf.Session() as sess:

    if use_fixed_seed:  # does not mean results are reproducible, see Args doc
      tf.set_random_seed(0)

    mask_gen = None

    if flag_opt_vis is not None and attrs is not None:

      if batch_size is None:
        batch_size = len(attrs)
      if flag_opt_vis == "ConstBack":
        const_param = 0.55

      elif flag_opt_vis == "Normal":
        const_param = tf.clip_by_value(
          tf.Variable(tf.random_normal(
            [batch_size, model.image_shape[0], model.image_shape[1], 3],
            dtype="float32")),
          0, 1)

      elif flag_opt_vis == "Noise":
        const_param = image_sample([batch_size, model.image_shape[0], model.image_shape[1], 3],
                                   sd=0.19, decay_power=1.5, decorrelate=True, flag_from_gamma_to_linear=False)

      elif flag_opt_vis == "BlurMaskNoise":
        const_param = image_sample([batch_size, model.image_shape[0], model.image_shape[1], 3],
                                   sd=0.2, decay_power=1., decorrelate=True, flag_from_gamma_to_linear=False)
        mask_gen = MaskGenerator(max_blur=0.1, ceil_size=3, temperature_param=20)

      elif flag_opt_vis == "NoisePyramid":
        num_levels = 6
        # sigma_arr = np.linspace(start=0.17, stop=0.2, num=num_levels, endpoint=True)[::-1]
        # decay_power_arr = np.linspace(start=1.3, stop=1.5, num=num_levels, endpoint=True)[::-1]
        sigma_arr = np.linspace(start=0.15, stop=0.2, num=num_levels, endpoint=True)[::-1]
        decay_power_arr = np.linspace(start=1.0, stop=1.5, num=num_levels, endpoint=True)[::-1]
        img_noise_pyramid_list = []

        for i_batch_size in range(batch_size):
          img_noise_pyramid_batch_list = []
          for i_param in range(num_levels):
            sigma = sigma_arr[i_param]
            decay_power = decay_power_arr[i_param]  # decay_power_arr[i_param]
            noise_img_tensor = image_sample([1, model.image_shape[0], model.image_shape[1], 3],
                                            decorrelate=True,
                                            flag_from_gamma_to_linear=False,
                                            sd=sigma, decay_power=decay_power)
            img_noise_pyramid_batch_list.append(noise_img_tensor)
          const_param_temp = tf.concat(img_noise_pyramid_batch_list, 0)
          img_noise_pyramid_list.append(const_param_temp)
        const_param = tf.stack(img_noise_pyramid_list, 0)

      else:
        const_param = image_sample([batch_size, model.image_shape[0], model.image_shape[1], 3],
                                   sd=0.2, decay_power=1.5, decorrelate=False, flag_from_gamma_to_linear=False)

      T = make_vis_T_with_flag(model, objective_f, param_f,
                               optimizer, transforms,
                               transparent_flag, flag_opt_vis,
                               const_param, mask_gen, init_learning_rate, max(thresholds),
                               attrs=attrs, lambda_param=lambda_param, masksize=masksize)
      print_objective_func = make_print_objective_func(print_objectives, T)
      loss, vis_op, t_image, learing_rate = T("loss"), T("vis_op"), T("input"), T("learning_rate")

      # blurred_image = T("blur_t_rgb_pyramid_tensor")

      tf.global_variables_initializer().run()

      losses = []
      images = []

      # writer = tf.summary.FileWriter('./graph_vis/', sess.graph)
      # writer.add_graph(tf.get_default_graph())
      # writer.close()

      try:
        for i in range(max(thresholds)+1):
          loss_, _ = sess.run([loss, vis_op])
          # aaa = t_image.eval()[0, 0:200, 0:200, 0]
          if i in thresholds:
            vis = t_image.eval()
            images.append(vis)
            if verbose:
              print("Iter {}, loss is {:.3f}, lr is {:.3f}".format(i, loss_, learing_rate.eval()))
              print_objective_func(sess)
          if draw_loss_flag:
            losses.append(loss_)
      except KeyboardInterrupt:
        log.warning("Interrupted optimization at step {:d}.".format(i+1))
        vis = t_image.eval()

    else:
      # frozen_variables = [v for v in tf.trainable_variables() if 'CPPN_STACK1' in v.name]
      # tmp_frozen_variables_np = sess.run(frozen_variables)
      # print(np.allclose(tmp_frozen_variables_np, sess.run(frozen_variables)))

      T = make_vis_T(model, objective_f, param_f, optimizer, transforms)
      print_objective_func = make_print_objective_func(print_objectives, T)
      loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
      # blurred_image = T("blur_t_rgb_pyramid_tensor")
      tf.global_variables_initializer().run()
      losses = []
      images = []
      try:
        for i in range(max(thresholds)+1):
          loss_, _ = sess.run([loss, vis_op])
          # aaa = t_image.eval()[0, 0:200, 0:200, 0]
          if i in thresholds:
            vis = t_image.eval()
            images.append(vis)
            if verbose:
              print("Iter {}, loss is {:.3f}".format(i, loss_))
              print_objective_func(sess)
          if draw_loss_flag:
            losses.append(loss_)
      except KeyboardInterrupt:
        log.warning("Interrupted optimization at step {:d}.".format(i+1))
        vis = t_image.eval()

    if draw_loss_flag:
      return images, losses
    # noise_t_rgb_pyramid_tensor = T("noise_t_rgb_pyramid_tensor").eval()
    # t_crop_rgb = T("t_crop_rgb").eval()
    return images


def make_vis_T_with_flag(model, objective_f, param_f=None,
                         optimizer=None, transforms=None, transparent_flag=False,
                         flag_opt_vis=None, const_param=None,
                         mask_gen=None, init_learning_rate=None,
                         iter_steps=None,
                         attrs=None, lambda_param=None, masksize=1):
  # pylint: disable=unused-variable
  t_image = make_t_image(param_f)
  objective_f = objectives.as_objective(objective_f)
  transform_f = make_transform_f(transforms)
  # optimizer = make_optimizer(optimizer, [])

  if transparent_flag:
    t_rgb = t_image[..., :3]
    t_alpha_ori = t_image[..., 3:]
    t_composed, t_alpha = gen_composed_img(t_rgb, t_alpha_ori, flag_opt_vis,
                                           const_param, mask_gen)

    t_composed = tf.concat([t_composed, t_alpha], -1)
    t_transformed = transform_f(t_composed)
    t_crop_rgb, t_crop_alpha = t_transformed[..., :3], t_transformed[..., 3:]

  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.train.exponential_decay(init_learning_rate,
                                             global_step=global_step,
                                             decay_steps=int(iter_steps/10),
                                             decay_rate=0.8)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # ---------------------------------------------
  # can be 15. 10. for Googlenet
  # can be 25 for Resnet

  init_lambda_param = np.sum(attrs, (1, 2, 3)) * 20
  init_lambda_param = init_lambda_param.astype(np.float32)

  # lambda_param = tf.train.exponential_decay(init_lambda_param,
  #                                           global_step=global_step,
  #                                           decay_steps=int(iter_steps/10),
  #                                           decay_rate=5)
  lambda_param = tf.train.exponential_decay(init_lambda_param,
                                            global_step=global_step,
                                            decay_steps=int(iter_steps/10),
                                            decay_rate=1.23)
  # ---------------------------------------------
  # for Resnet
  # optimizer = tf.train.AdamOptimizer(learning_rate)
  # lambda_param = tf.train.exponential_decay(20.,
  #                                           global_step=global_step,
  #                                           decay_steps=int(iter_steps/10),
  #                                           decay_rate=1.23)

  init_global_step = tf.variables_initializer([global_step])
  init_global_step.run()

  if transparent_flag:
    T = import_model(model, t_crop_rgb, t_image)
  else:
    T = import_model(model, transform_f(t_image), t_image)

  if transparent_flag:
    loss_temp = objective_f(T)

    # ----------------------------------------------------------------------------------
    # For Googlenet
    if len(attrs.shape) == 3:
      attr_heat_maps_b = np.sum(attrs, -1)
    else:
      attr_heat_maps_b = np.sum(attrs, -1)

    mask_area_betas = []
    contrain_mask_l = []

    # Otsu mask constraint
    for i_batch in range(attrs.shape[0]):
      attr_heat_maps = attr_heat_maps_b[i_batch]
      attr_heat_maps_scale = (attr_heat_maps - np.min(attr_heat_maps)) / \
                             (np.max(attr_heat_maps) - np.min(attr_heat_maps))
      attr_heat_maps_scale = np.array(cv.resize(attr_heat_maps_scale, (224, 224)), dtype=np.float32)

      attr_heat_maps_scale = (255 * attr_heat_maps_scale).astype(np.uint8)
      attr_heat_maps_scale = cv.GaussianBlur(attr_heat_maps_scale, (3, 3), 0)
      _, th3 = cv.threshold(attr_heat_maps_scale, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

      # spatial mask size
      # attr_heat_maps_scale = np.array(cv.resize(attr_heat_maps_scale, (224, 224)), dtype=np.float32)
      # attr_heat_maps_scale = attr_heat_maps_scale / 255.

      contrain_mask_spatial = th3.copy()
      contrain_mask_spatial[contrain_mask_spatial > 200] = 1.
      contrain_mask_spatial = np.expand_dims(contrain_mask_spatial, -1)
      contrain_mask_l.append(contrain_mask_spatial)

      # mean mask size
      counts = np.count_nonzero(th3)
      mask_area_beta = counts / (th3.shape[0] * th3.shape[1])
      mask_area_beta *= masksize
      # if mask_area_beta > 0.1:
      mask_area_beta /= 2
      mask_area_betas.append(mask_area_beta)

    sum_to_batch_attrs = np.sum(attrs, (1, 2, 3))

    t_alpha_mean_crop = tf.reduce_mean(t_crop_alpha, axis=(1, 2, 3))
    t_alpha_mean_full = tf.reduce_mean(t_alpha_ori, axis=(1, 2, 3))

    # # vector mask size
    # one_number = int(224*224*(mask_area_beta/2))
    # zero_number = int(224*224 - one_number)
    # contrain_mask_vec = tf.concat([tf.ones(one_number, dtype=tf.dtypes.float32),
    #                               tf.zeros(zero_number, dtype=tf.dtypes.float32)], 0)

    # mean mask size
    mask_area_betas = np.stack(mask_area_betas, 0)
    mask_area_betas = mask_area_betas.astype(np.float32)

    # another vector mask size
    # t_crop_alpha_flat = tf.reshape(t_crop_alpha, [224 * 224, ])
    # t_alpha_ori_flat = tf.reshape(t_alpha_ori, [224 * 224, ])
    # t_crop_alpha_flat = tf.sort(t_crop_alpha_flat, direction='ASCENDING')
    # t_alpha_ori_flat = tf.sort(t_alpha_ori_flat, direction='ASCENDING')

    # # spatial mask size
    # contrain_mask_vec = np.stack(contrain_mask_l, 0)
    # contrain_mask_vec = contrain_mask_vec.astype(np.float32)
    # contrain_mask_vec = np.mean(contrain_mask_vec, (1,2,3))

    # tf.losses.add_loss(-loss_temp + tf.reduce_sum(lambda_param *
    #                                               tf.square(
    #                                                 tf.reduce_mean(t_crop_alpha - contrain_mask_vec,
    #                                                                axis=(1, 2, 3))))
    #                    )
    # tf.losses.add_loss(-loss_temp + tf.reduce_sum(lambda_param *
    #                                               tf.square(
    #                                                 tf.reduce_mean(t_alpha_ori - contrain_mask_vec,
    #                                                                axis=(1, 2, 3))))
    #                    )

    # # mean mask size
    # tf.losses.add_loss(-loss_temp + tf.reduce_sum(sum_to_batch_attrs *
    #                                               lambda_param *
    #                                               tf.square(t_alpha_mean_crop - mask_area_betas)))
    # tf.losses.add_loss(-loss_temp + tf.reduce_sum(sum_to_batch_attrs *
    #                                               lambda_param *
    #                                               tf.square(t_alpha_mean_full - mask_area_betas)))

    # Visualization loss functions
    tf.losses.add_loss(-loss_temp + tf.reduce_sum(lambda_param *
                                                  tf.square(t_alpha_mean_crop - mask_area_betas)))
    tf.losses.add_loss(-loss_temp + tf.reduce_sum(lambda_param *
                                                  tf.square(t_alpha_mean_full - mask_area_betas)))

    loss = tf.losses.get_total_loss()
    # ----------------------------------------------------------------------------------
    vis_op = optimizer.minimize(loss, global_step=global_step)
  else:
    loss = objective_f(T)
    vis_op = optimizer.minimize(loss, global_step=global_step)

  local_vars = locals()
  # pylint: enable=unused-variable

  def T2(name):
    if name in local_vars:
      return local_vars[name]
    else: return T(name)

  return T2


def make_print_objective_func(print_objectives, T):
  print_objectives = print_objectives or []
  po_descriptions = [obj.description for obj in print_objectives]
  pos = [obj(T) for obj in print_objectives]

  def print_objective_func(sess):
    pos_results = sess.run(pos)
    for k, v, i in zip(po_descriptions, pos_results, range(len(pos_results))):
      print("{:02d}: {}: {:7.2f}".format(i+1, k, v))

  return print_objective_func

# pylint: enable=invalid-name


def make_t_image(param_f):
  if param_f is None:
    t_image = param.image(128)
  elif callable(param_f):
    t_image = param_f()
  elif isinstance(param_f, tf.Tensor):
    t_image = param_f
  else:
    raise TypeError("Incompatible type for param_f, " + str(type(param_f)) )
  return t_image


def make_transform_f(transforms):
  if type(transforms) is not list:
    transforms = transform.standard_transforms
  transform_f = transform.compose(transforms)
  return transform_f


def make_optimizer(optimizer, args):
  if optimizer is None:
    return tf.train.AdamOptimizer(0.05)
  elif callable(optimizer):
    return optimizer(*args)
  elif isinstance(optimizer, tf.train.Optimizer):
    return optimizer
  else:
    raise ("Could not convert optimizer argument to usable optimizer. "
           "Needs to be one of None, function from (graph, sess) to "
           "optimizer, or tf.train.Optimizer instance.")


def import_model(model, t_image, t_image_raw):

  model.import_graph(t_image, scope="import", forget_xy_shape=True)

  def T(layer):
    if layer == "input": return t_image_raw
    if layer == "labels": return model.labels
    if layer == "input_img_tensor:0": return t_image.graph.get_tensor_by_name(layer)
    return t_image.graph.get_tensor_by_name("import/%s:0"%layer)

  return T


# From lucid lib
def make_vis_T(model, objective_f, param_f=None, optimizer=None, transforms=None):
  # pylint: disable=unused-variable
  t_image = make_t_image(param_f)
  objective_f = objectives.as_objective(objective_f)
  transform_f = make_transform_f(transforms)
  optimizer = make_optimizer(optimizer, [])

  global_step = tf.train.get_or_create_global_step()
  init_global_step = tf.variables_initializer([global_step])
  init_global_step.run()

  T = import_model(model, transform_f(t_image), t_image)

  loss = objective_f(T)
  vis_op = optimizer.minimize(loss, global_step=global_step)

  local_vars = locals()
  # pylint: enable=unused-variable

  def T2(name):
    if name in local_vars:
      return local_vars[name]
    else: return T(name)

  return T2