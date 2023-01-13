"""
utils.py including misc functions, e.g.,
1 the class of matrix decomposition method
2 the class of clusterring method
3 some plot figure operators
4 Image preconditioning method for generating random image
  using different distribution,
  and decorrelate image color space.
"""
from lucid.optvis.param.resize_bilinear_nd import resize_bilinear_nd
import matplotlib.pyplot as plt
from re import findall
import numpy as np
import sklearn.decomposition
import sklearn.cluster
import tensorflow as tf
from decorator import decorator
import lucid.optvis.objectives as objectives
from skimage import feature, transform
from math import ceil
from utils.utils_random import from_linear_to_gamma, from_gamma_to_linear


def _make_arg_str(arg):
  arg = str(arg)
  too_big = len(arg) > 15 or "\n" in arg
  return "..." if too_big else arg


@decorator
def wrap_objective(f, *args, **kwds):
  """Decorator for creating Objective factories.

  Changes f from the closure: (args) => () => TF Tensor
  into an Obejective factory: (args) => Objective

  while perserving function name, arg info, docs... for interactive python.
  """
  objective_func = f(*args, **kwds)
  objective_name = f.__name__
  args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
  description = objective_name.title() + args_str
  return objectives.Objective(objective_func, objective_name, description)


@wrap_objective
def channel(layer, n_channel, batch=None):
  """Visualize a single channel"""
  if batch is None:
    return lambda T: tf.reduce_mean(T(layer)[..., n_channel])
  else:
    return lambda T: tf.reduce_mean(T(layer)[batch, ..., n_channel])


@wrap_objective
def abs_channel(layer, n_channel, batch=None):
  """Visualize a single channel"""
  if batch is None:
    return lambda T: tf.reduce_mean(tf.abs(T(layer)[..., n_channel]))
  else:
    return lambda T: tf.reduce_mean(tf.abs(T(layer)[batch, ..., n_channel]))


@wrap_objective
def L1_channel(layer, n_channel, batch=None, constant=None):
  """Visualize a single channel"""
  if batch is None:
    return lambda T: tf.reduce_mean(tf.abs(T(layer)[..., n_channel] - constant))
  else:
    return lambda T: tf.reduce_mean(tf.abs(T(layer)[batch, ..., n_channel] - constant))


@wrap_objective
def L1_add_ori_channel(layer, n_channel, batch=None, constant=None):
  """Visualize a single channel"""
  if batch is None:
    return lambda T: tf.reduce_mean(tf.abs(T(layer)[..., n_channel] + constant))
  else:
    return lambda T: tf.reduce_mean(tf.abs(T(layer)[batch, ..., n_channel] + constant))


@wrap_objective
def square_abs_channel(layer, n_channel, batch=None):
  """Visualize a single channel"""
  if batch is None:
    return lambda T: tf.reduce_mean(tf.abs(T(layer)[..., n_channel])) ** 2
  else:
    return lambda T: tf.reduce_mean(tf.abs(T(layer)[batch, ..., n_channel])) ** 2

@wrap_objective
def one_neuron_channel(layer, n_channel, attr, batch=None):
  """Visualize a single channel"""
  if batch is None:
    return lambda T: tf.reduce_mean(T(layer)[..., n_channel] * attr)
  else:
    return lambda T: tf.reduce_mean(T(layer)[batch, ..., n_channel] * attr)


@wrap_objective
def L1(layer="input", constant=0, batch=None, alpha=0.01):
  """L1 norm of layer. Generally used as penalty."""
  if batch is None:
    return lambda T: alpha * tf.reduce_sum(tf.abs(T(layer) - constant))
  else:
    return lambda T: alpha * tf.reduce_sum(tf.abs(T(layer)[batch] - constant))


@wrap_objective
def L2(layer="input", constant=0, epsilon=1e-6, batch=None, alpha=0.01):
  """L2 norm of layer. Generally used as penalty."""
  if batch is None:
    return lambda T: alpha * tf.sqrt(epsilon + tf.reduce_sum((T(layer) - constant) ** 2))
  else:
    return lambda T: alpha * tf.sqrt(epsilon + tf.reduce_sum((T(layer)[batch] - constant) ** 2))

@wrap_objective
def SquareL2(layer="input", constant=0, batch=None, alpha=0.01):
  """Square L2 norm of layer. Generally used as penalty."""
  if batch is None:
    return lambda T: alpha *  tf.reduce_sum((T(layer) - constant) ** 2)
  else:
    return lambda T: alpha * tf.reduce_sum((T(layer)[batch] - constant) ** 2)


def _dot_attr_actmaps(x, y):
  xy_dot = tf.reduce_sum(x * y, -1)
  return tf.reduce_mean(xy_dot)


@wrap_objective
def dot_attr_actmaps(layer, attr, batch=None):
  """Loss func to compute the dot of attribution and activation maps"""
  if batch is None:
    attr = attr[None, None, None]
    return lambda T: _dot_attr_actmaps(T(layer), attr)
  else:
    attr = attr[None, None]
    return lambda T: _dot_attr_actmaps(T(layer)[batch], attr)


@wrap_objective
def dot_separete_neuronattr_actmaps(layer, attr, n_channel, batch=None):
  """Loss func to compute the dot of attribution and activation maps"""
  if batch is None:
    return lambda T: _dot_attr_actmaps(T(layer)[..., n_channel], attr)
  else:
    return lambda T: _dot_attr_actmaps(T(layer)[batch, ..., n_channel], attr)


@wrap_objective
def dot_neuronattr_actmaps(layer, attr, batch=None):
  """Loss func to compute the dot of attribution and activation maps"""
  if batch is None:
    return lambda T: _dot_attr_actmaps(T(layer), attr)
  else:
    return lambda T: _dot_attr_actmaps(T(layer)[batch], attr)


class MatrixDecomposer(object):
  """For Matrix Decomposition to the innermost dimension of a tensor.

  This class wraps sklearn.decomposition classes to help them apply to arbitrary
  rank tensors. It saves lots of annoying reshaping.

  See the original sklearn.decomposition documentation:
  http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
  """

  def __init__(self, n_features=3, reduction_alg=None, **kwargs):
    """Constructor for MatrixDecomposer.

    Inputs:
      n_features: Numer of dimensions to reduce inner most dimension to.
      reduction_alg: A string or sklearn.decomposition class.
      kwargs: Additional kwargs to be passed on to the reducer.
    """
    if isinstance(reduction_alg, str):
      reduction_alg = sklearn.decomposition.__getattribute__(reduction_alg)
    self.n_features = n_features
    self._decomposer = reduction_alg(n_features, **kwargs)

  @classmethod
  def _apply_flat(cls, f, acts):
    """Utility for applying f to inner dimension of acts.

    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    orig_shape = acts.shape
    acts_flat = acts.reshape([-1, acts.shape[-1]])
    new_flat = f(acts_flat)
    if not isinstance(new_flat, np.ndarray):
      return new_flat
    shape = list(orig_shape[:-1]) + [-1]
    return new_flat.reshape(shape)

  @classmethod
  def prec_apply_sum(cls, f):
    """Utility for applying f to inner dimension of acts.

    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    new_flat = f()
    new_flat = np.sum(new_flat)
    return new_flat

  def get_precision(self):
    return MatrixDecomposer.prec_apply_sum(self._decomposer.get_precision)

  def get_score(self, AM, W):
    W = np.reshape(W, (-1, W.shape[-1]))
    prediction = np.dot(W, self._decomposer.components_)
    # prediction = self._decomposer.inverse_transform(W)
    prediction = np.reshape(prediction, (-1, prediction.shape[-1]))

    AM = np.reshape(AM, (-1, AM.shape[-1]))
    score = sklearn.metrics.explained_variance_score(AM, prediction)
    return score

  def fit(self, acts):
    return MatrixDecomposer._apply_flat(self._decomposer.fit, acts)

  def fit_transform(self, acts):
    return MatrixDecomposer._apply_flat(self._decomposer.fit_transform, acts)

  def transform(self, acts):
    return MatrixDecomposer._apply_flat(self._decomposer.transform, acts)

  # def transform(self, X):
  #   """
  #   E-step to compute transform X, or factors
  #   for factor analysis
  #   """
  #   orig_shape = X.shape
  #   X_flat = X.reshape([-1, X.shape[-1]])
  #   X_flat = check_array(X_flat)
  #   X_flat = X_flat - self._decomposer.mean_
  #   I = np.eye(len(self._decomposer.components_))
  #   temp = self._decomposer.components_ / self._decomposer.noise_variance_
  #   sigma = np.linalg.inv(I + np.dot(temp, self._decomposer.components_.T))
  #   X_transformed = np.dot(np.dot(X_flat, temp.T), sigma)
  #   shape = list(orig_shape[:-1]) + [-1]
  #   return X_transformed.reshape(shape)


class SklearnCluster(object):
  """Helper for clustering to the innermost dimension of a tensor.

  This class wraps sklearn.cluster classes to help them apply to arbitrary
  rank tensors. It saves lots of annoying reshaping.

  See the original sklearn.decomposition documentation:
  https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
  """
  def __init__(self, n_clusters=6, reduction_alg="KMeans", **kwargs):
    """Constructor for SklearnCluster.

    Inputs:
      n_features: Numer of dimensions to reduce inner most dimension to.
      reduction_alg: A string or sklearn.decomposition class. Defaults to
        "KMeans"
      kwargs: Additional kwargs to be passed on to the reducer.
    """
    if isinstance(reduction_alg, str):
      reduction_alg = sklearn.cluster.__getattribute__(reduction_alg)
    self.n_clusters = n_clusters
    self._decomposer = reduction_alg(n_clusters, **kwargs)

  @classmethod
  def _apply_flat(cls, f, acts):
    """Utility for applying f to inner dimension of acts.

    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    acts_flat = acts.reshape([-1, acts.shape[-1]])
    acts_flat = np.transpose(acts_flat, (1, 0))
    labels = f(acts_flat)
    return labels

  def fit_predict(self, acts):
    return SklearnCluster._apply_flat(self._decomposer.fit_predict, acts)

  def __dir__(self):
    dynamic_attrs = dir(self._decomposer)
    return self.__dict__.keys()


def save_baselineimgs(images, save_directory, attr_class, baseline_method
                      , no_slash_layer_name, save_np_only=True):
  if no_slash_layer_name == "input":
    if len(images.shape) > 3:
      for i_optimgs in range(len(images)):
        images_temp = images[i_optimgs]
        w = int(np.sqrt(images_temp.size / 3))
        img = images_temp.reshape(w, w, 3)
        baseline_method = findall('[A-Z]', baseline_method)
        baseline_method = ''.join(baseline_method)
        if not save_np_only:
          plt.imsave(save_directory + "/" + attr_class + '_' + baseline_method + '_' +
                     no_slash_layer_name + str(i_optimgs) + ".jpeg", img)
        np.save(save_directory + '/' + attr_class + '_' + baseline_method + '_' +
                no_slash_layer_name + str(i_optimgs) + '.npy', img)
    else:
      images_temp = images
      w = int(np.sqrt(images_temp.size / 3))
      img = images_temp.reshape(w, w, 3)
      baseline_method = findall('[A-Z]', baseline_method)
      baseline_method = ''.join(baseline_method)
      if not save_np_only:
        plt.imsave(save_directory + "/" + attr_class + '_' + baseline_method + '_' +
                   no_slash_layer_name + str(0) + ".jpeg", img)
      np.save(save_directory + '/' + attr_class + '_' + baseline_method + '_' +
              no_slash_layer_name + str(0) + '.npy', img)
  else:
    if len(images.shape) >3:
      for i_optimgs in range(len(images)):
        img = images[i_optimgs]
        baseline_method = findall('[A-Z]', baseline_method)
        baseline_method = ''.join(baseline_method)
        np.save(save_directory + '/' + attr_class + '_' + baseline_method + '_' +
                no_slash_layer_name + str(i_optimgs) + '.npy', img)
    else:
        img = images
        baseline_method = findall('[A-Z]', baseline_method)
        baseline_method = ''.join(baseline_method)
        np.save(save_directory + '/' + attr_class + '_' + baseline_method + '_' +
                no_slash_layer_name + str(0) + '.npy', img)


def save_imgs(images, save_directory, attr_class, decomposition_method
              , no_slash_layer_name, imgtype_name='opt'):
  for i_optimgs in range(len(images)):
    if len(images[i_optimgs]) > 1:
      images_temp = images[i_optimgs]
      w = int(np.sqrt(images_temp.size / 3))
      img = images_temp.reshape(w, w, 3)
      decomposition_method = findall('[A-Z]', decomposition_method)
      decomposition_method = ''.join(decomposition_method)
      plt.imsave(save_directory + "/" + attr_class + '_' + decomposition_method + '_' +
                 no_slash_layer_name + '_' + imgtype_name + str(i_optimgs) + ".jpeg", img)


def save_imgs_seperate_vis(images, save_directory, attr_class, decomposition_method
                           , no_slash_layer_name, channel_shap_one, vis_channel_index=None):
  for i_optimgs in range(len(images)):
    if len(images[i_optimgs]) > 1:
      images_temp = images[i_optimgs]
      w = int(np.sqrt(images_temp.size / 3))
      img = images_temp.reshape(w, w, 3)
      decomposition_method = findall('[A-Z]', decomposition_method)
      decomposition_method = ''.join(decomposition_method)
      plt.imsave(save_directory + '/' + channel_shap_one[i_optimgs] + attr_class + '_' + decomposition_method + '_' +
                 no_slash_layer_name + str(vis_channel_index[i_optimgs][0]) + '.jpeg', img)


def plot(data, save_directory, attr_class, decomposition_method,
         no_slash_layer_name, imgtype_name, index_saveimg, xi=None, cmap='RdBu_r', cmap2='seismic', alpha=0.8):
  plt.ioff()
  # plt.ion()
  fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

  axis = plt.Axes(fig, [0., 0., 1., 1.])
  axis.set_axis_off()
  fig.add_axes(axis)

  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1]+dx, dx)
  yy = np.arange(0.0, data.shape[0]+dy, dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  cmap_xi = plt.get_cmap(cmap2)
  cmap_xi.set_bad(alpha=0)
  overlay = xi
  if len(data.shape) == 3:
    data = np.mean(data, 2)
  # axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
  axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
  axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
  # decomposition_method = findall('[A-Z]', decomposition_method)
  decomposition_method = ''.join(decomposition_method)
  plt.savefig(save_directory + '/' + decomposition_method + '_' + attr_class + '_' +
              no_slash_layer_name + '_' + imgtype_name + str(index_saveimg) + '.jpeg')  # 'RdBu_r' 'hot'
  # plt.show()
  # plt.close(1)


def plot_deep_explain_sep(data, save_directory, attr_class, decomposition_method,
         no_slash_layer_name, imgtype_name, xi=None, cmap='RdBu_r', cmap2='seismic', alpha=0.3):
  plt.ioff()
  # plt.ion()
  fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

  axis = plt.Axes(fig, [0., 0., 1., 1.])
  axis.set_axis_off()
  fig.add_axes(axis)

  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1]+dx, dx)
  yy = np.arange(0.0, data.shape[0]+dy, dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  cmap_xi = plt.get_cmap(cmap2)
  cmap_xi.set_bad(alpha=0)
  overlay = xi
  if len(data.shape) == 3:
    data = np.mean(data, 2)
  # 此处数值需要单步微调
  abs_max = np.percentile(np.abs(data), 60)
  abs_min = abs_max
  # axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
  axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
  axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
  # plt.show()
  # decomposition_method = findall('[A-Z]', decomposition_method)
  # decomposition_method = ''.join(decomposition_method)
  plt.savefig(save_directory + '/' + attr_class + '_' + decomposition_method + '_' +
              no_slash_layer_name + '_' + imgtype_name + '.jpeg')  # 'RdBu_r' 'hot'
  # plt.show()
  # plt.close(1)


def plot_deep_explain(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1], dx)
  yy = np.arange(0.0, data.shape[0], dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  cmap_xi = plt.get_cmap('seismic')
  # cmap_xi = plt.get_cmap('Greys_r')
  cmap_xi.set_bad(alpha=0)
  overlay = None
  if xi is not None:
    # Compute edges (to overlay to heatmaps later)
    xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
    in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
    edges = feature.canny(in_image_upscaled).astype(float)
    edges[edges < 0.5] = np.nan
    edges[:5, :] = np.nan
    edges[-5:, :] = np.nan
    edges[:, :5] = np.nan
    edges[:, -5:] = np.nan
    overlay = edges

  abs_max = np.percentile(np.abs(data), percentile)
  abs_min = abs_max

  if len(data.shape) == 3:
    data = np.mean(data, 2)
  axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
  if overlay is not None:
    axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
  axis.axis('off')
  return axis


def plot_seperate(data, save_directory, attr_class, decomposition_method,
         no_slash_layer_name, imgtype_name, score_str, index_num, xi=None, cmap='RdBu_r', alpha=0.8):
  plt.ioff()
  # plt.ion()
  fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

  axis = plt.Axes(fig, [0., 0., 1., 1.])
  axis.set_axis_off()
  fig.add_axes(axis)

  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1]+dx, dx)
  yy = np.arange(0.0, data.shape[0]+dy, dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  cmap_xi = plt.get_cmap('seismic')
  cmap_xi.set_bad(alpha=0)
  overlay = xi
  if len(data.shape) == 3:
    data = np.mean(data, 2)
  # axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
  axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
  axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
  decomposition_method = findall('[A-Z]', decomposition_method)
  decomposition_method = ''.join(decomposition_method)
  plt.savefig(save_directory + '/' + str(score_str) + attr_class + '_' + decomposition_method + '_' +
              no_slash_layer_name + '_' + imgtype_name + index_num + '.jpeg')  # 'RdBu_r' 'hot'
  # plt.show()
  # plt.close(1)


def plot_mask_ori_img(data, save_directory, attr_class, decomposition_method,
                      no_slash_layer_name, imgtype_name, index_saveimg, xi=None, cmap='RdBu_r', alpha=0.8):
  # mask = data > 0
  mask = data <= np.quantile(data, 85 / 100)
  mask = np.expand_dims(mask, axis=-1)
  masked_img = mask * xi
  plt.imsave(save_directory + "/" + attr_class + '_' + decomposition_method + '_' +
             no_slash_layer_name + '_masked_' + imgtype_name + str(index_saveimg) +
             ".jpeg", masked_img)

  plt.ioff()
  fig = plt.figure(1, figsize=[2.24, 2.24], dpi=100, frameon=False)

  axis = plt.Axes(fig, [0., 0., 1., 1.])
  axis.set_axis_off()
  fig.add_axes(axis)

  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1]+dx, dx)
  yy = np.arange(0.0, data.shape[0]+dy, dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  cmap_xi = plt.get_cmap('seismic')
  cmap_xi.set_bad(alpha=0)
  overlay = xi

  if len(data.shape) == 3:
    data = np.mean(data, 2)
  # axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
  axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
  axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
  decomposition_method = findall('[A-Z]', decomposition_method)
  decomposition_method = ''.join(decomposition_method)
  # Does save the heat maps?
  # plt.savefig(save_directory + '/' + attr_class + '_' + decomposition_method + '_' +
  #             no_slash_layer_name + '_' + imgtype_name + str(index_saveimg) + '.jpeg')
  # 'RdBu_r' 'hot'
  # plt.close(1)


def resize_show(data, xi=None, cmap='RdBu_r', alpha=0.2):
  plt.ioff()
  fig = plt.figure(1, figsize=[2.24, 2.24], dpi=300)

  axis = plt.Axes(fig, [0., 0., 1., 1.])
  axis.set_axis_off()
  fig.add_axes(axis)

  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1]+dx, dx)
  yy = np.arange(0.0, data.shape[0]+dy, dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  cmap_xi = plt.get_cmap('seismic')
  cmap_xi.set_bad(alpha=0)
  overlay = xi
  if len(data.shape) == 3:
    data = np.mean(data, 2)
  # axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
  axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
  axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
  # plt.show()
  plt.show(block=False)
  plt.pause(3)
  plt.close()
  # print("Run show code")


def show_figure(data):
  plt.ioff()
  fig = plt.figure(1, figsize=[2.24, 2.24], dpi=300)

  axis = plt.Axes(fig, [0., 0., 1., 1.])
  axis.set_axis_off()
  fig.add_axes(axis)

  dx, dy = 0.05, 0.05
  xx = np.arange(0.0, data.shape[1]+dx, dx)
  yy = np.arange(0.0, data.shape[0]+dy, dy)
  xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
  extent = xmin, xmax, ymin, ymax
  if len(data.shape) == 3:
    data = np.mean(data, 2)
  axis.imshow(data, extent=extent, interpolation='none')
  plt.show()
  # plt.show(block=False)
  # plt.pause(3)
  plt.close()


def lowres_tensor(shape, underlying_shape, offset=None, sd=None, random_seed=0):
  """Produces a tensor paramaterized by a interpolated lower resolution tensor.

  This is like what is done in a laplacian pyramid, but a bit more general. It
  can be a powerful way to describe images.

  Args:
    shape: desired shape of resulting tensor
    underlying_shape: shape of the tensor being resized into final tensor
    offset: Describes how to offset the interpolated vector (like phase in a
      Fourier transform). If None, apply no offset. If a scalar, apply the same
      offset to each dimension; if a list use each entry for each dimension.
      If a int, offset by that much. If False, do not offset. If True, offset by
      half the ratio between shape and underlying shape (analagous to 90
      degrees).
    sd: Standard deviation of initial tensor variable.
    random_seed: set the random seed.

  Returns:
    A tensor paramaterized by a lower resolution tensorflow variable.
  """
  sd = sd or 0.01

  if random_seed > 0:
    np.random.seed(random_seed)
  init_val = sd*np.random.randn(*underlying_shape).astype("float32")
  underlying_t = tf.Variable(init_val)
  t = resize_bilinear_nd(underlying_t, shape)
  if offset is not None:
    # Deal with non-list offset
    if not isinstance(offset, list):
      offset = len(shape)*[offset]
    # Deal with the non-int offset entries
    for n in range(len(offset)):
      if offset[n] is True:
        offset[n] = shape[n]/underlying_shape[n]/2
      if offset[n] is False:
        offset[n] = 0
      offset[n] = int(offset[n])
    # Actually apply offset by padding and then croping off the excess.
    padding = [(pad, 0) for pad in offset]
    t = tf.pad(t, padding, "SYMMETRIC")
    begin = len(shape)*[0]
    t = tf.slice(t, begin, shape)
  return t


def make_batches(size, batch_size):
  """Returns a list of batch indices (tuples of indices).
  # Arguments
      size: Integer, total size of the data to slice into batches.
      batch_size: Integer, batch size.
  # Returns
      A list of tuples of array indices.
  """
  num_batches = (size + batch_size - 1) // batch_size  # round up
  return [(i * batch_size, min(size, (i + 1) * batch_size))
          for i in range(num_batches)]


def to_list(x, allow_tuple=False):
  if isinstance(x, list):
    return x
  if allow_tuple and isinstance(x, tuple):
    return list(x)
  return [x]


def unpack_singleton(x):
  """Gets the equivalent np-array if the iterable has only one value.
  Otherwise return the iterable.
  # Argument
    x: A list or tuple.
    # Returns
      The same iterable or the iterable converted to a np-array.
  """
  if len(x) == 1:
    return np.array(x)
  return x


def slice_arrays(arrays, start=None, stop=None):
  """Slices an array or list of arrays.
	"""
  if arrays is None:
    return [None]
  elif isinstance(arrays, list):
    return [None if x is None else x[start:stop] for x in arrays]
  else:
    return arrays[start:stop]


def placeholder_from_data(numpy_array):
  if numpy_array is None:
    return None
  return tf.placeholder('float', [None, ] + list(numpy_array.shape[1:]))


def checkerboard(h, w=None, channels=3, tiles=4, fg=.95, bg=.6):
  """Create a shape (w,h,1) array tiled with a checkerboard pattern."""
  w = w or h
  square_size = [ceil(float(d / tiles) / 2) for d in [h, w]]
  board = [[fg, bg] * tiles, [bg, fg] * tiles] * tiles
  scaled = np.kron(board, np.ones(square_size))[:w, :h]
  return np.dstack([scaled] * channels)


def composite_alpha_onto_backgrounds(rgba,
                   input_encoding='gamma',
                   output_encoding='gamma'):
  if input_encoding == 'gamma':
      rgba = rgba.copy()
      rgba[..., :3] = from_gamma_to_linear(rgba[..., :3])

  h, w = rgba.shape[:2]

  # Backgrounds
  black = np.zeros((h, w, 3), np.float32)
  white = np.ones((h, w, 3), np.float32)
  grid = checkerboard(h, w, 3, tiles=8)

  # Collapse transparency onto backgrounds
  rgb, a = rgba[..., :3], rgba[..., 3:]
  vis = []
  for background in [black, white, grid]:
    vis.append(background * (1.0 - a) + rgb * a)
  vis.append(white * a)  # show just the alpha channel separately

  # Reshape into 2x2 grid
  vis = np.float32(vis).reshape(2, 2, h, w, 3)
  vis = np.vstack(map(np.hstack, vis))
  if output_encoding == 'gamma':
      vis = from_linear_to_gamma(vis)
  return vis