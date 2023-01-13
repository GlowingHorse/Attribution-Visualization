"""
The attr_computing.py
provides two methods for computing channel Shapley
and transform them to feature attributions.
"""
import numpy as np
import tensorflow as tf

import utils.render_baseline as render
from utils.attr_explain import DeepExplain
import utils.utils_save as utils_save
import os


def compute_attr(img, model, attr_class, layer_name, logit_layer="softmax2_pre_activation",
                flag1=None, flag_read_attr=True, ori_model_input_name='input:0',
                iter_num=100, labels=None, img_name_split=None, img_info_name='img_info',
                experiment_dir_name='./experiment/'):
  xs = np.reshape(img, (1,) + img.shape)
  if layer_name != 'input':
    with tf.Graph().as_default(), tf.Session() as sess:
      model.input_name = ori_model_input_name
      model.image_value_range = (-117, 255 - 117)
      t_input = tf.placeholder_with_default(img, [None, None, 3])
      T = render.import_model(model, t_input, t_input)
      acts = T(layer_name).eval()
      logit = T(logit_layer)[0]
      logit4grad = T(logit_layer)[0, labels.index(attr_class)]
      logit_list = sess.run([logit], {T(layer_name): acts})[0]
      ys = sess.run([logit4grad], {T(layer_name): acts})[0]
      # print("sum of acts_ori is: {}".format(np.sum(acts)))
      # acts_sq = np.squeeze(acts)
    xs = acts.copy()
    acts_shape = (None, xs.shape[1], xs.shape[2], xs.shape[3])
    layer_output_name = "{}:0".format(layer_name)
    model.input_name = layer_output_name
    model.image_value_range = (0, 1)

  if logit_layer.startswith('resnet') and layer_name != 'input':
    layer_name = utils_save.filter_resnet_layer_name(layer_name)

  with tf.Graph().as_default(), tf.Session() as sess:
    with DeepExplain(session=sess, graph=sess.graph, flag=flag1) as de:
      baseline_dir = experiment_dir_name + img_info_name +'/' + img_name_split + '/ShapBaseline/' + attr_class + '/' + attr_class + '_'
      if layer_name == 'input':
        model.input_name = ori_model_input_name
        model.image_value_range = (-117, 255 - 117)
        t_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
      else:
        t_input = tf.placeholder(tf.float32, shape=acts_shape)

      # t_input = tf.placeholder_with_default(img, [None, None, 3])
      T = render.import_model(model, t_input, t_input)
      logit = T(logit_layer)[0]
      logit4grad = T(logit_layer)[..., labels.index(attr_class)]
      if layer_name == 'input':
        feed_dict = {t_input: xs}
        logit_list = sess.run([logit], feed_dict)[0]
        ys = sess.run([logit4grad], feed_dict)[0]

      if flag_read_attr:
        attributions = np.load(experiment_dir_name + img_info_name + '/' + img_name_split + '/' + attr_class + '_' + 'neuronAttr/' +
                                attr_class + '_' +flag1 +'_' + layer_name + '.npy')
      else:
        ys = np.expand_dims(ys, axis=0)
        if flag1 == "Occlusion":
          attributions = de.explain(method=flag1, T=logit4grad, X=t_input,
                                    xs=xs, window_shape=(3, 3, 3))
        elif flag1 == "ShapleySampling":
          attributions = de.explain(method=flag1, T=logit4grad, X=t_input,
                                    xs=xs, samples=2, sampling_dims=None, is_pixel=True)
        elif flag1 == "IntGradBlur":
          flag1 = "AShapley"
          baseline_type = baseline_dir + 'BLUR_' + layer_name + '0.npy'
          attributions = de.explain(method=flag1, T=logit4grad, X=t_input,
                                    xs=xs, steps=iter_num, baseline_type=baseline_type, originalX=xs)
        elif flag1 == "IntGradNoise":
          flag1 = "AShapley"
          baseline_type = baseline_dir + 'NOISE_' + layer_name + '0.npy'
          attributions = de.explain(method=flag1, T=logit4grad, X=t_input,
                                    xs=xs, steps=iter_num, baseline_type=baseline_type, originalX=xs)
        elif flag1.startswith('AShapley'):
          baseline_type = baseline_dir + flag1[8:].upper() + '_' + layer_name + '0.npy'
          flag1 = "AShapley"
          attributions = de.explain(method=flag1, T=logit4grad, X=t_input,
                                    xs=xs, ys=ys, steps=iter_num, baseline_type=baseline_type, originalX=xs)
        elif flag1 == "IntGrad":
          flag1 = "AShapley"
          # baseline_type = baseline_dir + 'ZERO_' + layer_name + '0.npy'
          attributions = de.explain(method=flag1, T=logit4grad, X=t_input,
                                    xs=xs, originalX=xs)
        elif flag1 in ["DeepLIFTRescale", "DeepSHAP"]:
            ref_dir = experiment_dir_name + img_info_name + '/' + img_name_split + '/GradChangeRef'
            if not os.path.exists(ref_dir):
              os.makedirs(ref_dir)
            ref_dir = ref_dir + '/ref' + flag1 + '.npz'
            attributions = de.explain(method=flag1, T=logit4grad, X=t_input, xs=xs, ref_dir=ref_dir)
        else:
          attributions = de.explain(method=flag1, T=logit4grad, X=t_input, xs=xs)

  return attributions, logit_list


def compute_deepliftreveal(img, model, attr_class, layer_name, logit_layer="softmax2_pre_activation",
                      flag1="AShapleyOpt", flag_read_attr=True,
                      ori_model_input_name='input:0',
                      iter_num=100, labels=None, img_name_split=None,
                      img_info_name='img_info'):
  xs = np.reshape(img, (1,) + img.shape)
  if layer_name != 'input':
    with tf.Graph().as_default(), tf.Session() as sess:
      model.input_name = ori_model_input_name
      model.image_value_range = (-117, 255 - 117)
      t_input = tf.placeholder_with_default(img, [None, None, 3])
      T = render.import_model(model, t_input, t_input)
      acts = T(layer_name).eval()
    xs = acts.copy()
    acts_shape = (None, xs.shape[1], xs.shape[2], xs.shape[3])
    layer_output_name = "{}:0".format(layer_name)
    model.input_name = layer_output_name
    model.image_value_range = (0, 1)

  with tf.Graph().as_default(), tf.Session() as sess:
    with DeepExplain(session=sess, graph=sess.graph, flag=flag1+'Pos') as de:
      if layer_name == 'input':
        model.input_name = ori_model_input_name
        model.image_value_range = (-117, 255 - 117)
        t_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
      else:
        t_input = tf.placeholder(tf.float32, shape=acts_shape)

      T = render.import_model(model, t_input, t_input)
      logit4grad = T(logit_layer)[0, labels.index(attr_class)]
      ref_dir = './experiment/' + img_info_name + '/' + img_name_split + '/GradChangeRef'
      if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
      ref_dir = ref_dir + '/ref' + flag1 + '.npz'
      attributions_pos = de.explain(method=flag1, T=logit4grad, X=t_input, xs=xs, ref_dir=ref_dir)
  with tf.Graph().as_default(), tf.Session() as sess:
    with DeepExplain(session=sess, graph=sess.graph, flag=flag1+'Neg') as de:
      if layer_name == 'input':
        model.input_name = ori_model_input_name
        model.image_value_range = (-117, 255 - 117)
        t_input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
      else:
        t_input = tf.placeholder(tf.float32, shape=acts_shape)

      T = render.import_model(model, t_input, t_input)
      logit4grad = T(logit_layer)[0, labels.index(attr_class)]
      logit = T(logit_layer)[0]

      attributions_neg = de.explain(method=flag1, T=logit4grad, X=t_input, xs=xs, ref_dir=ref_dir)
      # attributions_neg = de.explain(flag1, logit4grad, t_input, xs)
      feed_dict = {t_input: xs}
      # for k, v in zip(t_input, xs):
      logit_list = sess.run([logit], feed_dict)[0]
    attributions = attributions_pos + attributions_neg
  return attributions, logit_list, attributions_pos, attributions_neg


def raw_class_group_attr(img, model, layer, label, labels, group_vecs):
  """
  The method of Gradient * Activation maps
  """

  # Set up a graph for doing attribution...
  with tf.Graph().as_default(), tf.Session():
    t_input = tf.placeholder_with_default(img, [None, None, 3])
    T = render.import_model(model, t_input, t_input)

    # Compute activations
    acts = T(layer).eval()

    if label is None:
      return np.zeros(acts.shape[1:-1])

    # Compute gradient
    score = T("softmax2_pre_activation")[0, labels.index(label)]
    t_grad = tf.gradients([score], [T(layer)])[0]
    grad = t_grad.eval({T(layer): acts})

    # Linear approximation of effect of spatial position
    return [np.sum(group_vec * grad) for group_vec in group_vecs]


def score_f(logit, name, labels):
  if name is None:
    return 0
  elif name == "logsumexp":
    base = tf.reduce_max(logit)
    return base + tf.log(tf.reduce_sum(tf.exp(logit - base)))
  elif name in labels:
    return logit[labels.index(name)]
  else:
    raise RuntimeError("Unsupported")
