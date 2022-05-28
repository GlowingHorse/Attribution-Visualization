# To generate Aumann-Shapley attribution baseline
# Reduce the iteration number, if you feel it is too slow

import os
from operator import itemgetter

import numpy as np
import tensorflow as tf
import cv2

from utils.loading import load
from utils.reading import read
from utils.utils_network import InceptionV1
import utils.render_baseline as render
from utils.utils import save_baselineimgs
import utils.utils as utils
from utils.utils import plot

import timeit

from skimage.transform import resize


def main():
	tf.get_logger().setLevel('WARNING')
	# generate baseline and save as ".npy" file

	# import model
	model = InceptionV1()
	model.load_graphdef()

	labels_str = read(model.labels_path)
	labels_str = labels_str.decode("utf-8")
	labels = [line for line in labels_str.split("\n")]

	# ----------------------------------------

	img_paths = ["./data/cat-flower.jpg"]
	corres_attr_classes = [['vase']]

	# for ext in ('*.JPEG', '*.png', '*.jpg'):
	# 	img_paths.extend(glob(join("./data/images/vase_val_images", ext)))
	# corres_attr_classes = corres_attr_classes + [['vase']] * 100

	# ----------------------------------------
	# layers = ['input', 'mixed3a', 'maxpool4',
	# 			'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'maxpool10',
	# 			'mixed5a', 'mixed5b']

	layers = ['mixed4d']

	# ----------------------------------------

	logit_layer = "softmax2_pre_activation"
	last_logit_layer = "softmax1"
	opt_param = 16
	save_baseline_flag = True
	debug_baseline_flag = True

	img_info_name = 'multiobj_img_info'

	# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	def pad_images_to_same_size(image):
		"""
		:param images: sequence of images
		:return: list of images padded so that all images have same width and height (max width and height are used)
		"""
		width_max = 224
		height_max = 224

		curr_img_width = image.shape[1]
		curr_img_height = image.shape[0]

		fx = width_max/curr_img_width
		fy = height_max/curr_img_height

		image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
		return image

	for i in range(len(img_paths)):
		img = load(img_paths[i])
		img = pad_images_to_same_size(img)
		img_name = os.path.splitext(os.path.basename(img_paths[i]))[-2]
		attr_classes = corres_attr_classes[i]

		baseline_method_names = ["OPTimizeFull"]
		gen_baseline(attr_classes, layers, baseline_method_names, img, model, labels, img_name, logit_layer,
					 opt_param=opt_param, save_baseline_flag=save_baseline_flag, img_info_name=img_info_name,
					 debug_baseline_flag=debug_baseline_flag, last_logit_layer=last_logit_layer)


def gen_baseline(attr_classes, layers,  baseline_method_names, img, model, labels, img_name, logit_layer,
				 opt_param=20, save_baseline_flag=True, img_info_name='img_info',
				 debug_baseline_flag=False, last_logit_layer="softmax1"):
	# indices = [labels.index(attr_class) for attr_class in ['tabby', 'golden retriever', 'Labrador retriever']]
	for attr_class in attr_classes:
		with tf.Graph().as_default(), tf.Session() as sess_class:
			want_index = labels.index(attr_class)

			t_img_input = tf.placeholder_with_default(img, [None, None, 3])
			T = render.import_model(model, t_img_input, t_img_input)
			all_logit_score = T(logit_layer)[0]
			all_logit_score = all_logit_score.eval()

			sorted_logit = all_logit_score.argsort().astype(int)

			sorted_logit = sorted_logit[::-1]
			pred_index = sorted_logit[0:20]
			print("img name : {}".format(img_name))
			print(itemgetter(*pred_index)(labels))
			print(itemgetter(*pred_index)(all_logit_score))
			# print(logit[pred_index])

			other_class_index_names = []
			other_class_indices_L = []
			other_class_indices_S = set()

			other_class_num_param = 1

			for i_inner_attr_class in range(len(attr_classes)):
				attr_class_inner_temp = attr_classes[i_inner_attr_class]
				if attr_class_inner_temp != attr_class:
					other_class_index_names.append(attr_class_inner_temp)
					other_class_indices_L.append(labels.index(attr_class_inner_temp))
					other_class_indices_S.add(labels.index(attr_class_inner_temp))

			want_index_first_appear = np.where(sorted_logit == want_index)[0][0]
			if want_index_first_appear < other_class_num_param:
				other_class_indices_S.update(sorted_logit[0:want_index_first_appear])
				other_class_indices_S.update(sorted_logit[want_index_first_appear+1:want_index_first_appear+other_class_num_param])
			else:
				other_class_indices_S.update(sorted_logit[want_index_first_appear-other_class_num_param:want_index_first_appear])
				other_class_indices_S.update(sorted_logit[want_index_first_appear+1:want_index_first_appear+other_class_num_param])

			other_class_indices = list(other_class_indices_S)

			logit = T(logit_layer)[..., want_index]
			ori_logit_for_all = logit.eval()[0]
			print('target class is {}, score is {:.2f}'.format(attr_class, ori_logit_for_all))

		index = labels.index(attr_class)
		baseline_save_dir = './experiment/' + img_info_name + '/' + img_name + '/ShapBaseline/' + attr_class
		if not os.path.exists(baseline_save_dir):
			os.makedirs(baseline_save_dir)

		for i_layer in range(len(layers)):
			layer = layers[i_layer]
			with tf.Graph().as_default(), tf.Session() as sess_layer:
				t_input = tf.placeholder_with_default(img, [None, None, 3])
				T = render.import_model(model, t_input, t_input)
				acts = T(layer).eval()
				acts_ori = T(layer).eval()
				alpha_param = np.sum(acts_ori)

			for baseline_method_name in baseline_method_names:
				layer_output_name = "{}:0".format(layer)
				model.input_name = layer_output_name
				model.image_value_range = (0, 1)

				start_time = timeit.default_timer()
				# obj = utils.abs_channel(logit_layer, want_index, batch=0)

				# To drop the particular score to zero and keep baseline approaches to activation maps
				obj = utils.L1(constant=acts_ori[0], batch=0, alpha=opt_param / alpha_param) \
					  + utils.abs_channel(logit_layer, want_index, batch=0)

				# obj = utils.L1(constant=acts_ori[0], batch=0, alpha=opt_param / alpha_param) \
				# 	  + utils.L1_channel(logit_layer, want_index, batch=0, constant=np.min(all_logit_score))

				def keep_same():
					def inner(t_image):
						return t_image
					return inner

				transforms = [keep_same()]

				def interpolate_f():
					acts_tf = tf.Variable(acts, trainable=False)
					# lasso_mask_channel = tf.Variable(np.random.randn(acts.shape[0], 1, 1, acts.shape[-1]).astype("float32")
					# 						 , trainable=True)
					lasso_mask_spatial = tf.Variable(np.random.randn(acts.shape[0], acts.shape[1], acts.shape[2], 1).astype("float32")
											         , trainable=True)
					# a continuous mask
					lasso_mask_spatial = tf.nn.sigmoid(lasso_mask_spatial)
					# lasso_mask_channel = tf.nn.sigmoid(lasso_mask_channel)

					# a binary mask
					# lasso_mask = tf.nn.relu(tf.math.sign(lasso_mask))

					acts_tf = tf.reshape(acts_tf, acts.shape)
					return lasso_mask_spatial*acts_tf

				group_icons_spatial = render.render_vis(model, objective_f=obj, param_f=interpolate_f,
												transforms=transforms,
												optimizer=tf.train.AdamOptimizer(0.009),
												thresholds=(256, 512, 1024, 2048), verbose=True)[-1]
				group_icons = group_icons_spatial

				if baseline_method_name == "OPTimizeFull":
					del group_icons
					obj = utils.L1(constant=acts_ori[0], batch=0, alpha=opt_param / alpha_param) \
						  + utils.abs_channel(logit_layer, want_index, batch=0)

					# obj = utils.L1(constant=acts_ori[0], batch=0, alpha=opt_param / alpha_param) \
					# 	  + utils.L1_channel(logit_layer, want_index, batch=0, constant=np.min(all_logit_score))

					def keep_same():
						def inner(t_image):
							return t_image

						return inner

					transforms = [keep_same()]

					def interpolate_f():
						acts_tf = tf.Variable(acts, trainable=False)
						lasso_mask_channel = tf.Variable(np.random.randn(acts.shape[0], 1, 1, acts.shape[-1]).astype("float32")
												 , trainable=True)
						# lasso_mask_spatial = tf.Variable(
						# 	np.random.randn(acts.shape[0], acts.shape[1], acts.shape[2], 1).astype("float32")
						# 	, trainable=True)
						# a continuous mask
						# lasso_mask_spatial = tf.nn.sigmoid(lasso_mask_spatial)
						lasso_mask_channel = tf.nn.sigmoid(lasso_mask_channel)

						# a binary mask
						# lasso_mask = tf.nn.relu(tf.math.sign(lasso_mask))

						acts_tf = tf.reshape(acts_tf, acts.shape)
						return lasso_mask_channel * acts_tf

					group_icons_channel = render.render_vis(model, objective_f=obj, param_f=interpolate_f,
													transforms=transforms,
													optimizer=tf.train.AdamOptimizer(0.01),
													thresholds=(256, 512, 4096), verbose=True)[-1]
					group_icons = np.minimum(group_icons_spatial, group_icons_channel)
					# group_icons = np.maximum(group_icons_spatial, group_icons_channel)
					# group_icons = np.nan_to_num(group_icons, True, 0.0)
					# group_icons = group_icons_spatial * group_icons_channel

				stop_time = timeit.default_timer()
				print('OPT run time: {}; layer is: {}'.format(stop_time - start_time, layer))
				with tf.Graph().as_default(), tf.Session() as sess:
					t_input = tf.placeholder_with_default(acts, [None, None, None, None])
					T = render.import_model(model, t_input, t_input)
					logit_tensor = T(logit_layer)[0]
					last_layer_logit = T(last_logit_layer)[..., want_index]

					baseline_logit_all = logit_tensor.eval({T('input'): group_icons})
					# last_layer_logit = last_layer_logit.eval({T('input'): group_icons})

					baseline_logit = baseline_logit_all[want_index]

					print("img name : {}".format(img_name))
					print("Baseline method is: {}; layer is: {}".format(baseline_method_name, layer))
					print("Class is: {}; score: {:.2f}".format(attr_class, baseline_logit))

					if len(other_class_indices) == 0:
						pass
					else:
						print(itemgetter(*other_class_indices)(labels))
						print(itemgetter(*other_class_indices)(baseline_logit_all))

					# sorted_logit = baseline_logit_all.argsort()
					# sorted_logit = sorted_logit[::-1]
					# pred_index = sorted_logit[0:5]

					# print("Top classes name are:")
					# print(itemgetter(*pred_index)(labels))
					# print(itemgetter(*pred_index)(baseline_logit_all))

					print("OPT sum of original acts is: {:.2f} in layer: {}".
						  format(np.sum(acts_ori), layer))
					print("OPT sum of optimized acts is: {:.2f} in layer: {}".
						  format(np.sum(group_icons), layer))
					print("Sum of subtract of two acts is: {}".
						  format(np.sum(np.abs(group_icons - acts_ori))))
					print('')

				model.input_name = 'input:0'
				model.image_value_range = (-117, 255-117)

				if debug_baseline_flag:
					baseline_hm_save_dir = './experiment/' + img_info_name + '/' + img_name + '/ShapBaselineHM'
					if not os.path.exists(baseline_hm_save_dir):
						os.makedirs(baseline_hm_save_dir)
					debug_show_baseline_hm(acts, group_icons, img, baseline_hm_save_dir, attr_class, baseline_method_name,
										   layer)
				if save_baseline_flag:
					save_baselineimgs(group_icons, baseline_save_dir, attr_class, baseline_method_name, layer,
									  save_np_only=True)
				del group_icons
			print("layer {} is finished \n\n".format(layer))
		# score_img(labels, model, img_name, attr_class)
		# print(img_name)
		print("debug finish\n")


def debug_show_baseline_hm(acts, group_icons, img, save_directory,
						   attr_class, decomposition_method, no_slash_layer_name,
						   flag_save_sub_baseline=False):
	def inner_plot_summed_heatmap(heat_map_inner, prefix):
		heat_map_inner = resize(heat_map_inner, (224, 224), order=1,
						  mode='constant', anti_aliasing=False)
		heat_map_inner = heat_map_inner / max(heat_map_inner.max(), 0.00001) * 255
		plot(heat_map_inner, save_directory, prefix + attr_class, decomposition_method,
			 no_slash_layer_name, imgtype_name='',
			 index_saveimg='', xi=img, cmap='RdBu_r', cmap2='seismic', alpha=0.2)
	single_acts = acts[0]

	single_baseline = group_icons[0]
	heat_map = np.sum(single_baseline, axis=-1)
	inner_plot_summed_heatmap(heat_map, 'AllDesc-')

	baseline_squ_vec = np.max(single_baseline, axis=(0, 1))
	# baseline_squ_vec = np.sum(single_baseline, axis=(0, 1))

	sorted_baseline_vec = -np.sort(-baseline_squ_vec)
	indices_baseline_vec = np.argsort(-baseline_squ_vec)
	summed_desc_heatmap = np.sum(single_baseline[..., indices_baseline_vec[:int(indices_baseline_vec.shape[0]/4)]], -1)
	# inner_plot_summed_heatmap(summed_desc_heatmap, 'AllDesc-')

	indices_asc_baseline_vec = np.argsort(baseline_squ_vec)
	summed_asc_heatmap = np.sum(single_acts[..., indices_asc_baseline_vec[:int(50)]], -1)
	# inner_plot_summed_heatmap(summed_asc_heatmap, 'AllAsc-')

	summed_acts = np.sum(single_acts, -1)

	if flag_save_sub_baseline:
		for i_debug_hm in range(10):
			heat_map = single_baseline[..., indices_baseline_vec[i_debug_hm]]
			heat_map = resize(heat_map, (224, 224), order=1,
							  mode='constant', anti_aliasing=False)
			heat_map = heat_map / heat_map.max() * 255
			plot(heat_map, save_directory, attr_class, decomposition_method,
				 no_slash_layer_name, imgtype_name=str(i_debug_hm)+'-',
				 index_saveimg=str(indices_baseline_vec[i_debug_hm]), xi=img, cmap='RdBu_r', cmap2='seismic', alpha=0.2)
			# resize_show(heat_map, xi=img)
			# print('')


if __name__ == "__main__":
	main()
