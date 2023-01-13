# Attribution visualization
# Mask perturbation is located in "render.render_vis"
# Attribution-Visualization

import os
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import tensorflow as tf

from lucid.optvis import transform

from skimage.transform import resize

from utils.loading import load
from utils.saving import save
from utils.reading import read
from utils.utils_network import InceptionV1
import utils.utils as utils
import utils.utils_random as utils_random
import utils.render_baseline as render
import utils.util_transform as transform_other

from utils.utils_activation_maps import create_root_dir

from utils.utils_save import gen_spatial_heat_maps
from utils.utils_attrmap import filter_attr_map
from utils.utils import resize_show

from re import findall


def main():
	gpu_available = tf.test.is_gpu_available()
	model = InceptionV1()
	model.load_graphdef()
	labels_str = read(model.labels_path)
	labels_str = labels_str.decode("utf-8")
	labels = [line for line in labels_str.split("\n")]

	img_paths = ["./data/cat-flower.jpg"]
	corres_attr_classes = [['vase']]

	global_random_seed = 5
	image_size = 224  # 224
	debug_flag = False
	# whether generate optimized images
	flag_gen_opt_img = True
	flag_gen_heatmap = True

	flag_save_visual_one_dir = True

	flag_from_gamma_to_linear = False

	img_info_name = 'multiobj_img_info'

	layers = ['mixed4d']
	layer_alpha = [0.95, 0.7]
	layer_alpha = [alpha for alpha in layer_alpha]
	scale_param = [[90, 120], [100, 130], [80, 120]]

	heatmap_alpha = [65]*12
	step_length = 2048  # 4096

	for i in range(len(img_paths)):
		img_name = img_paths[i]
		attr_classes = corres_attr_classes[i]
		flag1s = ["AShapleyOptf"]

		flag_opt_vis_list = ["ConstBack", "BlurImage", "Normal", "Noise", "NoisePyramid"]
		# flag_opt_vis_list = ["Normal", "NoisePyramid"]
		init_learning_rates = [0.02, 0.02]*10

		# init_learning_rates = [0.005] * 6

		for flag1 in flag1s:
			for i_flag_opt_vis in range(len(flag_opt_vis_list)):
				init_learning_rate = init_learning_rates[i_flag_opt_vis]
				flag_opt_vis = flag_opt_vis_list[i_flag_opt_vis]
				pos_flag = 1
				# ---------------------------------------------------------------------------------------------------
				neuron_vis(img_name, layers, model, attr_classes=attr_classes,
						   flag1=flag1, labels=labels, layer_alpha=layer_alpha, heatmap_alpha=heatmap_alpha,
						   vis_random_seed=global_random_seed,
						   image_size=image_size,
						   debug_flag=debug_flag, flag_gen_opt_img=flag_gen_opt_img,
						   flag_gen_heatmap=flag_gen_heatmap,
						   flag_save_visual_one_dir=flag_save_visual_one_dir,
						   img_info_name=img_info_name, flag_from_gamma_to_linear=flag_from_gamma_to_linear,
						   flag_opt_vis=flag_opt_vis, scale_param=scale_param,
						   init_learning_rate=init_learning_rate, step_length=step_length)


def neuron_vis(img_name, layers, model, attr_classes,
			   flag1=None, labels=None, layer_alpha=None, heatmap_alpha=None,
			   vis_random_seed=0, image_size=0,
			   debug_flag=1, flag_gen_opt_img=True,
			   flag_gen_heatmap=True,
			   flag_save_visual_one_dir=False, img_info_name='img_info',
			   flag_from_gamma_to_linear=None, flag_opt_vis=None,
			   scale_param=None, init_learning_rate=0.01,
			   step_length=512):
	img = load(img_name)
	img_name_split = os.path.splitext(os.path.basename(img_name))[-2]
	for i_layer in range(len(layers)):
		layer = layers[i_layer]
		no_slash_layer_name = ''.join(layer.split('/'))
		if no_slash_layer_name == 'maxpool10':
			no_slash_layer_name = 'mixed4e'
		elif no_slash_layer_name == 'maxpool4':
			no_slash_layer_name = 'mixed3b'
		else:
			pass

		for class_idx in range(len(attr_classes)):
			attr_class = attr_classes[class_idx]

			root_directory = create_root_dir(img_name, attr_class, "neuronAttr", img_info_name=img_info_name)
			attr_ori = np.load(root_directory + '/' + attr_class + '_' + flag1 + '_' +
						   layer + '.npy')
			print("Begin -> Method {}; Layer {} and class {}".format(flag_opt_vis, no_slash_layer_name, attr_class))

			attr_squ = np.squeeze(attr_ori)

			# heat_maps = [filter_attr_map(attr_squ, heatmap_flag=False, filter_with_hm=False, total_thres=alpha[1])]
			heat_maps = [attr_squ]
			heat_maps = np.asarray(heat_maps, dtype=np.float32)

			attr_temp = filter_attr_map(attr_squ, filter_with_hm=False, total_thres=layer_alpha[i_layer],
									   filter_model='cumsumMax')
			attr_temp = np.maximum(0, attr_temp)
			attrs = [attr_temp]
			# heat_maps = attrs
			# heat_maps = attrs
			if debug_flag == 1:
				for heat_map in np.transpose(attr_temp, axes=(2, 0, 1)):
					heat_map = resize(heat_map, (model.image_shape[0], model.image_shape[1]), order=1,
											 mode='constant', anti_aliasing=False)
					heat_map = heat_map / heat_map.max() * 255
					resize_show(heat_map, xi=img)

			if flag_save_visual_one_dir:
				save_directory = './experiment/' + img_info_name + '/' + img_name_split + '/visual_result'
			else:
				save_directory = './experiment/' + img_info_name + '/' + img_name_split + '/' + flag1
			if not os.path.exists(save_directory):
				os.makedirs(save_directory)

			n_groups = len(attrs)
			attrs = np.asarray(attrs, dtype=np.float32)

			attr_ori = np.expand_dims(attr_squ, 0)

			heat_maps = np.mean(attrs, -1)
			# heat_maps = np.mean(heat_maps, -1)

			# spatial_factors1 = spatial_factors.copy()
			# def norm_2D_array(array):
			#   return (array-np.mean(array, axis=(0, 1)))/np.linalg.norm(array, axis=(0, 1))
			# spatial_factors1[0] = norm_2D_array(spatial_factors[0]) + norm_2D_array(spatial_factors[1]) + norm_2D_array(spatial_factors[2])
			# gen_spatial_heat_maps(85, n_groups, spatial_factors1, save_directory,
			#                       attr_class, decomposition_method, no_slash_layer_name, img, AM, model)

			if flag_gen_heatmap:
				gen_spatial_heat_maps(heatmap_alpha[i_layer], n_groups, heat_maps, save_directory,
									  attr_class, flag1, no_slash_layer_name, img,
									  imgtype_name1='Heatmap', model=model)
			if flag_gen_opt_img:
				# "BlurImage", "BlurMask", "BlurBoth", "NoiseTilde"
				attrs = np.maximum(0, attrs)

				# attrs = attr_ori
				obj = sum(utils.dot_neuronattr_actmaps(layer, attrs[i], batch=i)
						  for i in range(n_groups))

				# use heat maps to get mask area control
				print("Sum of attrs", np.sum(attrs))

				# For feature visualization, the library "lucid" will be useful because
				# it has implements many loss functions of different literatures, image processing operators,
				# and collected several pretrained tensorflow network.
				opt_name = 'opt'

				optimizer = None
				input_img_channel = 4

				transforms = [
					transform.pad(12),
					transform.jitter(8),
					# transform_other.random_shear(4),
					transform.random_scale(
						[n / 100. for n in range(scale_param[i_layer][0], scale_param[i_layer][1])]),

					# deeper layer -> e.g., 4d:70, 110; 5a:85, 130 alpha[0, 0.4]
					# shallow layer -> 70-100

					transform.random_rotate(
						# list(range(-10, 10)) + list(range(-4, 4))
						list(range(-12, 12)) + list(range(-6, 6))
						+ 5 * list(range(-2, 2))),

					transform.jitter(2),
					transform.crop_or_pad_to(image_size, image_size)
				]

				# if flag_opt_vis in ["ConstBack", ]:
				# 	transforms = [
				# 		transform.pad(14),
				# 		transform.jitter(6),
				# 		transform.jitter(2),
				# 		transform.crop_or_pad_to(image_size, image_size)
				# 	]
				# elif flag_opt_vis in ["BlurImage", "Normal"]:
				# 	transforms = [
				# 		transform.pad(14),
				# 		transform.jitter(6),
				#
				# 		transform.random_rotate(
				# 			list(range(-12, 12)) + list(range(-6, 6))
				# 			+ 5 * list(range(-2, 2))),
				#
				# 		transform.jitter(2),
				# 		transform.crop_or_pad_to(image_size, image_size)
				# 	]
				# elif flag_opt_vis in ["Noise"]:
				# 	transforms = [
				# 		transform.pad(14),
				# 		transform.jitter(6),
				# 		# transform_other.random_shear(4),
				# 		transform.jitter(2),
				# 		transform.crop_or_pad_to(image_size, image_size)
				# 	]
				# else:
				# 	if layer in ['conv2d0', 'conv2d1', 'conv2d2', 'mixed3a']:
				# 		transforms = [
				# 			transform.pad(12),
				# 			transform.jitter(8),
				# 			transform.jitter(4),
				# 			transform.jitter(2),
				# 			transform.crop_or_pad_to(image_size, image_size)
				# 		]
				# 	else:
				# 		transforms = [
				# 			transform.pad(12),
				# 			transform.jitter(8),
				# 			# transform_other.random_shear(4),
				# 			transform.random_scale(
				# 				[n / 100. for n in range(scale_param[i_layer][0], scale_param[i_layer][1])]),
				#
				# 			# deeper layer -> e.g., 4d:70, 110; 5a:85, 130 alpha[0, 0.4]
				# 			# shallow layer -> 70-100
				#
				# 			transform.random_rotate(
				# 				# list(range(-10, 10)) + list(range(-4, 4))
				# 				list(range(-12, 12)) + list(range(-6, 6))
				# 				+ 5 * list(range(-2, 2))),
				#
				# 			# transform_other.random_shear(4),
				# 			transform.jitter(2),
				# 			transform.crop_or_pad_to(image_size, image_size)
				# 		]

				# image parameterization with shared params for aligned optimizing images
				def interpolate_f():
					unique = utils_random.fft_image(
						(n_groups, image_size, image_size, input_img_channel),
						random_seed=vis_random_seed)
					t_image = utils_random.to_valid_rgb(unique[..., :3],
														decorrelate=True,
														flag_from_gamma_to_linear=flag_from_gamma_to_linear)
					a = tf.nn.sigmoid(unique[..., 3:])
					return tf.concat([t_image, a], -1)

				vis_imgs = render.render_vis(model, objective_f=obj, param_f=interpolate_f,
											 optimizer=optimizer,
											 transforms=transforms, thresholds=(128, int(step_length)),
											 verbose=True,
											 flag_opt_vis=flag_opt_vis, init_learning_rate=init_learning_rate,
											 attrs=attr_ori, masksize=1)[-1]

				flag1_save = findall('[A-Z]', flag1)
				flag1_save = ''.join(flag1_save)

				for i_optimgs in range(len(vis_imgs)):
					vis_img = vis_imgs[i_optimgs]
					out_name_alpha = save_directory + "/" + attr_class + '_' + flag_opt_vis + '_' + flag1_save + '_' + \
					                    no_slash_layer_name + '_alpha_' + opt_name + str(i_optimgs)
					# save(vis_img, out_name_alpha + '_' + flag_opt_vis + '.png')
					out_name_composed = save_directory + "/" + attr_class + '_' + flag_opt_vis + '_' + flag1_save + '_' + \
					                    no_slash_layer_name + '_comp_' + opt_name + str(i_optimgs)
					# composed_img = 0.55 * (1.0 - vis_img[..., 3:]) + vis_img[..., :3] * vis_img[..., 3:]
					composed_img = 0.55 * (1.0 - vis_img[..., 3:]) + vis_img[..., :3] * vis_img[..., 3:]
					save(composed_img, out_name_composed + '.jpeg')

					# for i_noise_t_rgb_pyramid_tensor in range(len(noise_t_rgb_pyramid_tensor)):
					# 	noise_pyramid = noise_t_rgb_pyramid_tensor[i_noise_t_rgb_pyramid_tensor]
					# 	out_name_noise_pyramid = save_directory + "/" + attr_class + '_comp_' + 'Noisepyramid_'\
					# 							 + str(i_noise_t_rgb_pyramid_tensor)
					# 	save(noise_pyramid, out_name_noise_pyramid + '.jpeg')

					# out_name = save_directory + "/" + attr_class + '_' + flag1 + '_' + \
					# 		   no_slash_layer_name + '_' + opt_name + str(i_optimgs)
					# save(vis_img[..., :3], out_name + '.jpeg')
					# alpha_img = vis_img[..., 3]
					# alpha_img = np.dstack([alpha_img] * 3)
					# joined = np.hstack([vis_img[..., :3], alpha_img])
					# save(joined, out_name + '1.jpg', quality=100)
					# save_imgs(group_icons, save_directory, attr_class, flag1,
					# 			no_slash_layer_name, imgtype_name=opt_name)
			print("Finished -> Method {}; Layer {} and class {}\n".
				  format(flag_opt_vis, no_slash_layer_name, attr_class))


if __name__ == "__main__":
	# execute only if run as a script
	main()
