# For the experiment of group attributions
# Using factor analysis to decompose attributions

import os
import numpy as np
from os.path import join
from glob import glob
from utils.utils_network import InceptionV1

from utils.utils_activation_maps import create_root_dir, print_result_from_logit
from utils.utils_activation_maps import debug_show_AM_plus_img, decompose_AM_get_group_num
from utils.utils_attrmap import filter_attr_map, filter_attr_map_channel_first, normalize_heatmap
import json


def main():
	model = InceptionV1()
	model.load_graphdef()

	img_paths = []
	corres_attr_classes = []

	# extract valid images from ImageNet before running this code
	for ext in ('*.JPEG', '*.png', '*.jpg'):
		img_paths.extend(glob(join("./data/images/vase_val_images", ext)))
	corres_attr_classes = corres_attr_classes + [['vase']]*len(img_paths)

	# ----------------------------------------

	img_info_name = 'multiobj_img_info'

	# layers = ['conv2d0', 'conv2d1', 'conv2d2',
	#           'mixed3a', 'maxpool4',
	# 		    'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d',
	# 		    'maxpool10', 'mixed5a', 'mixed5b']
	# layer_alpha = [0.9999, 0.9999, 0.9999,
	#                0.9999, 0.9999,
	# 			     0.65, 0.6, 0.6, 0.6,
	# 			     0.65, 0.05, 0.2]
	# layer_alpha = [alpha for alpha in layer_alpha]
	# scale_param = [[80, 120], [80, 120],
	#                [80, 120], [80, 120], [80, 120],
	# 		         [80, 125], [80, 125], [80, 120], [90, 120],
	# 			     [90, 130], [90, 130], [100, 101]]

	# layers = ['mixed4a', 'mixed4b', 'mixed4d', 'maxpool10']
	# layer_alpha = [0.5, 0.5, 0.5, 0.5]
	# layer_alpha = [alpha for alpha in layer_alpha]
	# scale_param = [[110, 140], [110, 140], [110, 140], [120, 140]]

	layers = ['maxpool4', 'mixed4d']
	layer_alpha = [0.3, 0.3]
	layer_alpha = [alpha for alpha in layer_alpha]

	flag1s = ["AShapleyOptf"]
	for flag1 in flag1s:
		neuron_vis(img_paths, corres_attr_classes, layers,
				   flag1=flag1, layer_alpha=layer_alpha, img_info_name=img_info_name)


def neuron_vis(img_paths, corres_attr_classes, layers,
			   flag1=None, layer_alpha=None, img_info_name='img_info'):

	thres_explained_var = 0.3
	decomposition_method = 'FactorAnalysis'

	for i_layer in range(len(layers)):

		attributions = []
		layer = layers[i_layer]
		no_slash_layer_name = ''.join(layer.split('/'))
		if no_slash_layer_name == 'maxpool10':
			no_slash_layer_name = 'mixed4e'
		elif no_slash_layer_name == 'maxpool4':
			no_slash_layer_name = 'mixed3b'
		else:
			pass

		for i in range(len(img_paths)):
			img_name = img_paths[i]
			attr_classes = corres_attr_classes[i]
			attr_class = attr_classes[0]

			root_directory = create_root_dir(img_name, attr_class, "neuronAttr", img_info_name=img_info_name)
			attr_ori = np.load(root_directory + '/' + attr_class + '_' + flag1 + '_' +
						   layer + '.npy')

			attr_squ = np.squeeze(attr_ori)

			attr_filtered, kept_channel_indices = \
				filter_attr_map(attr_squ, filter_with_hm=False, total_thres=layer_alpha[i_layer],
							    filter_model='cumsumMax', flag_kept_channel_list=True)
			attr_filtered = np.maximum(0, attr_filtered)

			attributions.append(attr_filtered)

		attributions = np.stack(attributions, axis=0)
		attributions = attributions.reshape((attributions.shape[0]*attributions.shape[1]*attributions.shape[2],
		                                     attributions.shape[3]))
		attr_corr_max_channel_indices = np.argsort(np.max(attributions, 0))[::-1]

		spatial_factors, channel_factors, n_groups = \
			decompose_AM_get_group_num(decomposition_method, attributions, thres_explained_var)

		# factor analysis
		channel_indices_in_group = np.where(np.max(channel_factors, 0) >= 0.02)[0]

		channel_factors_max_index = channel_factors.argmax(axis=0)

		maxAttr_maxChanFactor_intersect = np.intersect1d(attr_corr_max_channel_indices[:30],
		                                                 channel_indices_in_group)

		short_index = []
		for channel_factors_i in range(n_groups):
			channel_indices_in_one_group_temp = np.squeeze(np.argwhere(channel_factors_max_index == channel_factors_i), axis=1)
			channel_indices_in_one_group = np.intersect1d(channel_indices_in_one_group_temp, channel_indices_in_group)
			short_index.append(channel_indices_in_one_group)
		short_index = [ele for ele in short_index if ele != []]
		save_directory = './experiment/googlenet/' + attr_class
		if not os.path.exists(save_directory):
			os.makedirs(save_directory)

		np.save(save_directory + '/' + no_slash_layer_name + '_groupinfo.npy', short_index, allow_pickle=True)
		aaa = np.load(save_directory + '/' + no_slash_layer_name + '_groupinfo.npy', allow_pickle=True)
		print('Group number is {}'.format(len(short_index)))
		print('Layer {} and class {} is finished'.format(layer, attr_class))
		print()


if __name__ == "__main__":
	# execute only if run as a script
	main()
