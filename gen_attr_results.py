# Generate attribution results
#
from utils.loading import load
from utils.reading import read

import os
import numpy as np
import cv2

from utils.utils_network import InceptionV1
from utils.attr_computing import compute_attr, compute_deepliftreveal
from utils.utils_activation_maps import create_root_dir, print_result_from_logit
from utils.utils_save import gen_attr_heat_maps


def main():
	model = InceptionV1()
	model.load_graphdef()

	labels_str = read(model.labels_path)
	labels_str = labels_str.decode("utf-8")
	labels = [line for line in labels_str.split("\n")]

	img_paths = ["./data/cat-flower.jpg"]
	corres_attr_classes = [['vase']]

	img_info_name = 'multiobj_img_info'

	logit_layer = "softmax2_pre_activation"
	# logit_layer = "softmax1"

	# layers = ['conv2d0', 'conv2d1', 'conv2d2', 'mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c',
	# 			'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b']
	# layers = ['input', 'mixed3a', 'maxpool4',
	# 			'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'maxpool10',
	# 			'mixed5a', 'mixed5b']

	layers = ['mixed4d']
	# whether load the pre-computed feature attribution
	flag_read_attr = False

	flag1s = ["AShapleyOptf"]
	flag1_ASs = ["AShapleyOptf"]
	# iteration times for computing Shapley values
	iter_num = 200

	ori_model_input_name = model.input_name
	for i in range(len(img_paths)):
		img_name = img_paths[i]
		attr_classes = corres_attr_classes[i]
		gen_attr_results(img_name, layers, model,
						 ori_model_input_name=ori_model_input_name,
						 attr_classes=attr_classes, logit_layer=logit_layer, flag1s=flag1s,
						 flag_read_attr=flag_read_attr, iter_num=iter_num,
						 labels=labels, flag1_ASs=flag1_ASs, img_info_name=img_info_name)


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


def gen_attr_results(img_name, layers, model,
					 ori_model_input_name, attr_classes, logit_layer,
					 flag1s, flag_read_attr=False, iter_num=200,
					 labels=None, flag1_ASs=None, img_info_name='img_info'):
	img = load(img_name)
	img = pad_images_to_same_size(img)
	img_name_split = os.path.splitext(os.path.basename(img_name))[-2]

	# for adv samples
	# img_name_split = os.path.splitext(os.path.basename(img_name))[-2][:-5]

	for flag1 in flag1s:
		for attr_class in attr_classes:
			for layer in layers:
				print("Img name is: {}\nLayer is: {}\nMethod name is: {}\nClass name is: {}"
					  .format(img_name, layer, flag1, attr_class))
				root_directory = create_root_dir(img_name, attr_class, "neuronAttr", img_info_name=img_info_name)

				if flag1 == "DummyZERO" or flag1 == "Saliency" or flag1 == "GradxInput" or \
					 flag1 == "SmoothGrad" or flag1 == "EpsilonLRP" or flag1 == "DeepLIFTRescale" or flag1 == "Occlusion"\
					 or flag1 == "DeepSHAP" or flag1 == "ShapleySampling" or flag1 in flag1_ASs:
					attributions_temp, logit_list\
						= compute_attr(img, model, attr_class, layer, logit_layer=logit_layer,
										flag1=flag1, flag_read_attr=flag_read_attr,
										ori_model_input_name=ori_model_input_name,
										iter_num=iter_num, labels=labels,
										img_name_split=img_name_split, img_info_name=img_info_name)
					attributions = {flag1:attributions_temp}
					print_result_from_logit(logit_list, labels)
				elif flag1 == "DeepLIFTReveal":
					attributions_temp, logit_list, attributions_pos, attributions_neg\
						= compute_deepliftreveal(img, model, attr_class, layer,
												 flag1=flag1, flag_read_attr=flag_read_attr,
												 iter_num=iter_num, labels=labels,
												 img_name_split=img_name_split, img_info_name=img_info_name)
					# attributions = {"DeepLIFTReveal":attributions_temp, "DeepLIFTRevealPos":attributions_pos,
					# 								"DeepLIFTRevealNeg":attributions_neg}
					attributions = {"DeepLIFTReveal":attributions_temp}
					print_result_from_logit(logit_list, labels)
				elif flag1 == "ShapleySamplingRead":
					attributions_temp = np.load(root_directory+'/'+attr_class + "_ShapleySampling_HMs.npy")
					attributions = {"ShapleySampling": attributions_temp}
				else:
					continue
				save_directory = root_directory
				if not os.path.exists(save_directory):
					os.makedirs(save_directory)
				gen_attr_heat_maps(img, attributions, save_directory, attr_class,
								   layer_name=layer, save_np_flag=not flag_read_attr,
								   save_elegant_flag=True)
		print("method {} is finished\n".format(flag1))


if __name__ == "__main__":
	# execute only if run as a script
	main()
