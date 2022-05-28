import numpy as np
from scipy import stats


def normalize_heatmap(heatmap):
	return (heatmap - np.mean(heatmap)) / np.std(heatmap)


def biSearchCertainSumAttrs(attrs, threshold_value):
	# attrs is a vector
	# threshold_value between 0~1
	sum_attrs = np.where(attrs > 0, attrs, 0).sum(0)
	threshold_value_attr = sum_attrs * threshold_value
	sorted_attrs = -np.sort(-attrs)
	indices_attrs = np.argsort(-attrs)
	cumsum_attrs = np.cumsum(sorted_attrs)
	idx_sorted_ori_attr = np.searchsorted(cumsum_attrs, threshold_value_attr)
	nowanted_indices = indices_attrs[idx_sorted_ori_attr:]
	kept_channel_indices = indices_attrs[:idx_sorted_ori_attr]
	return nowanted_indices, kept_channel_indices


def filter_attr_map(attr_squ, filter_with_hm=False, total_thres=0.9,
					filter_model='max', flag_kept_channel_list=False):
	if filter_with_hm:
		heat_map_ori = np.mean(attr_squ, -1)
		heat_map_ori = heat_map_ori**4
		heat_map_ori = normalize_heatmap(heat_map_ori)
		heat_map_ori = heat_map_ori* (heat_map_ori>0)
		heat_map = np.expand_dims(heat_map_ori, -1)
		attr_squ = attr_squ * heat_map
	# attr_squ = np.maximum(0, attr_squ)
	attr_squ_abs = np.abs(attr_squ)

	if filter_model == 'cumsumMax':
		attr_squ_vec = np.max(attr_squ_abs, axis=(0, 1))
		nowanted_indices, kept_channel_indices = biSearchCertainSumAttrs(attr_squ_vec, total_thres)
		attr_squ[..., nowanted_indices] = 0
	elif filter_model == 'cumsumSum':
		attr_squ_vec = np.sum(attr_squ_abs, axis=(0, 1))
		nowanted_indices, kept_channel_indices = biSearchCertainSumAttrs(attr_squ_vec, total_thres)
		attr_squ[..., nowanted_indices] = 0
	else:
		attr_squ_vec = np.max(attr_squ_abs, axis=(0, 1))
		channel_quantile = np.quantile(attr_squ_vec, total_thres)
		attr_squ = attr_squ * (attr_squ_vec > channel_quantile)

	test_attr = np.sum(attr_squ, axis=(0, 1))
	print('In vis attrs, non_zero channel is: {}/{}'.format(len(np.nonzero(test_attr)[0]), test_attr.shape[-1]))
	# big_attr_idx = np.argsort(-test_attr)
	# permute_attr = np.transpose(attr_squ, (2, 0, 1))
	# big_attr = permute_attr[big_attr_idx[0:10]]
	if flag_kept_channel_list:
		return attr_squ, kept_channel_indices
	else:
		return attr_squ


def filter_attr_map_channel_first(attr_squ, heatmap_flag=False, spatial_thres=0.95, channel_thres=0.9):
	if heatmap_flag:
		attr_squ_temp = np.mean(attr_squ, axis=(0, 1))
		channel_quantile = np.quantile(attr_squ_temp, channel_thres)
		attr_hm = attr_squ * (attr_squ_temp > channel_quantile)

		along_channel_attr_quantile = np.quantile(attr_hm, spatial_thres, axis=(0, 1))
		attr_hm = attr_hm * (attr_hm > along_channel_attr_quantile)
		test_attr = np.max(attr_hm, axis=(0, 1))
		print('In heat maps, non_zero channel is: {}/{}'.format(len(np.nonzero(test_attr)[0]), test_attr.shape[-1]))
		return attr_hm
		# attr_a = normalize_heatmap(attr_temp)
	else:
		# if attr_filter is not None:
		# 	sum_attr_filter = np.max(attr_filter, axis=(0, 1))
		# 	filter_attr_idx = np.argsort(-sum_attr_filter)
		# 	attr_squ[..., filter_attr_idx[0:int(len(sum_attr_filter)/4)]] = 0

		# attr_squ_temp = np.max(attr_squ, axis=(0, 1))
		attr_squ_temp = np.mean(attr_squ, axis=(0, 1))
		channel_quantile = np.quantile(attr_squ_temp, channel_thres)
		attr_squ = attr_squ * (attr_squ_temp > channel_quantile)

		along_channel_attr_quantile = np.quantile(attr_squ, spatial_thres, axis=(0, 1))
		attr_vis = attr_squ * (attr_squ > along_channel_attr_quantile)

		test_attr = np.sum(attr_vis, axis=(0, 1))
		big_attr_idx = np.argsort(-test_attr)
		permute_attr = np.transpose(attr_vis, (2, 0, 1))
		big_attr = permute_attr[big_attr_idx[0:10]]
		print('In vis attrs, non_zero channel is: {}/{}'.format(len(np.nonzero(test_attr)[0]), test_attr.shape[-1]))
		return attr_vis

