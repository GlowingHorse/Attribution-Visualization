import numpy as np
from utils.utils import save_imgs, plot, plot_deep_explain_sep
from skimage.transform import rescale, resize, downscale_local_mean
from re import findall
import tabulate
import cv2
import matplotlib.pyplot as plt
import os


def filter_resnet_layer_name(layer_name):
  return ''.join([layer_name.split('/')[1], '_', layer_name.split('/')[2]])


def gen_attr_heat_maps(img, attributions, save_directory, attr_class,
                       layer_name, save_np_flag=False, save_elegant_flag=False):
  no_slash_layer_name = layer_name
  if no_slash_layer_name != 'input':
    for j, a in enumerate(attributions):
      if len(attributions[a].shape) == 3:
        spatial_factor_ori = attributions[a]
      elif len(attributions[a].shape) == 4:
        spatial_factor_ori = attributions[a][0]
      if save_np_flag:
        np.save(save_directory + '/' + attr_class + '_' + a + '_' +
                    no_slash_layer_name + '.npy', spatial_factor_ori)
  else:
    img = (img - np.min(img))
    img /= np.max(img)
    for j, a in enumerate(attributions):
      alpha = 0.2
      cmap2 = 'seismic'
      cmap2_r = cmap2 + '_r'
      if len(attributions[a].shape) == 3:
        spatial_factor_ori = attributions[a]
      elif len(attributions[a].shape) == 4:
        spatial_factor_ori = attributions[a][0]
      if save_np_flag:
        np.save(save_directory + '/' + attr_class + '_' + a + '_' +
                    no_slash_layer_name + '.npy', spatial_factor_ori)
      # plt.figure()
      # plt.imshow(spatial_factor_3D)

      # plt.imshow(spatial_factor_gray, cmap='gray')
      # plot_deep_explain_sep(spatial_factor_gray, save_directory, attr_class, attr_method_names,
      #                       no_slash_layer_name, imgtype_name='gray', xi=img, cmap2=cmap2, alpha=alpha)
      spatial_factor = np.max(spatial_factor_ori, 2)
      spatial_factor_mean = np.mean(spatial_factor_ori, 2)
      spatial_factor_pos = spatial_factor * (spatial_factor >= 0)
      spatial_factor_neg = spatial_factor * (spatial_factor < 0)
      # spatial_factor_pos2 = spatial_factor_mean * (spatial_factor_mean >= 0)
      spatial_factor_abs = np.abs(spatial_factor_mean)
      if save_elegant_flag:
        # plot_deep_explain_sep(spatial_factor_abs, save_directory, attr_class, a,
        #                       no_slash_layer_name, imgtype_name='abs', xi=img, cmap2=cmap2, alpha=alpha)
        plot_deep_explain_sep(spatial_factor_pos, save_directory, attr_class, a,
                              no_slash_layer_name, imgtype_name='pos', xi=img, cmap2=cmap2, alpha=alpha)
        # plot_deep_explain_sep(spatial_factor_pos2, save_directory, attr_class, a,
        #                       no_slash_layer_name, imgtype_name='pos2', xi=img, cmap2=cmap2, alpha=alpha)
      else:
        spatial_factor_gray = cv2.cvtColor(spatial_factor_ori, cv2.COLOR_RGB2GRAY)
        spatial_factor_gray = cv2.normalize(spatial_factor_gray, None, alpha=-1,
                                            beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        plot_deep_explain_sep(spatial_factor_gray, save_directory, attr_class, a,
                              no_slash_layer_name, imgtype_name='gray', xi=img, cmap2=cmap2, alpha=alpha)
        plot_deep_explain_sep(spatial_factor_abs, save_directory, attr_class, a,
                              no_slash_layer_name, imgtype_name='abs', xi=img, cmap2=cmap2, alpha=alpha)

        plot_deep_explain_sep(spatial_factor, save_directory, attr_class, a,
                              no_slash_layer_name, imgtype_name='all', xi=img, cmap2=cmap2, alpha=alpha)
        plot_deep_explain_sep(spatial_factor_pos, save_directory, attr_class, a,
                              no_slash_layer_name, imgtype_name='pos', xi=img, cmap2=cmap2_r, alpha=alpha)
        plot_deep_explain_sep(spatial_factor_neg, save_directory, attr_class, a,
                              no_slash_layer_name, imgtype_name='neg', xi=img, cmap2=cmap2_r, alpha=alpha)
      # for cmap2 in ['seismic', 'bwr', 'coolwarm']:
      #   plot_deep_explain_sep(spatial_factor, save_directory, attr_class, attr_method_names,
      #                         no_slash_layer_name, imgtype_name=cmap2, xi=img, cmap2=cmap2, alpha=0.3)


def gen_spatial_heat_maps(q_spatial, decomposed_channel_num, spatial_factors, save_directory,
                          attr_class, decomposition_method, no_slash_layer_name, img,
                          imgtype_name1 = 'SpatialHM', model=None):
  if q_spatial is not None:
    for i_decomposed_channel_num in range(decomposed_channel_num):
      spatial_factors[i_decomposed_channel_num, ...] = spatial_factors[i_decomposed_channel_num, ...] * (
        spatial_factors[i_decomposed_channel_num, ...] > np.quantile(spatial_factors[i_decomposed_channel_num, ...],
                                                                     q_spatial / 100))

  index_saveimg = 0
  for i_factor in range(spatial_factors.shape[0]):
    factor_resized = resize(spatial_factors[i_factor], (model.image_shape[0], model.image_shape[1]), order=1,
                            mode='constant', anti_aliasing=False)

    plot(factor_resized, save_directory, attr_class, decomposition_method, no_slash_layer_name,
         imgtype_name1, index_saveimg, xi=img, cmap2='seismic', alpha=0.3)
    index_saveimg = index_saveimg + 1

  # AM2 = AM * (
  #     AM > np.quantile(AM, 0.998))
  # for i_gradcam in range(AM2.shape[3]):
  #   grad_cam = AM2[..., i_gradcam]
  #   if np.quantile(grad_cam, 0.98) > 0:
  #     grad_cam = grad_cam.reshape([grad_cam.shape[1], grad_cam.shape[2]])
  #     grad_cam_resized = resize(grad_cam, (model.image_shape[0], model.image_shape[1]), order=1, mode='constant', anti_aliasing=False)
  #     # factor_resized = factor_resized * (factor_resized > np.quantile(factor_resized, 0.90))
  #     imgtype_name2 = 'GradcamHM'
  #     plot(grad_cam_resized, save_directory, attr_class, decomposition_method, no_slash_layer_name,
  #          imgtype_name2, i_gradcam, xi=img, alpha=0.3)
  #     index_saveimg = index_saveimg + 1

  # print('layer {} \'s heat maps have been saved'.format(no_slash_layer_name))


def gen_info_txt(channel_shap, decomposed_channel_num, save_directory, decomposition_method,
                 attr_class, every_group_attr_sorted):
  channel_shap_max_index = channel_shap.argmax(axis=0)
  # channel_factors_index_temp = np.squeeze(np.argwhere(np.sum(channel_shap, axis=0) == 0))
  channel_shap_max_index[np.squeeze(np.argwhere(np.sum(channel_shap, axis=0) == 0))] = 74
  channel_shap_unique, channel_shap_counts = \
    np.unique(channel_shap_max_index, return_counts=True)
  correspond_channel_shap_index = []
  for channel_factors_i in range(decomposed_channel_num):
    correspond_channel_shap_index_temp = \
      np.argwhere(channel_shap_max_index == channel_factors_i)
    if correspond_channel_shap_index_temp.ndim > 1:
      correspond_channel_shap_index_temp = \
        np.squeeze(correspond_channel_shap_index_temp, axis=1)
    correspond_channel_shap_index.append(list(correspond_channel_shap_index_temp))

  channel_factors_maxindex = dict(zip(channel_shap_unique, channel_shap_counts))
  with open(save_directory + "/" + decomposition_method + attr_class + '_SpatialAttrs.txt', 'w') as f:
    f.write("%s\n" % "Original Attrs from map 0->last")
    for item in range(decomposed_channel_num):
      f.write("%s " % str(item))
    f.write("\n\n")
    f.write("%s\n" % "Soft Index for map 0->last")
    for item in every_group_attr_sorted:
      f.write("%.2f " % item)
    f.write("\n\n")
    f.write("%s\n" % "Every component corr to channels' number")
    for k, v in channel_factors_maxindex.items():
      f.write(str(k) + '>>>' + str(v) + '  ')
    f.write("\n\n")
    for k, v in enumerate(correspond_channel_shap_index):
      if len(v) > 5:
        v = v[:5]
      f.write(str(k) + '>>>' + str(v))
      f.write("\n")


def gen_Shap_info_txt(channel_shap_ori, channel_shap_adv, channel_shap_sub, long_index_ori,
                      ns_sorted, decomposed_channel_num_ori, decomposed_channel_num_adv, save_directory,
                      decomposition_method, attr_class, layer):
  channel_shap_ori_sorted = channel_shap_ori[ns_sorted]
  channel_shap_adv_sorted = channel_shap_adv[ns_sorted]
  channel_shap_sub_sorted = channel_shap_sub[ns_sorted]
  long_index_ori_sorted = [long_index_ori[i] for i in ns_sorted]

  decomposition_method = findall('[A-Z]', decomposition_method)
  decomposition_method = ''.join(decomposition_method)

  with open(save_directory + '/' + decomposition_method + '_' + attr_class + '_ShapleyCompare.txt', 'w') as f:
    f.write("There are {} groups in {}. Original group nums is: {}\n".format(decomposed_channel_num_ori, layer, decomposed_channel_num_adv))
    for i_decomposed_channel_num_ori in range(decomposed_channel_num_ori):
      f.write("------------------------------ Group {} -----------------------------------------\n".format(i_decomposed_channel_num_ori))
      channel_shap_ori_short = channel_shap_ori_sorted[i_decomposed_channel_num_ori, long_index_ori_sorted[i_decomposed_channel_num_ori]]
      index_temp = np.argsort(-channel_shap_ori_short)
      channel_shap_ori_short = channel_shap_ori_short[index_temp]
      channel_shap_adv_short = channel_shap_adv_sorted[i_decomposed_channel_num_ori, long_index_ori_sorted[i_decomposed_channel_num_ori]][index_temp]
      channel_shap_sub_short = channel_shap_sub_sorted[i_decomposed_channel_num_ori, long_index_ori_sorted[i_decomposed_channel_num_ori]][index_temp]
      long_index_ori_short = long_index_ori_sorted[i_decomposed_channel_num_ori][index_temp]

      headers = []
      second_row = []
      third_row = []
      forth_row = []
      length_shap_group = min(len(channel_shap_ori_short), 20)
      for i_length_shap_group in range(length_shap_group):
        headers.append('{:0>3d}'.format(long_index_ori_short[i_length_shap_group]))
        second_row.append('{:.2f}'.format(channel_shap_ori_short[i_length_shap_group]))
        third_row.append('{:.2f}'.format(channel_shap_adv_short[i_length_shap_group]))
        forth_row.append('{:.2f}'.format(channel_shap_sub_short[i_length_shap_group]))

      headers =    ['       index:      total is  {}   '.format(len(long_index_ori_short))] + headers
      second_row = ['Shap Original:     total is {:.2f}'.format(np.sum(channel_shap_ori_short))] + second_row
      third_row =  ['Shap Adversarial:  total is {:.2f}'.format(np.sum(channel_shap_adv_short))] + third_row
      forth_row =  ['Shap Delta Change: total is {:.2f}'.format(np.sum(channel_shap_sub_short))] + forth_row

      table1 = [second_row, third_row, forth_row]
      f.write("Sort using original Shap\n")
      f.write(tabulate.tabulate(table1, headers=headers, tablefmt="simple"))
      f.write("\n\n")
      f.write("Sort using the amount of changed Shap\n")
      shap_sub_index = np.argsort(-channel_shap_sub_short)
      channel_shap_sub_short_sorted = channel_shap_sub_short[shap_sub_index]
      long_index_ori_short_sorted = long_index_ori_short[shap_sub_index]

      headers2 = []
      fifth_row = []
      for i_length_shap_group in range(length_shap_group):
        headers2.append('{:0>3d}'.format(long_index_ori_short_sorted[i_length_shap_group]))
        fifth_row.append('{:.2f}'.format(channel_shap_sub_short_sorted[i_length_shap_group]))
      headers2 =  ['       index:     total is  {}   '.format(len(long_index_ori_short))] + headers2
      fifth_row = ['Shap Adversarial: total is {:.2f}'.format(np.sum(channel_shap_sub_short_sorted))] + fifth_row

      table2 = [fifth_row]
      f.write(tabulate.tabulate(table2, headers=headers2, tablefmt="simple"))
      f.write("\n\n\n")


def gen_IGSG_info_txt(channel_shap_ori, channel_shap_adv, channel_shap_sub, long_index_ori,
                      ns_sorted, decomposed_channel_num_ori, decomposed_channel_num_adv, save_directory,
                      decomposition_method, attr_class, layer):
  channel_shap_ori_sorted = channel_shap_ori[ns_sorted]
  channel_shap_adv_sorted = channel_shap_adv[ns_sorted]
  channel_shap_sub_sorted = channel_shap_sub[ns_sorted]
  long_index_ori_sorted = [long_index_ori[i] for i in ns_sorted]

  decomposition_method = findall('[A-Z]', decomposition_method)
  decomposition_method = ''.join(decomposition_method)

  with open(save_directory + '/' + decomposition_method + '_' + attr_class + '_ShapleyCompare.txt', 'w') as f:
    f.write("There are {} groups in {}. Original group nums is: {}\n".format(decomposed_channel_num_ori, layer, decomposed_channel_num_adv))
    for i_decomposed_channel_num_ori in range(decomposed_channel_num_ori):
      f.write("------------------------------ Group {} -----------------------------------------\n".format(i_decomposed_channel_num_ori))
      channel_shap_ori_short = channel_shap_ori_sorted[i_decomposed_channel_num_ori, long_index_ori_sorted[i_decomposed_channel_num_ori]]
      index_temp = np.argsort(-channel_shap_ori_short)
      channel_shap_ori_short = channel_shap_ori_short[index_temp]
      channel_shap_adv_short = channel_shap_adv_sorted[i_decomposed_channel_num_ori, long_index_ori_sorted[i_decomposed_channel_num_ori]][index_temp]
      channel_shap_sub_short = channel_shap_sub_sorted[i_decomposed_channel_num_ori, long_index_ori_sorted[i_decomposed_channel_num_ori]][index_temp]
      long_index_ori_short = long_index_ori_sorted[i_decomposed_channel_num_ori][index_temp]

      headers = []
      second_row = []
      third_row = []
      forth_row = []
      length_shap_group = min(len(channel_shap_ori_short), 20)
      for i_length_shap_group in range(length_shap_group):
        headers.append('{:0>3d}'.format(long_index_ori_short[i_length_shap_group]))
        second_row.append('{:.2f}'.format(channel_shap_ori_short[i_length_shap_group]))
        third_row.append('{:.2f}'.format(channel_shap_adv_short[i_length_shap_group]))
        forth_row.append('{:.2f}'.format(channel_shap_sub_short[i_length_shap_group]))

      headers =    ['        index:  total is  {}   '.format(len(long_index_ori_short))] + headers
      second_row = ['Shap Original:  total is {:.2f}'.format(np.sum(channel_shap_ori_short))] + second_row
      third_row =  ['IGSG Original:  total is {:.2f}'.format(np.sum(channel_shap_adv_short))] + third_row
      forth_row =  ['IGSG Original:  total is {:.2f}'.format(np.sum(channel_shap_sub_short))] + forth_row

      table1 = [second_row, third_row, forth_row]
      f.write("Sort using original Shap\n")
      f.write(tabulate.tabulate(table1, headers=headers, tablefmt="simple"))
      f.write("\n\n")
      f.write("Sort using original IGSG\n")
      shap_sub_index = np.argsort(-channel_shap_sub_short)
      channel_shap_sub_short_sorted = channel_shap_sub_short[shap_sub_index]
      long_index_ori_short_sorted = long_index_ori_short[shap_sub_index]

      headers2 = []
      fifth_row = []
      for i_length_shap_group in range(length_shap_group):
        headers2.append('{:0>3d}'.format(long_index_ori_short_sorted[i_length_shap_group]))
        fifth_row.append('{:.2f}'.format(channel_shap_sub_short_sorted[i_length_shap_group]))
      headers2 =  ['       index:   total is  {}   '.format(len(long_index_ori_short))] + headers2
      fifth_row = ['IGSG Original:  total is {:.2f}'.format(np.sum(channel_shap_sub_short_sorted))] + fifth_row

      table2 = [fifth_row]
      f.write(tabulate.tabulate(table2, headers=headers2, tablefmt="simple"))
      f.write("\n\n\n")


def gen_Shap_info_txt_SameGroup(channel_shap_ori, channel_shap_adv, channel_shap_sub, long_index_ori,
                                ns_sorted, decomposed_channel_num_ori, decomposed_channel_num_adv, save_directory,
                                decomposition_method, attr_class, layer):
  channel_shap_ori_sorted = channel_shap_ori[ns_sorted]
  channel_shap_adv_sorted = channel_shap_adv[ns_sorted]
  channel_shap_sub_sorted = channel_shap_sub[ns_sorted]
  long_index_ori_sorted = [long_index_ori[i] for i in ns_sorted]

  decomposition_method = findall('[A-Z]', decomposition_method)
  decomposition_method = ''.join(decomposition_method)

  with open(save_directory + '/' + decomposition_method + '_' + attr_class + '_ShapleySameGroups.txt', 'w') as f:
    f.write("There are {} groups in {}. Original group nums is: {}\n".format(decomposed_channel_num_ori, layer, decomposed_channel_num_adv))
    for i_decomposed_channel_num_ori in range(decomposed_channel_num_ori):
      f.write("------------------------------ Group {} -----------------------------------------\n".format(i_decomposed_channel_num_ori))
      channel_shap_ori_short = channel_shap_ori_sorted[i_decomposed_channel_num_ori, long_index_ori_sorted[i_decomposed_channel_num_ori]]
      index_temp = np.argsort(-channel_shap_ori_short)
      channel_shap_ori_short = channel_shap_ori_short[index_temp]
      channel_shap_adv_short = channel_shap_adv_sorted[i_decomposed_channel_num_ori, long_index_ori_sorted[i_decomposed_channel_num_ori]][index_temp]
      channel_shap_sub_short = channel_shap_sub_sorted[i_decomposed_channel_num_ori, long_index_ori_sorted[i_decomposed_channel_num_ori]][index_temp]
      long_index_ori_short = long_index_ori_sorted[i_decomposed_channel_num_ori][index_temp]

      headers = []
      second_row = []
      third_row = []
      forth_row = []
      length_shap_group = min(len(channel_shap_ori_short), 20)
      for i_length_shap_group in range(length_shap_group):
        headers.append('{:0>3d}'.format(long_index_ori_short[i_length_shap_group]))
        second_row.append('{:.2f}'.format(channel_shap_ori_short[i_length_shap_group]))
        third_row.append('{:.2f}'.format(channel_shap_adv_short[i_length_shap_group]))
        forth_row.append('{:.2f}'.format(channel_shap_sub_short[i_length_shap_group]))

      headers =    ['       index:      total is  {}   '.format(len(long_index_ori_short))] + headers
      second_row = ['Shap Original:     total is {:.2f}'.format(np.sum(channel_shap_ori_short))] + second_row
      third_row =  ['Shap Adversarial:  total is {:.2f}'.format(np.sum(channel_shap_adv_short))] + third_row
      forth_row =  ['Shap Delta Change: total is {:.2f}'.format(np.sum(channel_shap_sub_short))] + forth_row

      table1 = [second_row, third_row, forth_row]
      f.write("Sort using original Shap\n")
      f.write(tabulate.tabulate(table1, headers=headers, tablefmt="simple"))
      f.write("\n\n")
      f.write("Sort using the amount of changed Shap\n")
      shap_adv_index = np.argsort(-channel_shap_adv_short)
      channel_shap_adv_short_sorted = channel_shap_adv_short[shap_adv_index]
      long_index_ori_short_sorted = long_index_ori_short[shap_adv_index]

      headers2 = []
      fifth_row = []
      for i_length_shap_group in range(length_shap_group):
        headers2.append('{:0>3d}'.format(long_index_ori_short_sorted[i_length_shap_group]))
        fifth_row.append('{:.2f}'.format(channel_shap_adv_short_sorted[i_length_shap_group]))
      headers2 =  ['       index:     total is  {}   '.format(len(long_index_ori_short))] + headers2
      fifth_row = ['Shap Adversarial: total is {:.2f}'.format(np.sum(channel_shap_adv_short_sorted))] + fifth_row

      table2 = [fifth_row]
      f.write(tabulate.tabulate(table2, headers=headers2, tablefmt="simple"))
      f.write("\n\n\n")


def format_baseline_method_name(ori_names):
  for i_ori_name in range(len(ori_names)):
    ori_name = ori_names[i_ori_name]
    if ori_name == 'AShapleyBFGSREV':
      ori_names[i_ori_name] = 'AS-QARev'
    elif ori_name == 'AShapleyOpt':
      ori_names[i_ori_name] = 'AS-OPT'
    elif ori_name == 'AShapleyBFGS':
      ori_names[i_ori_name] = 'AS-QA'
    elif ori_name == 'AShapleyPAPERBFGS':
      ori_names[i_ori_name] = 'AS-QA'
    elif ori_name == 'AShapleyQAS':
      ori_names[i_ori_name] = 'AS-QA'
  return ori_names
