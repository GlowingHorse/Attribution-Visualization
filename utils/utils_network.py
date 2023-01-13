import tensorflow as tf
from lucid.modelzoo.vision_base import Model, _layers_from_list_of_dicts


def _populate_inception_bottlenecks(scope):
  """Add Inception bottlenecks and their pre-Relu versions to the graph."""
  graph = tf.get_default_graph()
  for op in graph.get_operations():
    if op.name.startswith(scope+'/') and 'Concat' in op.type:
      name = op.name.split('/')[1]
      pre_relus = []
      for tower in op.inputs[1:]:
        if tower.op.type == 'Relu':
          tower = tower.op.inputs[0]
        pre_relus.append(tower)
      concat_name = scope + '/' + name + '_pre_relu'
      _ = tf.concat(pre_relus, -1, name=concat_name)


class InceptionV1(Model):
  """InceptionV1 (or 'GoogLeNet')

  This is a (re?)implementation of InceptionV1
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
  The weights were trained at Google and released in an early TensorFlow
  tutorial. It is possible the parameters are the original weights
  (trained in TensorFlow's predecessor), but we haven't been able to
  confirm this.

  As far as we can tell, it is exactly the same as the model described in
  the original paper, where as the slim and caffe implementations have
  minor implementation differences (such as eliding the heads).
  """
  model_path = 'gs://modelzoo/vision/other_models/InceptionV1.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_alternate.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]
  image_value_range = (-117, 255-117)
  input_name = 'input:0'

  def post_import(self, scope):
    _populate_inception_bottlenecks(scope)

InceptionV1.layers = _layers_from_list_of_dicts(InceptionV1, [
  {'tags': ['conv'], 'name': 'conv2d0', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2d1', 'depth': 64},
  {'tags': ['conv'], 'name': 'conv2d2', 'depth': 192},
  {'tags': ['conv'], 'name': 'mixed3a', 'depth': 256},
  {'tags': ['conv'], 'name': 'mixed3b', 'depth': 480},
  {'tags': ['conv'], 'name': 'maxpool4', 'depth': 480},
  {'tags': ['conv'], 'name': 'mixed4a', 'depth': 508},
  {'tags': ['conv'], 'name': 'mixed4b', 'depth': 512},
  {'tags': ['conv'], 'name': 'mixed4c', 'depth': 512},
  {'tags': ['conv'], 'name': 'mixed4d', 'depth': 528},
  {'tags': ['conv'], 'name': 'mixed4e', 'depth': 832},
  {'tags': ['conv'], 'name': 'maxpool10', 'depth': 832},
  {'tags': ['conv'], 'name': 'mixed5a', 'depth': 832},
  {'tags': ['conv'], 'name': 'mixed5b', 'depth': 1024},
  {'tags': ['conv'], 'name': 'head0_bottleneck', 'depth': 128},
  {'tags': ['dense'], 'name': 'nn0', 'depth': 1024},
  {'tags': ['dense'], 'name': 'softmax0', 'depth': 1008},
  {'tags': ['conv'], 'name': 'head1_bottleneck', 'depth': 128},
  {'tags': ['dense'], 'name': 'nn1', 'depth': 1024},
  {'tags': ['dense'], 'name': 'softmax1', 'depth': 1008},
  {'tags': ['dense'], 'name': 'softmax2', 'depth': 1008},
])


class ResnetV1_50_slim(Model):
  """ResnetV150 as implemented by the TensorFlow slim framework.

  This function provides the pre-trained reimplementation from TF slim:
  https://github.com/tensorflow/models/tree/master/research/slim
  """

  model_path  = 'gs://modelzoo/vision/slim_models/ResnetV1_50.pb'
  labels_path = 'gs://modelzoo/labels/ImageNet_standard.txt'
  dataset = 'ImageNet'
  image_shape = [224, 224, 3]


  image_value_range = (-117, 255-117) # Inferred by testing, may not be exactly right
  input_name = 'input'

# In ResNetV1, each add (joining the residual branch) is followed by a Relu
# this seems to be the natural "layer" position
ResnetV1_50_slim.layers = _layers_from_list_of_dicts(ResnetV1_50_slim, [
  {'tags': ['conv'], 'name': 'resnet_v1_50/conv1/Relu', 'depth': 64},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block1/unit_1/bottleneck_v1/Relu', 'depth': 256},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block1/unit_2/bottleneck_v1/Relu', 'depth': 256},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block1/unit_3/bottleneck_v1/Relu', 'depth': 256},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block2/unit_1/bottleneck_v1/Relu', 'depth': 512},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block2/unit_2/bottleneck_v1/Relu', 'depth': 512},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block2/unit_3/bottleneck_v1/Relu', 'depth': 512},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block2/unit_4/bottleneck_v1/Relu', 'depth': 512},

  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_1/bottleneck_v1/add', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_1/bottleneck_v1/Relu', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_2/bottleneck_v1/add', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_2/bottleneck_v1/Relu', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_3/bottleneck_v1/add', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_3/bottleneck_v1/Relu', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_4/bottleneck_v1/add', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_4/bottleneck_v1/Relu', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_5/bottleneck_v1/add', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_5/bottleneck_v1/Relu', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block3/unit_6/bottleneck_v1/Relu', 'depth': 1024},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block4/unit_1/bottleneck_v1/Relu', 'depth': 2048},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block4/unit_2/bottleneck_v1/Relu', 'depth': 2048},
  {'tags': ['conv'], 'name': 'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu', 'depth': 2048},
  {'tags': ['dense'], 'name': 'resnet_v1_50/predictions/Softmax', 'depth': 1000},
])