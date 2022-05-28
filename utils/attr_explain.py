# Added more attribution methods than original DeepExplain library
# Refer to https://github.com/marcoancona/DeepExplain
# But I find Aumann-Shapley can always produce better results for most cases I tested

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from skimage.util import view_as_windows
import warnings, logging
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_grad, math_grad
from collections import OrderedDict
from .utils import make_batches, slice_arrays, to_list, unpack_singleton, placeholder_from_data
from scipy.ndimage import gaussian_filter
import os


SUPPORTED_ACTIVATIONS = [
    'Relu', 'Elu', 'Sigmoid', 'Tanh', 'Softplus'
]

DEEPSHAP_OVERRIDE = [
    'Relu', 'Elu', 'Sigmoid', 'Tanh', 'Softplus', 'Exp', 'ClipByValue', 'Rsqrt', 'Square', 'Max'
]

DEEPSHAPMaxPool_OVERRIDE = [
    'MaxPool'
]

UNSUPPORTED_ACTIVATIONS = [
    'CRelu', 'Relu6', 'Softsign'
]

_ENABLED_METHOD_CLASS = None
_GRAD_OVERRIDE_CHECKFLAG = 0


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------


def activation(type):
    """
    Returns Tensorflow's activation op, given its type
    :param type: string
    :return: op
    """
    if type not in SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % type)
    f = getattr(tf.nn, type.lower())
    return f


def original_grad(op, grad):
    """
    Return original Tensorflow gradient for an op
    :param op: op
    :param grad: Tensor
    :return: Tensor
    """
    if op.type not in SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % op.type)
    opname = '_%sGrad' % op.type
    if hasattr(nn_grad, opname):
        f = getattr(nn_grad, opname)
    else:
        f = getattr(math_grad, opname)
    return f(op, grad)


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS BASE CLASSES
# -----------------------------------------------------------------------------


class AttributionMethod(object):
    """
    Attribution method base class
    """
    def __init__(self, T, X, session, keras_learning_phase=None):
        self.T = T  # target Tensor
        self.X = X  # input Tensor
        self.Y_shape = [None,] + T.get_shape().as_list()[1:]
        # Most often T contains multiple output units. In this case, it is often necessary to select
        # a single unit to compute contributions for. This can be achieved passing 'ys' as weight for the output Tensor.
        self.Y = tf.placeholder(tf.float32, self.Y_shape)
        # placeholder_from_data(ys) if ys is not None else 1.0  # Tensor that represents weights for T
        self.T = self.T * self.Y
        self.symbolic_attribution = None
        self.session = session
        self.keras_learning_phase = keras_learning_phase
        self.has_multiple_inputs = type(self.X) is list or type(self.X) is tuple
        logging.info('Model with multiple inputs: %s' % self.has_multiple_inputs)

        # Set baseline
        # TODO: now this sets a baseline also for those methods that does not require it
        self._set_check_baseline()

        # References
        self._init_references()

        # Create symbolic explanation once during construction (affects only gradient-based methods)
        self.explain_symbolic()

    def explain_symbolic(self):
        return None

    def run(self, xs, ys=None, batch_size=None):
        pass

    def _init_references(self):
        pass

    def _check_input_compatibility(self, xs, ys=None, batch_size=None):
        if ys is not None and len(ys) != len(xs):
            raise RuntimeError('When provided, the number of elements in ys must equal the number of elements in xs')
        if batch_size is not None and batch_size > 0:
            if self.T.shape[0].value is not None and self.T.shape[0].value is not batch_size:
                raise RuntimeError('When using batch evaluation, the first dimension of the target tensor '
                                   'must be compatible with the batch size. Found %s instead' % self.T.shape[0].value)
            if isinstance(self.X, list):
                for x in self.X:
                    if x.shape[0].value is not None and x.shape[0].value is not batch_size:
                        raise RuntimeError('When using batch evaluation, the first dimension of the input tensor '
                                           'must be compatible with the batch size. Found %s instead' % x.shape[
                                               0].value)
            else:
                if self.X.shape[0].value is not None and self.X.shape[0].value is not batch_size:
                    raise RuntimeError('When using batch evaluation, the first dimension of the input tensor '
                                       'must be compatible with the batch size. Found %s instead' % self.X.shape[0].value)

    def _session_run_batch(self, T, xs, ys=None):
        feed_dict = {}
        if self.has_multiple_inputs:
            for k, v in zip(self.X, xs):
                feed_dict[k] = v
        else:
            feed_dict[self.X] = xs

        # If ys is not passed, produce a vector of ones that will be broadcasted to all batch samples
        feed_dict[self.Y] = ys if ys is not None else np.ones([1,] + self.Y_shape[1:])

        if self.keras_learning_phase is not None:
            feed_dict[self.keras_learning_phase] = 0
        return self.session.run(T, feed_dict)

    def _session_run(self, T, xs, ys=None, batch_size=None):
        num_samples = len(xs)
        if self.has_multiple_inputs is True:
            num_samples = len(xs[0])
            if len(xs) != len(self.X):
                raise RuntimeError('List of input tensors and input data have different lengths (%s and %s)'
                                   % (str(len(xs)), str(len(self.X))))
            if batch_size is not None:
                for xi in xs:
                    if len(xi) != num_samples:
                        raise RuntimeError('Evaluation in batches requires all inputs to have '
                                           'the same number of samples')

        if batch_size is None or batch_size <= 0 or num_samples <= batch_size:
            return self._session_run_batch(T, xs, ys)
        else:
            outs = []
            batches = make_batches(num_samples, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                # Get a batch from data
                xs_batch = slice_arrays(xs, batch_start, batch_end)
                # If the target tensor has one entry for each sample, we need to batch it as well
                ys_batch = None
                if ys is not None:
                    ys_batch = slice_arrays(ys, batch_start, batch_end)
                batch_outs = self._session_run_batch(T, xs_batch, ys_batch)
                batch_outs = to_list(batch_outs)
                if batch_index == 0:
                    # Pre-allocate the results arrays.
                    for batch_out in batch_outs:
                        shape = (num_samples,) + batch_out.shape[1:]
                        outs.append(np.zeros(shape, dtype=batch_out.dtype))
                for i, batch_out in enumerate(batch_outs):
                    outs[i][batch_start:batch_end] = batch_out
            return unpack_singleton(outs)

    def _set_check_baseline(self):
        # Do nothing for those methods that have no baseline required
        if not hasattr(self, "baseline"):
            return

        if self.baseline is None:
            if self.has_multiple_inputs:
                self.baseline = [np.zeros([1,] + xi.get_shape().as_list()[1:], dtype=np.float32) for xi in self.X]
                # self.baseline_input = [np.zeros([1, ] + [224, 224, 3]) for _ in self.X]
            else:
                # Change here to change IG's baselines
                # self.baseline = np.random.normal(scale=0.1, size=[1, ] + self.X.get_shape().as_list()[1:])
                self.baseline = np.zeros([1,] + self.X.get_shape().as_list()[1:], dtype=np.float32)
                # self.baseline_input = np.zeros([1, ] + [224, 224, 3])
        else:
            if self.has_multiple_inputs:
                for i, xi in enumerate(self.X):
                    if list(self.baseline[i].shape) == xi.get_shape().as_list()[1:]:
                        self.baseline[i] = np.expand_dims(self.baseline[i], 0)
                    else:
                        raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                           % (self.baseline[i].shape, self.X.get_shape().as_list()[1:]))
            else:
                if list(self.baseline.shape) == self.X.get_shape().as_list()[1:]:
                    self.baseline = np.expand_dims(self.baseline, 0)
                else:
                    raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                       % (self.baseline.shape, self.X.get_shape().as_list()[1:]))


class GradientBasedMethod(AttributionMethod):
    """
    Base class for gradient-based attribution methods
    """
    def get_symbolic_attribution(self):
        return tf.gradients(self.T, self.X)

    def explain_symbolic(self):
        if self.symbolic_attribution is None:
            self.symbolic_attribution = self.get_symbolic_attribution()
        return self.symbolic_attribution

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)
        results = self._session_run(self.explain_symbolic(), xs, ys, batch_size)
        return results[0] if not self.has_multiple_inputs else results

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return original_grad(op, grad)


class PerturbationBasedMethod(AttributionMethod):
    """
       Base class for perturbation-based attribution methods
       """
    def __init__(self, T, X, session, keras_learning_phase):
        super(PerturbationBasedMethod, self).__init__(T, X, session, keras_learning_phase)
        self.base_activation = None


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS
# -----------------------------------------------------------------------------

class AShapley(GradientBasedMethod):

    def __init__(self, T, X, session, keras_learning_phase, steps=100, baseline=None,
                 baseline_type='ZERO', originalX=None):
        self.steps = steps
        has_multiple_inputs = type(X) is list or type(X) is tuple
        if baseline_type == 'blur':
            if has_multiple_inputs:
                baselines = []
                for xi in originalX:
                    sigma = [0, xi.shape[1] * 0.03, xi.shape[1] * 0.03, 0]
                    result = gaussian_filter(xi, sigma=sigma)
                    baselines.append(result)
                self.baseline = baselines # [np.zeros([1, ] + xi.get_shape().as_list()[1:]) for xi in originalX]
            else:
                sigma = [0, originalX.shape[1] * 0.03, originalX.shape[1] * 0.03, 0]
                result = gaussian_filter(originalX, sigma=sigma)
                self.baseline = result[0]
        elif os.path.isfile(baseline_type):
            self.baseline = np.load(baseline_type).astype(np.float32)
        else:
            print('baseline_type cannot correspond to a file')
            self.baseline = baseline
        super(AShapley, self).__init__(T, X, session, keras_learning_phase)

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)

        gradient = None
        for alpha in list(np.linspace(1. / self.steps, 1.0, self.steps)):
            xs_mod = [b + (x - b) * alpha for x, b in zip(xs, self.baseline)] if self.has_multiple_inputs \
                else self.baseline + (xs - self.baseline) * alpha
            # aaa = np.sum(xs_mod[0], -1)
            # aaa1 = np.sum(xs[0], -1)
            # aaa2 = np.sum(self.baseline[0], -1)
            _attr = self._session_run(self.explain_symbolic(), xs_mod, ys, batch_size)
            if gradient is None:
                gradient = _attr
            else:
                gradient = [g + a for g, a in zip(gradient, _attr)]
        # gradient = gradient.astype(np.float32)
        results = []
        for g, x, b in zip(gradient,
                           xs if self.has_multiple_inputs else [xs],
                           self.baseline if self.has_multiple_inputs else [self.baseline]):
            aaa = np.sum((x - b)[0], axis=-1)
            aaa1 = np.sum((g * (x - b) / self.steps)[0], axis=-1)
            results.append((g * (x - b) / self.steps).astype(np.float32))

        # results = [(g * (x - b) / self.steps).astype(np.float32) for g, x, b in zip(
        #     gradient,
        #     xs if self.has_multiple_inputs else [xs],
        #     self.baseline if self.has_multiple_inputs else [self.baseline])]

        return results[0] if not self.has_multiple_inputs else results


"""
Returns zero attributions. For testing only.
"""


class DummyZero(GradientBasedMethod):

    def get_symbolic_attribution(self,):
        return tf.gradients(self.T, self.X)

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        input = op.inputs[0]
        return tf.zeros_like(input)

"""
Saliency maps
https://arxiv.org/abs/1312.6034
"""


class Saliency(GradientBasedMethod):

    def get_symbolic_attribution(self):
        return [tf.abs(g) for g in tf.gradients(self.T, self.X)]


"""
Gradient * Input
https://arxiv.org/pdf/1704.02685.pdf - https://arxiv.org/abs/1611.07270
"""


class GradientxInput(GradientBasedMethod):

    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X])]




"""
Smooth Gradients
https://arxiv.org/pdf/1706.03825.pdf
"""
class SmoothGradients(GradientBasedMethod):

    def __init__(self, T, X, session, keras_learning_phase, steps=100):
        self.steps = steps
        super(SmoothGradients, self).__init__(T, X, session, keras_learning_phase)

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)
        stdev = 0.15 * (np.max(xs) - np.min(xs))
        gradient = None
        for _ in range(self.steps):
            if self.has_multiple_inputs:
                xs_mod = []
                for x in xs:
                    random_noise = np.random.normal(0, stdev, xs.shape).astype(np.float32)
                    x = x + random_noise
                    x = (x - np.min(x)) / (np.max(x) - np.min(x))
                    xs_mod.append(x)
            else:
                random_noise = np.random.normal(0, stdev, xs.shape).astype(np.float32)
                xs_mod = xs + random_noise
                xs_mod = (xs_mod - np.min(xs_mod)) / (np.max(xs_mod) - np.min(xs_mod))
            _attr = self._session_run(self.explain_symbolic(), xs_mod, ys, batch_size)
            if gradient is None:
                gradient = _attr
            else:
                gradient = [g + a for g, a in zip(gradient, _attr)]
        results = [g / self.steps for g in gradient]

        return results[0] if not self.has_multiple_inputs else results


"""
Layer-wise Relevance Propagation with epsilon rule
http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
"""
class EpsilonLRP(GradientBasedMethod):
    eps = None

    def __init__(self, T, X, session, keras_learning_phase, epsilon=1e-3):
        assert epsilon > 0.0, 'LRP epsilon must be greater than zero'
        global eps
        eps = epsilon
        super(EpsilonLRP, self).__init__(T, X, session, keras_learning_phase)

    def get_symbolic_attribution(self):
        # return [g * x for g, x in zip(
        #     tf.gradients(self.T, self.X),
        #     self.X if self.has_multiple_inputs else [self.X])]
        tensor_list = zip(tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X])
        return [g * x for g, x in tensor_list]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        small_input_tensor = eps * tf.where(input >= 0, tf.ones_like(input), -1 * tf.ones_like(input))
        overrided_grad = grad * output / (input + small_input_tensor)
        return overrided_grad
        # return grad * output / (input + eps *
        #                         tf.where(input >= 0, tf.ones_like(input), -1 * tf.ones_like(input)))



"""
DeepLIFT
This reformulation only considers the "Rescale" rule
https://arxiv.org/abs/1704.02685
"""
class DeepLIFTRescale(GradientBasedMethod):

    _deeplift_ref = {}

    def __init__(self, T, X, session, keras_learning_phase, baseline=None, ref_dir=None):
        self.baseline = baseline
        self.ref_dir = ref_dir
        if os.path.isfile(ref_dir):
            YR_temp = np.load(ref_dir)
            self.YR = []
            for i in range(len(YR_temp.keys())):
                self.YR.append(YR_temp['arr_'+str(i)])
            # self.YR = [YR_temp[key] for key in YR_temp]
        else:
            self.YR_name = ref_dir
            self.YR = None
        super(DeepLIFTRescale, self).__init__(T, X, session, keras_learning_phase)

    def get_symbolic_attribution(self):
        # return [g * (x - b) for g, x, b in zip(
        #     tf.gradients(self.T, self.X),
        #     self.X if self.has_multiple_inputs else [self.X],
        #     self.baseline if self.has_multiple_inputs else [self.baseline])]
        if self.has_multiple_inputs:
            baseline_zip = self.baseline
            x_zip = self.X
        else:
            baseline_zip = [self.baseline]
            x_zip = [self.X]
        return_list = []
        gradients_zip = tf.gradients(self.T, self.X)
        for g, x, b in zip(gradients_zip, x_zip, baseline_zip):
            return_list.append(g*(x-b))
        return return_list

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        ref_input = cls._deeplift_ref[op.name]
        ref_output = activation(op.type)(ref_input)
        delta_out = output - ref_output
        delta_in = input - ref_input
        instant_grad = activation(op.type)(0.5 * (ref_input + input))
        return tf.where(tf.abs(delta_in) > 1e-5, grad * delta_out / delta_in,
                        original_grad(instant_grad.op, grad))

    def _init_references(self):
        # print ('DeepLIFT: computing references...')
        sys.stdout.flush()
        self._deeplift_ref.clear()
        ops = []
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in SUPPORTED_ACTIVATIONS:
                    ops.append(op)
        if self.YR is not None:
            YR = self.YR
        else:
            YR = self._session_run([o.inputs[0] for o in ops], self.baseline)
            np.savez(self.YR_name, *YR)
        for (r, op) in zip(YR, ops):
            self._deeplift_ref[op.name] = r
        # print('DeepLIFT: references ready')
        sys.stdout.flush()


"""
DeepLIFTRevelcancel
This reformulation only considers the "Revelcancel" rule
https://arxiv.org/abs/1704.02685
"""
class DeepLIFTRevelcancel(GradientBasedMethod):

    _deeplift_ref = {}

    def __init__(self, T, X, session, keras_learning_phase, baseline=None, ref_dir=None):
        self.baseline = baseline
        self.pos_flag = True
        if os.path.isfile(ref_dir):
            YR_temp = np.load(ref_dir)
            self.YR = []
            for i in range(len(YR_temp.keys())):
                self.YR.append(YR_temp['arr_'+str(i)])
        else:
            self.YR_name = ref_dir
            self.YR = None
        super(DeepLIFTRevelcancel, self).__init__(T, X, session, keras_learning_phase)

    def get_symbolic_attribution(self):
        # return [g * (x - b) for g, x, b in zip(
        #     tf.gradients(self.T, self.X),
        #     self.X if self.has_multiple_inputs else [self.X],
        #     self.baseline if self.has_multiple_inputs else [self.baseline])]
        if self.has_multiple_inputs:
            baseline_zip = self.baseline
            x_zip = self.X
        else:
            baseline_zip = [self.baseline]
            x_zip = [self.X]
        return_list = []
        gradients_zip = tf.gradients(self.T, self.X)
        for g, x, b in zip(gradients_zip, x_zip, baseline_zip):
            return_list.append(g*(x-b))
        return return_list

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        # positive DeepLIFTRevelcancel attribution computing
        input = op.inputs[0]
        # output = op.outputs[0]
        ref_input = cls._deeplift_ref[op.name]
        ref_output = activation(op.type)(ref_input)
        delta_in = input - ref_input

        delta_in_pos = tf.where(delta_in > 0, delta_in, tf.fill(tf.shape(delta_in), -1e-6))
        delta_in_neg = tf.where(delta_in < 0, delta_in, tf.fill(tf.shape(delta_in), 1e-6))
        first_output = activation(op.type)(ref_input + delta_in_pos)
        third_output = activation(op.type)(ref_input + delta_in)
        fourth_output = activation(op.type)(ref_input + delta_in_neg)
        delta_out_pos = 0.5*(first_output-ref_output) + 0.5*(third_output-fourth_output)
        return tf.where(tf.abs(delta_in) > 1e-6, grad * delta_out_pos / delta_in_pos,
                        original_grad(op, grad))

    @classmethod
    def nonlinearity_grad_override_neg(cls, op, grad):
        # negative DeepLIFTRevelcancel attribution computing
        input = op.inputs[0]
        # output = op.outputs[0]
        ref_input = cls._deeplift_ref[op.name]
        ref_output = activation(op.type)(ref_input)
        delta_in = input - ref_input

        delta_in_pos = tf.where(delta_in > 0, delta_in, tf.fill(tf.shape(delta_in), -1e-6))
        delta_in_neg = tf.where(delta_in < 0, delta_in, tf.fill(tf.shape(delta_in), 1e-6))
        first_output = activation(op.type)(ref_input + delta_in_neg)
        third_output = activation(op.type)(ref_input + delta_in)
        fourth_output = activation(op.type)(ref_input + delta_in_pos)
        delta_out_neg = 0.5*(first_output-ref_output) + 0.5*(third_output-fourth_output)
        return tf.where(tf.abs(delta_in) > 1e-7, grad * delta_out_neg / delta_in_neg,
                        original_grad(op, grad))

    def _init_references(self):
        # print ('DeepLIFT: computing references...')
        sys.stdout.flush()
        self._deeplift_ref.clear()
        ops = []
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in SUPPORTED_ACTIVATIONS:
                    ops.append(op)
        if self.YR is not None:
            YR = self.YR
        else:
            YR = self._session_run([o.inputs[0] for o in ops], self.baseline)
            np.savez(self.YR_name, *YR)
        # YR = self._session_run([o.inputs[0] for o in ops], self.baseline)
        for (r, op) in zip(YR, ops):
            self._deeplift_ref[op.name] = r
        # print('DeepLIFT: references ready')
        sys.stdout.flush()


"""
DeepSHAP
"""
class DeepSHAP(GradientBasedMethod):

    _deepshap_ref = {}
    _deepshapgrad_ref = {}
    def __init__(self, T, X, session, keras_learning_phase, baseline=None, ref_dir=None):
        self.baseline = baseline
        if os.path.isfile(ref_dir):
            YR_temp = np.load(ref_dir)
            self.YR = []
            for i in range(len(YR_temp.keys())):
                self.YR.append(YR_temp['arr_'+str(i)])
        else:
            self.YR = None
            self.YR_name = ref_dir
        super(DeepSHAP, self).__init__(T, X, session, keras_learning_phase)

    def get_symbolic_attribution(self):
        # return [g * (x - b) for g, x, b in zip(
        #     tf.gradients(self.T, self.X),
        #     self.X if self.has_multiple_inputs else [self.X],
        #     self.baseline if self.has_multiple_inputs else [self.baseline])]
        if self.has_multiple_inputs:
            baseline_zip = self.baseline
            x_zip = self.X
        else:
            baseline_zip = [self.baseline]
            x_zip = [self.X]
        return_list = []
        gradients_zip = tf.gradients(self.T, self.X)
        for g, x, b in zip(gradients_zip, x_zip, baseline_zip):
            return_list.append(g*(x-b))
        return return_list

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        ref_input = cls._deepshap_ref[op.name]
        ref_output = activation(op.type)(ref_input)
        delta_out = output - ref_output
        delta_in = input - ref_input
        # instant_grad = activation(op.type)(0.5 * (ref_input + input))
        # return tf.where(tf.abs(delta_in) > 1e-5, grad * delta_out / delta_in,
        #                 original_grad(instant_grad.op, grad))
        attribution = tf.where(tf.abs(delta_in) < 1e-6,
                               original_grad(op, grad), grad * delta_out / delta_in)
        return attribution

    @classmethod
    def maxpool_grad_override(cls, op, grad):
        input = op.inputs[0]
        output = op.outputs[0]
        ref_input = cls._deepshap_ref[op.name]
        ref_output = getattr(tf.nn, 'max_pool')(ref_input, op.get_attr('ksize'), op.get_attr('strides'), op.get_attr('padding'))

        delta_in = input - ref_input
        cross_max = tf.maximum(output, ref_output)
        diffs = cross_max - ref_output
        xmax_pos = original_grad(op, grad * diffs)
        attribution = tf.where(tf.abs(delta_in) < 1e-7,
                               tf.zeros_like(delta_in),
                               xmax_pos / delta_in)
        return attribution

    def _init_references(self):
        # print ('DeepLIFT: computing references...')
        sys.stdout.flush()
        self._deepshap_ref.clear()
        self._deepshapgrad_ref.clear()
        ops = []
        ops_grad = []
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in DEEPSHAP_OVERRIDE:
                    ops.append(op)
                elif op.type in DEEPSHAPMaxPool_OVERRIDE:
                    ops.append(op)
                    ops_grad.append(op)
        if self.YR is not None:
            YR = self.YR
        else:
            YR = self._session_run([o.inputs[0] for o in ops], self.baseline)
            np.savez(self.YR_name, *YR)
        # YR = self._session_run(op_inputs, self.baseline)
        # YR = self._session_run([o.inputs[0] for o in ops], self.baseline)
        for (r, op) in zip(YR, ops):
            self._deepshap_ref[op.name] = r
        sys.stdout.flush()
        # t_grad_list = []
        # for o in ops_grad:
        #     t_grad_list.append(tf.gradients(self.T, self.session.graph.get_tensor_by_name(o.outputs[0].name)))
        # feed_dict = {self.X: self.baseline}
        # Grad = self.session.run(t_grad_list, feed_dict)
        # for (r, op) in zip(Grad, ops_grad):
        #     self._deepshapgrad_ref[op.name] = r
        # sys.stdout.flush()


"""
Occlusion method
Generalization of the grey-box method presented in https://arxiv.org/pdf/1311.2901.pdf
This method performs a systematic perturbation of contiguous hyperpatches in the input,
replacing each patch with a user-defined value (by default 0).

window_shape : integer or tuple of length xs_ndim
Defines the shape of the elementary n-dimensional orthotope the rolling window view.
If an integer is given, the shape will be a hypercube of sidelength given by its value.

step : integer or tuple of length xs_ndim
Indicates step size at which extraction shall be performed.
If integer is given, then the step is uniform in all dimensions.
"""
class Occlusion(PerturbationBasedMethod):

    def __init__(self, T, X, session, keras_learning_phase, window_shape=None, step=None):
        super(Occlusion, self).__init__(T, X, session, keras_learning_phase)
        if self.has_multiple_inputs:
            raise RuntimeError('Multiple inputs not yet supported for perturbation methods')

        input_shape = X[0].get_shape().as_list()
        if window_shape is not None:
            assert len(window_shape) == len(input_shape), \
                'window_shape must have length of input (%d)' % len(input_shape)
            self.window_shape = tuple(window_shape)
        else:
            self.window_shape = (1,) * len(input_shape)

        if step is not None:
            assert isinstance(step, int) or len(step) == len(input_shape), \
                'step must be integer or tuple with the length of input (%d)' % len(input_shape)
            self.step = step
        else:
            self.step = 1
        self.replace_value = 0.0
        logging.info('Input shape: %s; window_shape %s; step %s' % (input_shape, self.window_shape, self.step))

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)
        input_shape = xs.shape[1:]
        batch_size = xs.shape[0]
        # total_dim = np.asscalar(np.prod(input_shape))
        total_dim = np.prod(input_shape).item()
        # Create mask
        index_matrix = np.arange(total_dim).reshape(input_shape)
        idx_patches = view_as_windows(index_matrix, self.window_shape, self.step).reshape((-1,) + self.window_shape)
        heatmap = np.zeros_like(xs, dtype=np.float32).reshape((-1), total_dim)
        w = np.zeros_like(heatmap)

        # Compute original output
        eval0 = self._session_run(self.T, xs, ys, batch_size)

        # Start perturbation loop
        for i, p in enumerate(idx_patches):
            mask = np.ones(input_shape).flatten()
            mask[p.flatten()] = self.replace_value
            masked_xs = mask.reshape((1,) + input_shape) * xs
            delta = eval0 - self._session_run(self.T, masked_xs, ys, batch_size)
            delta_aggregated = np.sum(delta.reshape((batch_size, -1)), -1, keepdims=True)
            heatmap[:, p.flatten()] += delta_aggregated
            w[:, p.flatten()] += p.size

        attribution = np.reshape(heatmap / w, xs.shape)
        if np.isnan(attribution).any():
            warnings.warn('Attributions generated by Occlusion method contain nans, '
                          'probably because window_shape and step do not allow to cover the all input.')
        return attribution


"""
Shapley Value sampling
Computes approximate Shapley Values using "Polynomial calculation of the Shapley value based on sampling",
Castro et al, 2009 (https://www.sciencedirect.com/science/article/pii/S0305054808000804)

samples : integer (default 5)
Defined the number of samples for each input feature. 
Notice that evaluating a model samples * n_input_feature times might take a while.

sampling_dims : list of dimension indexes to run sampling on (feature dimensions).
By default, all dimensions except the batch dimension will be sampled.
For example, with a 4-D tensor that contains color images, single color channels are sampled.
To sample pixels, instead, use sampling_dims=[1,2]
"""
class ShapleySampling(PerturbationBasedMethod):

    def __init__(self, T, X, session, keras_learning_phase, samples=30, sampling_dims=None, is_pixel=False):
        super(ShapleySampling, self).__init__(T, X, session, keras_learning_phase)
        if self.has_multiple_inputs:
            raise RuntimeError('Multiple inputs not yet supported for perturbation methods')
        dims = len(X.shape)
        if sampling_dims is not None:
            if not 0 < len(sampling_dims) <= (dims - 1):
                raise RuntimeError('sampling_dims must be a list containing 1 to %d elements' % (dims-1))
            if 0 in sampling_dims:
                raise RuntimeError('Cannot sample batch dimension: remove 0 from sampling_dims')
            if any([x < 1 or x > dims-1 for x in sampling_dims]):
                raise RuntimeError('Invalid value in sampling_dims')
        else:
            sampling_dims = list(range(1, dims))
        self.is_pixel = is_pixel
        self.samples = samples
        self.sampling_dims = sampling_dims

    def run(self, xs, ys=None, batch_size=None):
        xs_shape = list(xs.shape)
        batch_size = xs.shape[0]
        if self.is_pixel:
            # n_features = int(np.asscalar(np.prod([xs.shape[i] for i in self.sampling_dims])))
            n_features = int(np.prod([xs.shape[i] for i in self.sampling_dims[0:2]]).item())
            result = np.zeros((xs_shape[0], n_features))

            run_shape = list((xs_shape[0], -1, xs_shape[-1]))  # a copy
            reconstruction_shape = [xs_shape[0]]
            for j in self.sampling_dims:
                reconstruction_shape.append(xs_shape[j])

            for r in range(self.samples):
                p = np.random.permutation(n_features)
                x = xs.copy().reshape(run_shape)
                y = None
                for i in p:
                    if y is None:
                        y = self._session_run(self.T, x.reshape(xs_shape), ys, batch_size)
                    x[:, i, :] = 0
                    y0 = self._session_run(self.T, x.reshape(xs_shape), ys, batch_size)
                    delta = y - y0
                    delta_aggregated = np.sum(delta.reshape((batch_size, -1)), -1, keepdims=False)
                    result[:, i] += delta_aggregated
                    y = y0

            shapley = result / self.samples
            shapley = np.stack((shapley, shapley, shapley), axis=-1)
            return shapley.reshape(reconstruction_shape)
        else:
            n_features = int(np.prod([xs.shape[i] for i in self.sampling_dims]))
            result = np.zeros((xs_shape[0], n_features))

            run_shape = list(xs_shape)  # a copy
            run_shape = np.delete(run_shape, self.sampling_dims).tolist()
            run_shape.insert(1, -1)

            reconstruction_shape = [xs_shape[0]]
            for j in self.sampling_dims:
                reconstruction_shape.append(xs_shape[j])

            for r in range(self.samples):
                p = np.random.permutation(n_features)
                x = xs.copy().reshape(run_shape)
                y = None
                for i in p:
                    if y is None:
                        y = self._session_run(self.T, x.reshape(xs_shape), ys, batch_size)
                    x[:, i] = 0
                    y0 = self._session_run(self.T, x.reshape(xs_shape), ys, batch_size)
                    delta = y - y0
                    delta_aggregated = np.sum(delta.reshape((batch_size, -1)), -1, keepdims=False)
                    result[:, i] += delta_aggregated
                    y = y0

            shapley = result / self.samples
            return shapley.reshape(reconstruction_shape)

# -----------------------------------------------------------------------------
# END ATTRIBUTION METHODS
# -----------------------------------------------------------------------------


attribution_methods = OrderedDict({
    'DummyZERO': (DummyZero, 0),
    'Saliency': (Saliency, 1),
    'GradxInput': (GradientxInput, 2),
    'SmoothGrad': (SmoothGradients, 3),
    'EpsilonLRP': (EpsilonLRP, 4),
    'DeepLIFTRescale': (DeepLIFTRescale, 5),
    'Occlusion': (Occlusion, 6),
    'AShapley': (AShapley, 7),
    'DeepSHAP': (DeepSHAP, 8),
    'ShapleySampling': (ShapleySampling, 9),
    'DeepLIFTReveal': (DeepLIFTRevelcancel, 10)
})


@ops.RegisterGradient("DeepExplainGrad")
def deepexplain_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
    _GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None \
        and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
            return _ENABLED_METHOD_CLASS.nonlinearity_grad_override(op, grad)
    else:
        return original_grad(op, grad)


@ops.RegisterGradient("DeepLIFTNegGrad")
def deepliftneg_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
    _GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None \
        and issubclass(_ENABLED_METHOD_CLASS, DeepLIFTRevelcancel):
            return _ENABLED_METHOD_CLASS.nonlinearity_grad_override_neg(op, grad)
    else:
        return original_grad(op, grad)


@ops.RegisterGradient("DeepSHAPMaxPoolGrad")
def deepshapmaxpool_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
    _GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None \
        and issubclass(_ENABLED_METHOD_CLASS, DeepSHAP):
            return _ENABLED_METHOD_CLASS.maxpool_grad_override(op, grad)
    else:
        return original_grad(op, grad)


class DeepExplain(object):

    def __init__(self, graph=None, session=tf.get_default_session(), **kwargs):
        self.method = None
        self.batch_size = None
        self.session = session
        self.graph = session.graph if graph is None else graph
        self.graph_context = self.graph.as_default()
        if "flag" in kwargs.keys():
            self.override_context = self.graph.gradient_override_map(self.get_override_map(kwargs["flag"]))
            self.flag = kwargs["flag"]
        else:
            self.override_context = self.graph.gradient_override_map(self.get_override_map())
        self.keras_phase_placeholder = None
        self.context_on = False
        if self.session is None:
            raise RuntimeError('DeepExplain: could not retrieve a session. Use DeepExplain(session=your_session).')

    def __enter__(self):
        # Override gradient of all ops created in context
        self.graph_context.__enter__()
        self.override_context.__enter__()
        self.context_on = True
        return self

    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)
        self.context_on = False

    def get_explainer(self, method, T, X, **kwargs):
        if not self.context_on:
            raise RuntimeError('Explain can be called only within a DeepExplain context.')
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG, METHOD_FLAG
        self.method = method
        if self.method in attribution_methods:
            method_class, method_flag = attribution_methods[self.method]
        else:
            raise RuntimeError('Method must be in %s' % list(attribution_methods.keys()))
        if isinstance(X, list):
            for x in X:
                if 'tensor' not in str(type(x)).lower():
                    raise RuntimeError('If a list, X must contain only Tensorflow Tensor objects')
        else:
            if 'tensor' not in str(type(X)).lower():
                raise RuntimeError('X must be a Tensorflow Tensor object or a list of them')

        if 'tensor' not in str(type(T)).lower():
            raise RuntimeError('T must be a Tensorflow Tensor object')

        logging.info('DeepExplain: running "%s" explanation method (%d)' % (self.method, method_flag))
        self._check_ops()
        _GRAD_OVERRIDE_CHECKFLAG = 0
        METHOD_FLAG = self.flag
        _ENABLED_METHOD_CLASS = method_class
        method = _ENABLED_METHOD_CLASS(T, X,
                                       self.session,
                                       keras_learning_phase=self.keras_phase_placeholder,
                                       **kwargs)

        if issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod) and _GRAD_OVERRIDE_CHECKFLAG == 0:
            warnings.warn('DeepExplain detected you are trying to use an attribution method that requires '
                          'gradient override but the original gradient was used instead. You might have forgot to '
                          '(re)create your graph within the DeepExlain context. Results are not reliable!')
        _ENABLED_METHOD_CLASS = None
        _GRAD_OVERRIDE_CHECKFLAG = 0
        self.keras_phase_placeholder = None
        return method

    def explain(self, method, T, X, xs, ys=None, batch_size=None, **kwargs):
        explainer = self.get_explainer(method, T, X, **kwargs)
        result = explainer.run(xs, ys, batch_size)
        return result

    @staticmethod
    def get_override_map(flag=None):
        if flag is not None:
            if flag == 'DeepSHAP':
                override_dict = dict((a, 'DeepExplainGrad') for a in DEEPSHAP_OVERRIDE)
                override_dict.update(dict((a, 'DeepSHAPMaxPoolGrad') for a in DEEPSHAPMaxPool_OVERRIDE))
            elif flag == 'DeepLIFTRevealCancelPos':
                override_dict = dict((a, 'DeepExplainGrad') for a in SUPPORTED_ACTIVATIONS)
            elif flag == 'DeepLIFTRevealCancelNeg':
                override_dict = dict((a, 'DeepLIFTNegGrad') for a in SUPPORTED_ACTIVATIONS)
            else:
                override_dict = dict((a, 'DeepExplainGrad') for a in SUPPORTED_ACTIVATIONS)
        else:
            override_dict = dict((a, 'DeepExplainGrad') for a in SUPPORTED_ACTIVATIONS)
        return override_dict

    def _check_ops(self):
        """
        Heuristically check if any op is in the list of unsupported activation functions.
        This does not cover all cases where explanation methods would fail, and must be improved in the future.
        Also, check if the placeholder named 'keras_learning_phase' exists in the graph. This is used by Keras
         and needs to be passed in feed_dict.
        :return:
        """
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in UNSUPPORTED_ACTIVATIONS:
                    warnings.warn('Detected unsupported activation (%s). '
                                  'This might lead to unexpected or wrong results.' % op.type)
            elif 'keras_learning_phase' in op.name:
                self.keras_phase_placeholder = op.outputs[0]
        # print('check ops names OK, all ops are supported')




