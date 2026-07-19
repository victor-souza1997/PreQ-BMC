import os
from utils.preqbmc_utils import *

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from copy import deepcopy


class DP_DNN_neuron(object):

    def __init__(self):
        self.algebra_lower = None
        self.algebra_upper = None
        self.concrete_algebra_lower = None
        self.concrete_algebra_upper = None
        self.concrete_lower = None
        self.concrete_lower_noClip = None
        self.concrete_upper = None
        self.concrete_upper_noClip = None
        self.concrete_highest_lower = None
        self.concrete_lowest_upper = None
        self.weight = None
        self.bias = None
        self.prev_abs_mode = None
        self.prev_abs_mode_min = None
        self.certain_flag = 0
        self.actMode = 0  # 1: activated ; 2: deactivated; 3: lb+ub>=0; 4: lb+ub<0

    def clear(self):
        self.certain_flag = 0
        self.actMode = 0
        self.concrete_highest_lower = None
        self.concrete_lowest_upper = None
        self.prev_abs_mode = None


class DP_DNN_layer(object):
    INPUT_LAYER = 0
    AFFINE_LAYER = 1
    RELU_LAYER = 2

    def __init__(self):
        self.size = None
        self.neurons = None
        self.layer_type = None

    def clear(self):
        for i in range(len(self.neurons)):
            self.neurons[i].clear()


class DP_DNN_network(object):

    def __init__(self, ifSignedOutput):
        self.MODE_QUANTITIVE = 0
        self.MODE_ROBUSTNESS = 1

        self.numlayers = None
        self.layerSizes = None
        self.inputSize = None
        self.outputSize = None
        self.mins = None
        self.maxes = None
        self.ranges = None
        self.layers = None
        self.property_flag = None
        self.property_region = None
        self.abs_mode_changed = None
        self.reluN = 6
        self.outSigned = True
        self.outputSigned = ifSignedOutput

    def clear(self):
        for i in range(len(self.layers)):
            self.layers[i].clear()

    def deeppoly(self):

        def pre(cur_neuron, i):
            if i == 0:
                cur_neuron.concrete_algebra_lower = deepcopy(cur_neuron.algebra_lower)
                cur_neuron.concrete_algebra_upper = deepcopy(cur_neuron.algebra_upper)
            lower_bound = deepcopy(cur_neuron.algebra_lower)
            upper_bound = deepcopy(cur_neuron.algebra_upper)
            for k in range(i + 1)[::-1]:
                tmp_lower = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                tmp_upper = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                assert (self.layers[k].size + 1 == len(lower_bound))
                assert (self.layers[k].size + 1 == len(upper_bound))
                for p in range(self.layers[k].size):
                    if lower_bound[p] >= 0:
                        tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_lower
                    else:
                        tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_upper

                    if upper_bound[p] >= 0:
                        tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_upper
                    else:
                        tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_lower
                tmp_lower[-1] += lower_bound[-1]
                tmp_upper[-1] += upper_bound[-1]
                lower_bound = deepcopy(tmp_lower)
                upper_bound = deepcopy(tmp_upper)
                if k == 1:
                    cur_neuron.concrete_algebra_upper = deepcopy(upper_bound)
                    cur_neuron.concrete_algebra_lower = deepcopy(lower_bound)
            assert (len(lower_bound) == 1)
            assert (len(upper_bound) == 1)
            cur_neuron.concrete_lower = lower_bound[0]
            cur_neuron.concrete_upper = upper_bound[0]
            if (cur_neuron.concrete_highest_lower == None) or (
                    cur_neuron.concrete_highest_lower < cur_neuron.concrete_lower):
                cur_neuron.concrete_highest_lower = cur_neuron.concrete_lower
            if (cur_neuron.concrete_lowest_upper == None) or (
                    cur_neuron.concrete_lowest_upper > cur_neuron.concrete_upper):
                cur_neuron.concrete_lowest_upper = cur_neuron.concrete_upper

        self.abs_mode_changed = 0
        self.abs_mode_changed_min = 0
        for i in range(len(self.layers) - 1):
            gp_layer_count = 0
            pre_layer = self.layers[i]
            cur_layer = self.layers[i + 1]
            pre_neuron_list = pre_layer.neurons
            cur_neuron_list = cur_layer.neurons

            if cur_layer.layer_type == DP_DNN_layer.AFFINE_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    cur_neuron.algebra_lower = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_upper = np.append(cur_neuron.weight, [cur_neuron.bias])
                    pre(cur_neuron, i)

                    cur_neuron.concrete_lower_noClip = cur_neuron.concrete_lower
                    cur_neuron.concrete_upper_noClip = cur_neuron.concrete_upper
                    cur_neuron.actMode = 0  # affine mode
                gp_layer_count = gp_layer_count + 1

            elif cur_layer.layer_type == DP_DNN_layer.RELU_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    pre_neuron = pre_neuron_list[j]
                    cur_neuron.concrete_lower_noClip = pre_neuron.concrete_lower_noClip
                    cur_neuron.concrete_upper_noClip = pre_neuron.concrete_upper_noClip
                    if pre_neuron.concrete_highest_lower >= 0 or cur_neuron.certain_flag == 1:
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 1
                        cur_neuron.algebra_upper[j] = 1
                        cur_neuron.concrete_algebra_lower = deepcopy(pre_neuron.concrete_algebra_lower)
                        cur_neuron.concrete_algebra_upper = deepcopy(pre_neuron.concrete_algebra_upper)
                        cur_neuron.concrete_lower = pre_neuron.concrete_lower
                        cur_neuron.concrete_upper = pre_neuron.concrete_upper
                        cur_neuron.concrete_highest_lower = pre_neuron.concrete_highest_lower
                        cur_neuron.concrete_lowest_upper = pre_neuron.concrete_lowest_upper
                        cur_neuron.certain_flag = 1
                        cur_neuron.actMode = 1
                    elif pre_neuron.concrete_lowest_upper < 0 or cur_neuron.certain_flag == 2:
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.concrete_algebra_lower = np.zeros(self.inputSize + 1)
                        cur_neuron.concrete_algebra_upper = np.zeros(self.inputSize + 1)
                        cur_neuron.concrete_lower = 0
                        cur_neuron.concrete_upper = 0
                        cur_neuron.concrete_highest_lower = 0
                        cur_neuron.concrete_lowest_upper = 0
                        cur_neuron.certain_flag = 2
                        cur_neuron.actMode = 2
                    elif pre_neuron.concrete_highest_lower + pre_neuron.concrete_lowest_upper <= 0:
                        if (cur_neuron.prev_abs_mode != None) and (cur_neuron.prev_abs_mode != 0):
                            self.abs_mode_changed += 1
                        cur_neuron.prev_abs_mode = 0

                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        aux = pre_neuron.concrete_lowest_upper / (
                                pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper[j] = aux
                        cur_neuron.algebra_upper[-1] = -aux * pre_neuron.concrete_highest_lower
                        pre(cur_neuron, i)
                        cur_neuron.actMode = 3
                    else:
                        if (cur_neuron.prev_abs_mode != None) and (cur_neuron.prev_abs_mode != 1):
                            self.abs_mode_changed += 1
                        cur_neuron.prev_abs_mode = 1

                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 1
                        aux = pre_neuron.concrete_lowest_upper / (
                                pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper[j] = aux
                        cur_neuron.algebra_upper[-1] = -aux * pre_neuron.concrete_highest_lower
                        pre(cur_neuron, i)
                        cur_neuron.actMode = 4

    def load_quantized_dnn(self, deepModel, frac_bit_LL, int_bit_LL):
        layersize = []
        self.layers = []

        layersize.append(deepModel._input_shape[-1])

        new_in_layer = DP_DNN_layer()
        new_in_layer.layer_type = DP_DNN_layer.INPUT_LAYER
        new_in_layer.size = layersize[-1]
        new_in_layer.neurons = []
        for i in range(layersize[-1]):
            new_neuron = DP_DNN_neuron()
            new_in_layer.neurons.append(new_neuron)
        self.layers.append(new_in_layer)

        numDensLayers = len(deepModel.dense_layers)

        for i, l in enumerate(deepModel.dense_layers):

            # get frac_bit and int_bit and all_bit
            frac_bit = frac_bit_LL[i]
            int_bit = int_bit_LL[i]
            all_bit = frac_bit + int_bit

            tf_layer = deepModel.dense_layers[i]
            w, b = tf_layer.get_weights()
            w = w.T
            layersize.append(l.units)
            if (i < numDensLayers - 1):
                new_hidden_layer = DP_DNN_layer()
                new_hidden_layer.layer_type = DP_DNN_layer.AFFINE_LAYER
                new_hidden_layer.size = layersize[-1]
                new_hidden_layer.neurons = []
                for k in range(layersize[-1]):
                    new_hidden_neuron = DP_DNN_neuron()
                    new_hidden_neuron.weight = quantize_int(w[k], all_bit, frac_bit) / (2 ** frac_bit)
                    new_hidden_neuron.bias = quantize_int(b[k], all_bit, frac_bit) / (2 ** frac_bit)
                    new_hidden_layer.neurons.append(new_hidden_neuron)

                self.layers.append(new_hidden_layer)

                new_hidden_layer = DP_DNN_layer()
                new_hidden_layer.layer_type = DP_DNN_layer.RELU_LAYER
                new_hidden_layer.size = layersize[-1]
                new_hidden_layer.neurons = []
                for k in range(layersize[-1]):
                    new_hidden_neuron = DP_DNN_neuron()
                    new_hidden_layer.neurons.append(new_hidden_neuron)
                self.layers.append(new_hidden_layer)

            else:
                new_out_layer = DP_DNN_layer()
                new_out_layer.layer_type = new_out_layer.AFFINE_LAYER
                new_out_layer.size = layersize[-1]
                new_out_layer.neurons = []
                for k in range(layersize[-1]):
                    new_out_neuron = DP_DNN_neuron()
                    new_out_neuron.weight = quantize_int(w[k], all_bit, frac_bit) / (2 ** frac_bit)
                    new_out_neuron.bias = quantize_int(b[k], all_bit, frac_bit) / (2 ** frac_bit)
                    new_out_layer.neurons.append(new_out_neuron)
                self.layers.append(new_out_layer)

        self.layerSizes = layersize
        self.inputSize = layersize[0]
        self.outputSize = layersize[-1]
        self.numLayers = len(layersize) - 1

    def load_dnn(self, quantized_model):

        layersize = []
        self.layers = []

        layersize.append(quantized_model._input_shape[-1])

        new_in_layer = DP_DNN_layer()
        new_in_layer.layer_type = DP_DNN_layer.INPUT_LAYER
        new_in_layer.size = layersize[-1]
        new_in_layer.neurons = []
        for i in range(layersize[-1]):
            new_neuron = DP_DNN_neuron()
            new_in_layer.neurons.append(new_neuron)
        self.layers.append(new_in_layer)

        numDensLayers = len(quantized_model.dense_layers)
        for i, l in enumerate(quantized_model.dense_layers):
            tf_layer = quantized_model.dense_layers[i]
            w, b = tf_layer.get_weights()
            w = w.T
            layersize.append(l.units)
            if (i < numDensLayers - 1):
                new_hidden_layer = DP_DNN_layer()
                new_hidden_layer.layer_type = DP_DNN_layer.AFFINE_LAYER
                new_hidden_layer.size = layersize[-1]
                new_hidden_layer.neurons = []
                for k in range(layersize[-1]):
                    new_hidden_neuron = DP_DNN_neuron()
                    new_hidden_neuron.weight = w[k]
                    new_hidden_neuron.bias = b[k]
                    new_hidden_layer.neurons.append(new_hidden_neuron)
                self.layers.append(new_hidden_layer)

                new_hidden_layer = DP_DNN_layer()
                new_hidden_layer.layer_type = DP_DNN_layer.RELU_LAYER
                new_hidden_layer.size = layersize[-1]
                new_hidden_layer.neurons = []
                for k in range(layersize[-1]):
                    new_hidden_neuron = DP_DNN_neuron()
                    new_hidden_layer.neurons.append(new_hidden_neuron)
                self.layers.append(new_hidden_layer)

            else:
                new_out_layer = DP_DNN_layer()
                new_out_layer.layer_type = new_out_layer.AFFINE_LAYER
                new_out_layer.size = layersize[-1]
                new_out_layer.neurons = []
                for k in range(layersize[-1]):
                    new_out_neuron = DP_DNN_neuron()
                    new_out_neuron.weight = w[k]
                    new_out_neuron.bias = b[k]
                    new_out_layer.neurons.append(new_out_neuron)
                self.layers.append(new_out_layer)

        self.layerSizes = layersize
        self.inputSize = layersize[0]
        self.outputSize = layersize[-1]
        self.numLayers = len(layersize) - 1