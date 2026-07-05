from symbolic_pp.DeepPoly_quadapter import *
from utils.quadapter_utils import *

import math
from gurobipy import GRB
import gurobipy as gp
import time
import numpy as np
from gurobipy import quicksum
from numpy import tile


class LayerEncoding:
    def __init__(
            self,
            gp_model,
            layer_index,
            layer_size,
            layer_paras,
            multiNum,
            bit_lb,
            bit_ub,
            if_hid,
            preimg_mode,
    ):
        self.layer_index = layer_index
        self.layer_size = layer_size
        self.layer_paras = layer_paras  # weight+bias
        self.bit_lb = bit_lb
        self.bit_ub = bit_ub
        self.frac_bit = None
        self.grad = None
        self.realVal_set = []
        self.actMode_set = []
        self.if_hid = if_hid
        self.preimg_mode = preimg_mode
        if if_hid:
            neuron_lb_after = 0
        else:
            neuron_lb_after = -GRB.MAXINT

        neuron_lb_before = -GRB.MAXINT

        self.lb_set = []
        self.ub_set = []

        self.clipped_lb_set = []
        self.clipped_ub_set = []

        self.qu_lb_set = []
        self.qu_ub_set = []

        self.qu_clipped_lb_set = []
        self.qu_clipped_ub_set = []

        if layer_index > 0:
            self.max_weight = np.round(max(np.max(layer_paras[0]), np.max(layer_paras[1])))
            self.min_weight = np.round(min(np.min(layer_paras[0]), np.min(layer_paras[1])))
            self.max_int = max(abs(self.max_weight), abs(self.min_weight))
            if self.max_int == 0:
                self.int_bit = 1
            elif self.max_int == 1:
                self.int_bit = 2
            else:
                self.int_bit = int(np.ceil(math.log(self.max_int, 2)) + 1)
        else:
            self.int_bit = None

        if self.preimg_mode == "milp":
            self.gp_vars_before_set = []
            self.gp_vars_after_set = []
        else:
            self.gp_vars_lb_before_set = []
            self.gp_vars_ub_before_set = []

        self.relaxed_lb_expression_set = []
        self.relaxed_ub_expression_set = []

        self.alpha_set = []

        self.alpha_before_set = []
        self.alpha_after_set = []

        self.beta_set = []

        self.beta_before_set = []
        self.beta_after_set = []

        self.relaxed_lb_set = []  # relaxed lb, lb before relu function
        self.relaxed_ub_set = []  # relaxed ub, ub before relu function

        print("The quantization bit size for integer parts of Layer ", self.layer_index, " is: ", self.int_bit)

    # TODO: add bound constraint for gurobi
    def append_input_bounds_multi(self, low, high):
        self.lb_set.append(low)
        self.ub_set.append(high)

        # unsigned input
        self.clipped_lb_set.append(low)
        self.clipped_ub_set.append(high)

        self.qu_clipped_lb_set.append(low)
        self.qu_clipped_ub_set.append(high)

        self.qu_lb_set.append(low)
        self.qu_ub_set.append(high)

    # def set_grad(self, grad):
    #     self.grad = grad
    #     print("We set the gradient for the layer ", self.layer_index)

    def set_realVal_multi(self, realVal):
        self.realVal_set.append(realVal)
        # print("We set the real output values for the layer ", self.layer_index)

    def update_scale_factors(self, gp_model, preimg_mode, multiNum):
        layer_size = self.layer_size

        actMode = np.zeros(layer_size, dtype=np.int32)
        actMode_set = tile(actMode, (multiNum, 1))
        self.actMode_set.extend(actMode_set)

        lb = np.zeros(layer_size, dtype=np.float32)
        lb_set = tile(lb, (multiNum, 1))
        self.lb_set.extend(lb_set)

        ub = np.zeros(layer_size, dtype=np.float32)
        ub_set = tile(ub, (multiNum, 1))
        self.ub_set.extend(ub_set)

        clipped_lb = np.zeros(layer_size, dtype=np.float32)
        clipped_lb_set = tile(clipped_lb, (multiNum, 1))
        self.clipped_lb_set.extend(clipped_lb_set)

        clipped_ub = np.zeros(layer_size, dtype=np.float32)
        clipped_ub_set = tile(clipped_ub, (multiNum, 1))
        self.clipped_ub_set.extend(clipped_ub_set)

        relaxed_lb = np.zeros(layer_size, dtype=np.float32)  # lower bound of preimage ==> relaxed region
        relaxed_lb_set = tile(relaxed_lb, (multiNum, 1))  # lower bound of preimage ==> relaxed region
        self.relaxed_lb_set.extend(relaxed_lb_set)

        relaxed_ub = np.zeros(layer_size, dtype=np.float32)  # upper bound of preimage ==> relaxed region
        relaxed_ub_set = tile(relaxed_ub, (multiNum, 1))  # upper bound of preimage ==> relaxed region
        self.relaxed_ub_set.extend(relaxed_ub_set)

        qu_lb = np.zeros(layer_size, dtype=np.float32)  # lower bound of quantized region
        qu_lb_set = tile(qu_lb, (multiNum, 1))  # lower bound of quantized region
        self.qu_lb_set.extend(qu_lb_set)

        qu_ub = np.zeros(layer_size, dtype=np.float32)  # upper bound of quantized region
        qu_ub_set = tile(qu_ub, (multiNum, 1))  # upper bound of quantized region
        self.qu_ub_set.extend(qu_ub_set)

        qu_clipped_lb = np.zeros(layer_size, dtype=np.float32)
        qu_clipped_lb_set = tile(qu_clipped_lb, (multiNum, 1))
        self.qu_clipped_lb_set.extend(qu_clipped_lb_set)

        qu_clipped_ub = np.zeros(layer_size, dtype=np.float32)
        qu_clipped_ub_set = tile(qu_clipped_ub, (multiNum, 1))
        self.qu_clipped_ub_set.extend(qu_clipped_ub_set)

        if self.if_hid:
            neuron_lb_after = 0
        else:
            neuron_lb_after = -GRB.MAXINT

        neuron_lb_before = -GRB.MAXINT

        for sample_index in range(multiNum):
            relaxed_lb_expression = [-1000 for i in range(layer_size)]
            relaxed_ub_expression = [1000 for i in range(layer_size)]

            self.relaxed_lb_expression_set.append(relaxed_lb_expression)
            self.relaxed_ub_expression_set.append(relaxed_ub_expression)

            if preimg_mode == 'milp':
                gp_vars_before = [gp_model.addVar(lb=neuron_lb_before, vtype=GRB.CONTINUOUS) for s in
                                  range(layer_size)]  # before relu
                gp_vars_after = [gp_model.addVar(lb=0, vtype=GRB.CONTINUOUS) for s in
                                 range(layer_size)]  # before relu

                self.gp_vars_before_set.append(gp_vars_before)
                self.gp_vars_after_set.append(gp_vars_after)

                alpha = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
                beta = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]

                self.alpha_set.append(alpha)
                self.beta_set.append(beta)

            elif preimg_mode == 'abstr':
                gp_vars_lb_before = [gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for s in
                                     range(layer_size)]  # before relu
                gp_vars_ub_before = [gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for s in
                                     range(layer_size)]  # before relu

                self.gp_vars_lb_before_set.append(gp_vars_lb_before)
                self.gp_vars_ub_before_set.append(gp_vars_ub_before)

                alpha_before = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
                alpha_after = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]

                beta_before = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
                beta_after = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]

                self.alpha_before_set.append(alpha_before)
                self.alpha_after_set.append(alpha_after)

                self.beta_before_set.append(beta_before)
                self.beta_after_set.append(beta_after)
            else:
                print("Wrong option for the preimage computation mode!")
                exit(0)

            gp_model.update()


class GPEncoding_multi:
    def __init__(self, arch, model, args):
        self.gp_model = gp.Model("gp_encoding")
        # self.myFeasibilityTol = 1e-6
        self.tole = 1e-6
        self.gp_model.Params.IntFeasTol = 1e-6
        self.gp_model.Params.FeasibilityTol = self.tole
        self.gp_model.setParam(GRB.Param.Threads, 30)
        self.gp_model.setParam(GRB.Param.OutputFlag, 0)
        self.bit_lb = args.bit_lb
        self.bit_ub = args.bit_ub
        self.scaleFactor_threshold = args.scaleFactor_threshold
        self.forward_index_set = []
        self.K = args.K
        self.ifRelax = args.ifRelax
        self.scaleValueSet_set = []
        self.preimg_mode = args.preimg_mode

        self._stats = {
            "encoding_time": 0,
            "solving_time": 0,
            "backward_time": 0,
            "forward_time": 0,
            "total_time": 0,
        }

        # self.pool = Pool(thread)
        self.dense_layers = []
        self.nnparas = []
        self.outputPath = args.outputPath
        self.deep_model = model
        self.layerNum = len(model.dense_layers)
        self.targetCls = args.targetCls
        self.layers_RelaxEnc = []  # reversed-order
        self.deepPolyNets_DNN_set = []
        self.input_gp_vars_set = []
        self.maxIndexSet = []  # the maximum scale for each K sampled properties

        for i, l in enumerate(model.dense_layers):
            tf_layer = model.dense_layers[i]
            w_cont, b_cont = tf_layer.get_weights()
            paras = [w_cont.T, b_cont]
            self.nnparas.append(paras)

        ########## output layer
        self.output_layer = LayerEncoding(self.gp_model,
                                          layer_index=len(self.nnparas), layer_size=arch[-1],
                                          layer_paras=self.nnparas[-1], multiNum=self.K, bit_lb=self.bit_lb,
                                          bit_ub=self.bit_ub, if_hid=False, preimg_mode=self.preimg_mode)

        ########## hidden layer
        for layer in range(len(arch) - 2):
            self.dense_layers.append(
                LayerEncoding(self.gp_model,
                              layer_index=layer + 1, layer_size=arch[layer + 1], layer_paras=self.nnparas[layer],
                              multiNum=self.K, bit_lb=self.bit_lb, bit_ub=self.bit_ub, if_hid=True,
                              preimg_mode=self.preimg_mode)
            )

        ########## input layer
        input_size = arch[0]

        self.input_layer = LayerEncoding(self.gp_model,
                                         layer_index=0, layer_size=input_size, layer_paras=None,
                                         multiNum=self.K, bit_lb=self.bit_lb, bit_ub=self.bit_ub,
                                         if_hid=False, preimg_mode=self.preimg_mode)



    def add_deepPolyNets_DNN_set(self, K):
        for i in range(K):
            i_deepPolyNets_DNN = DP_DNN_network(True)
            i_deepPolyNets_DNN.load_dnn(self.deep_model)
            self.deepPolyNets_DNN_set.append(i_deepPolyNets_DNN)
            sizeOflen = len(self.dense_layers)
            self.scaleValueSet_set.append([0 for i in range(sizeOflen)])


    def set_backward_input_bounds(self, x_low_real_set_backward, x_high_real_set_backward):
        for sample_index in range(len(x_low_real_set_backward)):
            x_low_real = x_low_real_set_backward[sample_index]
            x_high_real = x_high_real_set_backward[sample_index]
            input_gp_vars = []
            for input_index in range(self.input_layer.layer_size):
                x_lb = x_low_real[input_index]
                x_ub = x_high_real[input_index]
                cur_var = self.gp_model.addVar(lb=x_lb, ub=x_ub, vtype=GRB.CONTINUOUS)
                input_gp_vars.append(cur_var)

            self.input_gp_vars_set.append(input_gp_vars)

    def update_scale_factors(self):
        self.output_layer.update_scale_factors(self.gp_model, self.preimg_mode, self.K)
        for dense_layer in self.dense_layers:
            dense_layer.update_scale_factors(self.gp_model, self.preimg_mode, self.K)


    def assert_input_box_multi(self, x_lb_set, x_ub_set, n):
        input_size = self.input_layer.layer_size

        for index in range(len(x_lb_set)):
            low, high = x_lb_set[index], x_ub_set[index]

            # Ensure low is a vector
            low = np.array(low, dtype=np.float32) * np.ones(input_size, dtype=np.float32)
            high = np.array(high, dtype=np.float32) * np.ones(input_size, dtype=np.float32)

            self.input_layer.append_input_bounds_multi(low, high)

            real_sample_index = n * self.K + index

            cur_deepPolyNets_DNN = self.deepPolyNets_DNN_set[real_sample_index]

            cur_deepPolyNets_DNN.property_region = 1

            for i in range(cur_deepPolyNets_DNN.layerSizes[0]):
                cur_deepPolyNets_DNN.layers[0].neurons[i].concrete_lower = low[i]
                cur_deepPolyNets_DNN.layers[0].neurons[i].concrete_upper = high[i]
                cur_deepPolyNets_DNN.property_region *= (high[i] - low[i])
                cur_deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_lower = np.array([low[i]])
                cur_deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_upper = np.array([high[i]])
                cur_deepPolyNets_DNN.layers[0].neurons[i].algebra_lower = np.array([low[i]])
                cur_deepPolyNets_DNN.layers[0].neurons[i].algebra_upper = np.array([high[i]])



    def symbolic_propagate_multi(self, n):

        for sample_index in range(self.K):
            real_sample_index = n * self.K + sample_index
            cur_deepPolyNets_DNN = self.deepPolyNets_DNN_set[real_sample_index]

            cur_deepPolyNets_DNN.deeppoly()

            for i, l in enumerate(self.dense_layers):

                for out_index in range(l.layer_size):
                    lb = cur_deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].concrete_lower_noClip
                    ub = cur_deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].concrete_upper_noClip

                    lb_clipped = cur_deepPolyNets_DNN.layers[2 * (i + 1)].neurons[
                        out_index].concrete_lower
                    ub_clipped = cur_deepPolyNets_DNN.layers[2 * (i + 1)].neurons[
                        out_index].concrete_upper

                    l.lb_set[real_sample_index][out_index] = lb
                    l.ub_set[real_sample_index][out_index] = ub

                    l.clipped_lb_set[real_sample_index][out_index] = max(lb_clipped, 0)
                    l.clipped_ub_set[real_sample_index][out_index] = max(ub_clipped, 0)

                    if self.preimg_mode == 'abstr':
                        act_mode = self.deepPolyNets_DNN_set[real_sample_index].layers[2 * (i + 1)].neurons[
                            out_index].actMode
                        l.actMode_set[real_sample_index][out_index] = act_mode

                    # print("lower and upper bounds of out_index ", out_index, ": [", lb, ',', ub, ']')

            for out_index in range(self.output_layer.layer_size):
                lb = cur_deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_lower_noClip
                ub = cur_deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_upper_noClip
                self.output_layer.lb_set[real_sample_index][out_index] = lb
                self.output_layer.ub_set[real_sample_index][out_index] = ub
                self.output_layer.actMode_set[real_sample_index][out_index] = -1  # output acMode=-1

                # print("lower and upper bounds of out_index ", out_index, ": [", lb, ',', ub, ']')


    def backward_preimage_computation_multi(self, n):
        ifSuccess = False
        scaleSet = []

        for sample_index in range(self.K):
            print("\n################################# Now we are doing the backward relaxation for the ", sample_index,
                  "-th input sample of the ", n, "-th Backward #################################")

            real_sample_index = sample_index + n * self.K

            if self.preimg_mode == 'milp':
                fst_scale = self.backward_MILP_based_multi_single(real_sample_index)
                scaleSet.append(fst_scale)
            else:
                fst_scale = self.backward_Abstr_based_with_var_multi_single(real_sample_index)
                scaleSet.append(fst_scale)

        maxScale = max(scaleSet)
        maxIndex = np.argmax(scaleSet)
        if maxScale >= self.scaleFactor_threshold:
            ifSuccess = True
            self.maxIndexSet.append(maxIndex)
        else:
            maxIndex = -1
            self.maxIndexSet.append(-1)
        return ifSuccess, maxIndex


    ## backward_poly_based_with_var_multi_single
    def backward_Abstr_based_with_var_multi_single(self, real_sample_index):
        cur_layer = self.output_layer
        in_layer_index = len(self.dense_layers)
        fst_scale_factor = 0

        for in_layer in reversed(self.dense_layers):
            enc_start_time = time.time()
            in_layer_index -= 1
            relaxScale_LL = []

            var_ll = []
            prop_cstr_ll = []
            model_cstr_ll = []
            w = cur_layer.layer_paras[0]
            b = cur_layer.layer_paras[1]

            relaxScale = self.gp_model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS)
            relaxScale_LL.append(relaxScale)

            # define relaxed_region for gp_vars_after (gp_vars_before is for concrete value comparison)
            for in_index in range(in_layer.layer_size):

                neuron_val = in_layer.realVal_set[real_sample_index][in_index]
                actMode = in_layer.actMode_set[real_sample_index][in_index]

                neuron_lb = in_layer.lb_set[real_sample_index][in_index]
                neuron_ub = in_layer.ub_set[real_sample_index][in_index]

                if actMode == 1:
                    alpha_K = neuron_val - neuron_lb
                    beta_K = neuron_ub - neuron_val

                    model_cstr_ll.append(
                        self.gp_model.addConstr(
                            in_layer.alpha_before_set[real_sample_index][in_index] == (alpha_K * relaxScale)))
                    model_cstr_ll.append(
                        self.gp_model.addConstr(
                            in_layer.beta_after_set[real_sample_index][in_index] == (beta_K * relaxScale)))

                    model_cstr_ll.append(
                        self.gp_model.addGenConstrMin(in_layer.alpha_after_set[real_sample_index][in_index],
                                                      [in_layer.alpha_before_set[real_sample_index][in_index],
                                                       in_layer.lb_set[real_sample_index][in_index]]))
                elif actMode == 2:
                    continue
                else:  # actMode == 3 or 4:
                    model_cstr_ll.append(
                        self.gp_model.addConstr(
                            in_layer.alpha_after_set[real_sample_index][in_index] == (-neuron_lb * relaxScale)))
                    model_cstr_ll.append(
                        self.gp_model.addConstr(
                            in_layer.beta_after_set[real_sample_index][in_index] == (neuron_ub * relaxScale)))

            self.gp_model.update()

            # compute relaxed accumulated bounds instead of exactly encoding cur_layer's computation
            for out_index in range(cur_layer.layer_size):
                weights = w[out_index]
                tmp_add_lower = 0
                tmp_add_upper = 0

                # get new added biases
                for in_index in range(in_layer.layer_size):
                    actMode = in_layer.actMode_set[real_sample_index][in_index]
                    if actMode == 1:
                        if weights[in_index] >= 0:
                            tmp_add_lower -= weights[in_index] * in_layer.alpha_after_set[real_sample_index][in_index]
                            tmp_add_upper += weights[in_index] * in_layer.beta_after_set[real_sample_index][in_index]
                        else:
                            tmp_add_lower += weights[in_index] * in_layer.beta_after_set[real_sample_index][in_index]
                            tmp_add_upper -= weights[in_index] * in_layer.alpha_after_set[real_sample_index][in_index]
                    elif actMode == 2:
                        continue

                    elif actMode == 3:
                        # update bounds
                        K = in_layer.ub_set[real_sample_index][in_index] / (
                                in_layer.ub_set[real_sample_index][in_index] - in_layer.lb_set[real_sample_index][
                            in_index])
                        if weights[in_index] >= 0:
                            tmp_add_upper += weights[in_index] * K * (
                                    in_layer.beta_after_set[real_sample_index][in_index] +
                                    in_layer.alpha_after_set[real_sample_index][in_index])
                        else:
                            tmp_add_lower += weights[in_index] * K * (
                                    in_layer.beta_after_set[real_sample_index][in_index] +
                                    in_layer.alpha_after_set[real_sample_index][in_index])

                    else:  # actMode == 4
                        K = in_layer.ub_set[real_sample_index][in_index] / (
                                in_layer.ub_set[real_sample_index][in_index] - in_layer.lb_set[real_sample_index][
                            in_index])
                        if weights[in_index] >= 0:
                            tmp_add_lower -= weights[in_index] * in_layer.alpha_after_set[real_sample_index][in_index]
                            tmp_add_upper += weights[in_index] * K * (
                                    in_layer.beta_after_set[real_sample_index][in_index] +
                                    in_layer.alpha_after_set[real_sample_index][in_index])
                        else:
                            tmp_add_lower += weights[in_index] * K * (
                                    in_layer.beta_after_set[real_sample_index][in_index] +
                                    in_layer.alpha_after_set[real_sample_index][in_index])
                            tmp_add_upper -= weights[in_index] * in_layer.alpha_after_set[real_sample_index][in_index]

                model_cstr_ll.append(self.gp_model.addConstr(
                    (tmp_add_lower + cur_layer.lb_set[real_sample_index][out_index]) ==
                    cur_layer.gp_vars_lb_before_set[real_sample_index][out_index]))
                model_cstr_ll.append(self.gp_model.addConstr(
                    (tmp_add_upper + cur_layer.ub_set[real_sample_index][out_index]) ==
                    cur_layer.gp_vars_ub_before_set[real_sample_index][out_index]))

                self.gp_model.update()

            enc_finish_time = time.time()
            model_encoding_time = enc_finish_time - enc_start_time

            prop_start_time = time.time()

            # encoding cur_layer's property, computing concrete bounds
            if cur_layer.layer_index == (len(self.dense_layers) + 1):
                sumOfK = 0
                for var_index, var in enumerate(cur_layer.gp_vars_lb_before_set[real_sample_index]):
                    if var_index == self.targetCls:
                        continue
                    k_i_lb = self.gp_model.addVar(vtype=GRB.BINARY)
                    var_ll.append(k_i_lb)

                    prop_cstr_ll.append(self.gp_model.addConstr(
                        var >= cur_layer.gp_vars_lb_before_set[real_sample_index][self.targetCls] + 1000 * (
                                k_i_lb - 1)))

                    prop_cstr_ll.append(self.gp_model.addConstr(
                        var <= cur_layer.gp_vars_lb_before_set[real_sample_index][self.targetCls] + 1000 * k_i_lb))
                    sumOfK = sumOfK + k_i_lb

                prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= 1))

            else:
                for var_index, var in enumerate(cur_layer.gp_vars_lb_before_set[real_sample_index]):
                    if cur_layer.actMode_set[real_sample_index][var_index] == 1:
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            cur_layer.gp_vars_ub_before_set[real_sample_index][var_index] <=
                            cur_layer.relaxed_ub_set[real_sample_index][var_index]))
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            cur_layer.gp_vars_lb_before_set[real_sample_index][var_index] >=
                            cur_layer.relaxed_lb_set[real_sample_index][var_index]))
                    elif cur_layer.actMode_set[real_sample_index][var_index] == 2:
                        assert cur_layer.relaxed_ub_set[real_sample_index][var_index] == 0
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            cur_layer.gp_vars_ub_before_set[real_sample_index][var_index] <= 0))  # relaxed_ub>=0
                    else:
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            cur_layer.gp_vars_ub_before_set[real_sample_index][var_index] <=
                            cur_layer.relaxed_ub_set[real_sample_index][var_index]))  # relaxed_ub>=0
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            cur_layer.gp_vars_lb_before_set[real_sample_index][var_index] >=
                            cur_layer.relaxed_lb_set[real_sample_index][var_index]))

            self.gp_model.update()

            self.gp_model.setObjective(relaxScale, GRB.MAXIMIZE)
            self.gp_model.update()
            self.gp_model.setParam('DualReductions', 0)  # set this value to 0, to get a more definite result
            opt_start_time = time.time()

            self.gp_model.optimize()

            opt_finish_time = time.time()
            optimization_time = opt_finish_time - opt_start_time

            ifgpINF_OR_UNBD = self.gp_model.status == GRB.INF_OR_UNBD
            ifgpINFEASIBLE = self.gp_model.status == GRB.INFEASIBLE
            ifgpUNBOUNDED = self.gp_model.status == GRB.UNBOUNDED
            ifgpOptimize = self.gp_model.status == GRB.OPTIMAL


            if ifgpOptimize:
                scaleValue = relaxScale.X
                self.scaleValueSet_set[real_sample_index][in_layer.layer_index - 1] = scaleValue
                max_alpha = []
                max_beta = []
                for in_index in range(in_layer.layer_size):
                    alpha_after = in_layer.alpha_after_set[real_sample_index][in_index].X
                    beta_after = in_layer.beta_after_set[real_sample_index][in_index].X
                    max_alpha.append(alpha_after)
                    max_beta.append(beta_after)
                    if in_layer.ub_set[real_sample_index][in_index] <= 0:  # Case B
                        in_layer.relaxed_ub_set[real_sample_index][in_index] = 0
                        in_layer.relaxed_lb_set[real_sample_index][
                            in_index] = -GRB.MAXINT

                    else:  # Case A,C,D
                        in_layer.relaxed_ub_set[real_sample_index][in_index] = np.float32(
                            in_layer.ub_set[real_sample_index][in_index] + beta_after)
                        in_layer.relaxed_lb_set[real_sample_index][in_index] = np.float32(
                            in_layer.lb_set[real_sample_index][in_index] - alpha_after)

                    # update symbolic expression
                    # get symbolic lower bounds w.r.t. input vars
                    in_lb_algebra = deepcopy(
                        self.deepPolyNets_DNN_set[real_sample_index].layers[2 * (in_layer_index + 1) - 1].neurons[
                            in_index].concrete_algebra_lower)
                    in_ub_algebra = deepcopy(
                        self.deepPolyNets_DNN_set[real_sample_index].layers[2 * (in_layer_index + 1) - 1].neurons[
                            in_index].concrete_algebra_upper)

                    # actMode = in_layer.actMode[in_index]
                    if in_layer.ub_set[real_sample_index][in_index] > 0:  # Case A,C,D
                        relaxed_lb_bias = in_lb_algebra[-1] - alpha_after
                        relaxed_ub_bias = in_ub_algebra[-1] + beta_after
                        relaxed_symbolic_lb_expression = np.dot(in_lb_algebra[:-1],
                                                                self.input_gp_vars_set[real_sample_index])
                        relaxed_symbolic_lb_expression = relaxed_symbolic_lb_expression + relaxed_lb_bias
                        relaxed_symbolic_ub_expression = np.dot(in_ub_algebra[:-1],
                                                                self.input_gp_vars_set[real_sample_index])
                        relaxed_symbolic_ub_expression = relaxed_symbolic_ub_expression + relaxed_ub_bias

                        in_layer.relaxed_lb_expression_set[real_sample_index][in_index] = relaxed_symbolic_lb_expression
                        in_layer.relaxed_ub_expression_set[real_sample_index][in_index] = relaxed_symbolic_ub_expression
                    else:
                        in_layer.relaxed_lb_expression_set[real_sample_index][in_index] = - GRB.MAXINT
                        in_layer.relaxed_ub_expression_set[real_sample_index][in_index] = 0

                fst_scale_factor = min(max(max_alpha), max(max_beta))
                print("\n########################### scaleValue (After) for Relax layer index : ", in_layer_index,
                      " is: ",
                      fst_scale_factor, " ###########################")
                self.gp_model.remove(prop_cstr_ll)
                self.gp_model.remove(model_cstr_ll)
                self.gp_model.remove(relaxScale_LL)
                self.gp_model.remove(var_ll)
                self.gp_model.update()

            cur_layer = in_layer

        return fst_scale_factor


    #
    def backward_MILP_based_multi_single(self, real_sample_index):
        cur_layer = self.output_layer
        in_layer_index = len(self.dense_layers)
        fst_scale_factor = 0

        for in_layer in reversed(self.dense_layers):

            in_layer_index -= 1
            relaxScale_LL = []

            var_ll = []
            prop_cstr_ll = []
            model_cstr_ll = []
            w = cur_layer.layer_paras[0]
            b = cur_layer.layer_paras[1]

            relaxScale = self.gp_model.addVar(lb=0, vtype=GRB.CONTINUOUS)
            relaxScale_LL.append(relaxScale)

            for in_index in range(in_layer.layer_size):
                neuron_val = in_layer.realVal_set[real_sample_index][in_index]
                neuron_lb = in_layer.lb_set[real_sample_index][in_index]
                neuron_ub = in_layer.ub_set[real_sample_index][in_index]

                alpha_K = max(neuron_val - neuron_lb, 1e-3)
                beta_K = max(neuron_ub - neuron_val, 1e-3)

                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.alpha_set[real_sample_index][in_index] == (alpha_K * relaxScale)))
                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.beta_set[real_sample_index][in_index] == (beta_K * relaxScale)))

                # get symbolic lower bounds w.r.t. input vars
                in_lb_algebra = deepcopy(
                    self.deepPolyNets_DNN_set[real_sample_index].layers[2 * (in_layer_index + 1) - 1].neurons[
                        in_index].concrete_algebra_lower)
                in_ub_algebra = deepcopy(
                    self.deepPolyNets_DNN_set[real_sample_index].layers[2 * (in_layer_index + 1) - 1].neurons[
                        in_index].concrete_algebra_upper)

                relaxed_lb_bias = in_lb_algebra[-1] - in_layer.alpha_set[real_sample_index][in_index]
                relaxed_ub_bias = in_ub_algebra[-1] + in_layer.beta_set[real_sample_index][in_index]

                # get symbolic upper bounds w.r.t. input vars
                symbolic_lb_expression = np.dot(in_lb_algebra[:-1], self.input_gp_vars_set[real_sample_index])
                symbolic_lb_expression = symbolic_lb_expression + relaxed_lb_bias

                symbolic_ub_expression = np.dot(in_ub_algebra[:-1], self.input_gp_vars_set[real_sample_index])
                symbolic_ub_expression = symbolic_ub_expression + relaxed_ub_bias

                model_cstr_ll.append(
                    self.gp_model.addConstr(
                        in_layer.gp_vars_before_set[real_sample_index][in_index] <= symbolic_ub_expression))

                self.gp_model.update()

                model_cstr_ll.append(
                    self.gp_model.addConstr(
                        in_layer.gp_vars_before_set[real_sample_index][in_index] >= symbolic_lb_expression))
                self.gp_model.update()
                model_cstr_ll.append(
                    self.gp_model.addGenConstrMax(in_layer.gp_vars_after_set[real_sample_index][in_index],
                                                  [in_layer.gp_vars_before_set[real_sample_index][in_index], 0]))

            self.gp_model.update()

            # compute relaxed accumulated bounds for the cur_layer
            for out_index in range(cur_layer.layer_size):
                weights = w[out_index]
                bias = b[out_index]

                accumulation = np.dot(weights, in_layer.gp_vars_after_set[real_sample_index]) + bias  # + 10000

                model_cstr_ll.append(
                    self.gp_model.addConstr(cur_layer.gp_vars_before_set[real_sample_index][out_index] == accumulation))

            self.gp_model.update()

            if cur_layer.layer_index == (len(self.dense_layers) + 1):

                target_gp_value = cur_layer.gp_vars_before_set[real_sample_index][self.targetCls]

                for var_index, var in enumerate(cur_layer.gp_vars_before_set[real_sample_index]):
                    if var_index == self.targetCls:
                        continue
                    prop_cstr_ll.append(self.gp_model.addConstr(target_gp_value >= var + self.tole))

            else:
                bigM = 1000
                sumOfK = 0
                for i in range(cur_layer.layer_size):
                    k_i_lb = self.gp_model.addVar(vtype=GRB.BINARY)
                    relaxScale_LL.append(k_i_lb)

                    prop_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_before_set[real_sample_index][i] <=
                                                                cur_layer.relaxed_lb_expression_set[real_sample_index][
                                                                    i] - bigM * (k_i_lb - 1) - 2 * self.tole))
                    prop_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_before_set[real_sample_index][i] >=
                                                                cur_layer.relaxed_lb_expression_set[real_sample_index][
                                                                    i] - bigM * k_i_lb + 2 * self.tole))
                    sumOfK = sumOfK + k_i_lb

                    # k_i encodes: is not included
                    # for upper bounds
                    k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                    relaxScale_LL.append(k_i_ub)
                    prop_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_before_set[real_sample_index][i] >=
                                                                cur_layer.relaxed_ub_expression_set[real_sample_index][
                                                                    i] + bigM * (k_i_ub - 1)))
                    prop_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_before_set[real_sample_index][i] <=
                                                                cur_layer.relaxed_ub_expression_set[real_sample_index][
                                                                    i] + bigM * k_i_ub))

                    sumOfK = sumOfK + k_i_ub

                prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= 1))

            self.gp_model.update()

            self.gp_model.setObjective(relaxScale, GRB.MINIMIZE)

            self.gp_model.update()
            self.gp_model.setParam('DualReductions', 0)

            self.gp_model.optimize()

            ifgpINF_OR_UNBD = self.gp_model.status == GRB.INF_OR_UNBD
            ifgpINFEASIBLE = self.gp_model.status == GRB.INFEASIBLE
            ifgpOptimize = self.gp_model.status == GRB.OPTIMAL
            ifgpUNBOUNDED = self.gp_model.status == GRB.UNBOUNDED

            if ifgpOptimize:
                scaleValue = relaxScale.X
                print("########################### scaleValue for Relax layer index ", in_layer_index, " is: ",
                      scaleValue, " ###########################")
                fst_scale_factor = scaleValue
                self.scaleValueSet_set[real_sample_index][in_layer.layer_index - 1] = scaleValue

                for in_index in range(in_layer.layer_size):
                    alpha = in_layer.alpha_set[real_sample_index][in_index].X
                    beta = in_layer.beta_set[real_sample_index][in_index].X

                    relaxed_ub = in_layer.ub_set[real_sample_index][in_index] + beta
                    relaxed_lb = in_layer.lb_set[real_sample_index][in_index] - alpha
                    in_layer.relaxed_ub_set[real_sample_index][in_index] = relaxed_ub
                    in_layer.relaxed_lb_set[real_sample_index][in_index] = relaxed_lb

                    in_lb_algebra = \
                        self.deepPolyNets_DNN_set[real_sample_index].layers[2 * (in_layer_index + 1) - 1].neurons[
                            in_index].concrete_algebra_lower
                    in_ub_algebra = \
                        self.deepPolyNets_DNN_set[real_sample_index].layers[2 * (in_layer_index + 1) - 1].neurons[
                            in_index].concrete_algebra_upper

                    relaxed_lb_bias = in_lb_algebra[-1] - alpha
                    relaxed_ub_bias = in_ub_algebra[-1] + beta

                    # get symbolic upper bounds w.r.t. input vars
                    relaxed_symbolic_lb_expression = np.dot(in_lb_algebra[:-1],
                                                            self.input_gp_vars_set[real_sample_index])
                    relaxed_symbolic_lb_expression = relaxed_symbolic_lb_expression + relaxed_lb_bias

                    relaxed_symbolic_ub_expression = np.dot(in_ub_algebra[:-1],
                                                            self.input_gp_vars_set[real_sample_index])
                    relaxed_symbolic_ub_expression = relaxed_symbolic_ub_expression + relaxed_ub_bias

                    in_layer.relaxed_lb_expression_set[real_sample_index][in_index] = relaxed_symbolic_lb_expression
                    in_layer.relaxed_ub_expression_set[real_sample_index][in_index] = relaxed_symbolic_ub_expression

                    if relaxed_ub <= 0:
                        in_layer.relaxed_ub_expression_set[real_sample_index][in_index] = 0

            else:
                print("The backward procedure is not successful for the layer: ", in_layer.layer_index,
                      " for the sample_index ", real_sample_index)

                self.gp_model.remove(prop_cstr_ll)
                self.gp_model.remove(model_cstr_ll)
                self.gp_model.remove(relaxScale_LL)
                self.gp_model.remove(var_ll)
                self.gp_model.update()

                return -1

            self.gp_model.remove(prop_cstr_ll)
            self.gp_model.remove(model_cstr_ll)
            self.gp_model.remove(relaxScale_LL)
            self.gp_model.remove(var_ll)
            self.gp_model.update()

            cur_layer = in_layer

        return fst_scale_factor


    def forward_quantization_backdoor(self, success_X_n):

        print("\nNow we begin to do the forward quantization!")
        qu_list = []
        qu_frac_list = []
        qu_int_list = []

        nonInputLayers = self.dense_layers.copy()
        nonInputLayers.append(self.output_layer)

        in_layer_index = -1

        success_samples = []

        for n in success_X_n:
            maxIndex = self.maxIndexSet[n]
            new_sample = n * self.K + maxIndex
            success_samples.append(new_sample)

        for cur_layer in nonInputLayers:
            in_layer_index += 1

            if cur_layer.layer_index == 1:
                in_layer = self.input_layer
            else:
                in_layer = self.dense_layers[cur_layer.layer_index - 2]

            w = cur_layer.layer_paras[0]
            b = cur_layer.layer_paras[1]

            # test for all bits
            lower_bit = self.bit_lb
            upper_bit = self.bit_ub

            ifFound = False
            ifNextBit = False

            for rela_bit in range(upper_bit - lower_bit + 1):

                pre_mul_qu_lb_deepPoly_set = []
                pre_mul_qu_ub_deepPoly_set = []

                if ifFound:
                    break

                frac_bit = rela_bit + lower_bit
                int_bit = cur_layer.int_bit
                all_bit = frac_bit + int_bit

                qu_w = quantize_int(w, all_bit, frac_bit) / (2 ** frac_bit)
                qu_b = quantize_int(b, all_bit, frac_bit) / (2 ** frac_bit)

                rela_index = -1

                for real_sample_index in success_samples:

                    rela_index += 1

                    model_cstr_ll = []
                    prop_cstr_ll = []
                    var_ll = []

                    pre_mul_qu_lb_deepPoly = []
                    pre_mul_qu_ub_deepPoly = []

                    target_lb = 0
                    target_ub = 0
                    other_lbs = []
                    other_ubs = []
                    sumOfK = 0
                    numOfK = 0
                    for out_index in range(cur_layer.layer_size):
                        qu_weights = qu_w[out_index]
                        qu_bias = qu_b[out_index]

                        tmp_acc_lower = 0
                        tmp_acc_upper = 0

                        # for var_index_poly in range(self.bit_ub - self.bit_lb + 1):
                        lower_bound = np.append(qu_weights, qu_bias)  # cur_layer's paras (size of input_layer)
                        upper_bound = np.append(qu_weights, qu_bias)  # cur_layer's paras (size of input_layer)

                        # reverse, from cur_layer's affine layer to input layer
                        cur_neuron_concrete_algebra_lower = None
                        cur_neuron_concrete_algebra_upper = None

                        if in_layer_index == 0:
                            cur_neuron_concrete_algebra_lower = deepcopy(lower_bound)
                            cur_neuron_concrete_algebra_upper = deepcopy(upper_bound)

                        # reverse, from cur_layer's affine layer to input layer
                        for kk in range(2 * (in_layer_index + 1) - 1)[::-1]:
                            # size of input
                            tmp_lower = np.zeros(
                                len(self.deepPolyNets_DNN_set[real_sample_index].layers[kk].neurons[0].algebra_lower))
                            tmp_upper = np.zeros(
                                len(self.deepPolyNets_DNN_set[real_sample_index].layers[kk].neurons[0].algebra_lower))

                            assert (self.deepPolyNets_DNN_set[real_sample_index].layers[kk].size + 1 == len(
                                lower_bound))
                            assert (self.deepPolyNets_DNN_set[real_sample_index].layers[kk].size + 1 == len(
                                upper_bound))

                            for pp in range(self.deepPolyNets_DNN_set[real_sample_index].layers[kk].size):
                                if lower_bound[pp] >= 0:
                                    tmp_lower += np.float32(
                                        lower_bound[pp] *
                                        self.deepPolyNets_DNN_set[real_sample_index].layers[kk].neurons[
                                            pp].algebra_lower)
                                else:
                                    tmp_lower += np.float32(
                                        lower_bound[pp] *
                                        self.deepPolyNets_DNN_set[real_sample_index].layers[kk].neurons[
                                            pp].algebra_upper)

                                if upper_bound[pp] >= 0:
                                    tmp_upper += np.float32(
                                        upper_bound[pp] *
                                        self.deepPolyNets_DNN_set[real_sample_index].layers[kk].neurons[
                                            pp].algebra_upper)
                                else:
                                    tmp_upper += np.float32(
                                        upper_bound[pp] *
                                        self.deepPolyNets_DNN_set[real_sample_index].layers[kk].neurons[
                                            pp].algebra_lower)

                            tmp_lower[-1] += lower_bound[-1]
                            tmp_upper[-1] += upper_bound[-1]
                            lower_bound = deepcopy(tmp_lower)
                            upper_bound = deepcopy(tmp_upper)
                            #
                            if kk == 1:
                                cur_neuron_concrete_algebra_lower = deepcopy(lower_bound)
                                cur_neuron_concrete_algebra_upper = deepcopy(upper_bound)

                        assert (len(lower_bound) == 1)
                        assert (len(upper_bound) == 1)

                        cur_neuron_concrete_lower = lower_bound[0]
                        cur_neuron_concrete_upper = upper_bound[0]

                        tmp_acc_lower += cur_neuron_concrete_lower
                        tmp_acc_upper += cur_neuron_concrete_upper

                        pre_mul_qu_lb_deepPoly.append(tmp_acc_lower)
                        pre_mul_qu_ub_deepPoly.append(tmp_acc_upper)

                        # get quantized_ub_expression from backward-procedure
                        quantized_lb_expression = np.dot(cur_neuron_concrete_algebra_lower[:-1],
                                                         self.input_gp_vars_set[real_sample_index])

                        quantized_lb_expression = quantized_lb_expression + cur_neuron_concrete_algebra_lower[
                            -1]

                        quantized_ub_expression = np.dot(cur_neuron_concrete_algebra_upper[:-1],
                                                         self.input_gp_vars_set[real_sample_index])

                        quantized_ub_expression = quantized_ub_expression + cur_neuron_concrete_algebra_upper[
                            -1]

                        # add property
                        if cur_layer.layer_index == (len(self.dense_layers) + 1):
                            if out_index == self.targetCls:
                                target_lb = quantized_lb_expression
                                target_ub = quantized_ub_expression
                            else:
                                other_lbs.append(quantized_lb_expression)
                                other_ubs.append(quantized_ub_expression)
                        else:
                            k_i_lb = self.gp_model.addVar(vtype=GRB.BINARY)
                            var_ll.append(k_i_lb)

                            if cur_layer.relaxed_ub_set[real_sample_index][out_index] > 0:
                                prop_cstr_ll.append(self.gp_model.addConstr(
                                    quantized_lb_expression <= cur_layer.relaxed_lb_expression_set[real_sample_index][
                                        out_index] - 1000 * (
                                            k_i_lb - 1) - self.tole))
                                prop_cstr_ll.append(self.gp_model.addConstr(
                                    quantized_lb_expression >= cur_layer.relaxed_lb_expression_set[real_sample_index][
                                        out_index] - 1000 * k_i_lb + self.tole))
                                sumOfK = sumOfK + k_i_lb
                                numOfK += 1

                            # k_i encodes: is not included
                            # for upper bounds
                            k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                            var_ll.append(k_i_ub)
                            prop_cstr_ll.append(self.gp_model.addConstr(
                                quantized_ub_expression >= cur_layer.relaxed_ub_expression_set[real_sample_index][
                                    out_index] + 1000 * (
                                        k_i_ub - 1) + self.tole))
                            prop_cstr_ll.append(self.gp_model.addConstr(
                                quantized_ub_expression <= cur_layer.relaxed_ub_expression_set[real_sample_index][
                                    out_index] + 1000 * k_i_ub - self.tole))

                            numOfK += 1
                            sumOfK = sumOfK + k_i_ub

                    if len(other_lbs) > 0:  # output layer
                        for other_single_lb in other_lbs:
                            prop_cstr_ll.append(self.gp_model.addConstr(
                                other_single_lb <= target_ub))

                    else:
                        if self.ifRelax == 1:
                            scale = 0.25

                            # # for better performance, try this relaxation
                            # if self.scaleValueSet_set[real_sample_index][in_layer_index] <= 0.01 and in_layer_index > 0:
                            #     scale = 0.35

                            prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= int(numOfK * scale) + 1))
                        else:

                            prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= 1))

                    self.gp_model.update()
                    self.gp_model.setParam('DualReductions', 0)

                    self.gp_model.optimize()

                    ifgpUNSat = self.gp_model.status == GRB.INFEASIBLE

                    self.gp_model.remove(model_cstr_ll)
                    self.gp_model.remove(prop_cstr_ll)
                    self.gp_model.remove(var_ll)
                    self.gp_model.update()

                    if not ifgpUNSat:  # CheckSAT returns false
                        ifNextBit = True
                        print("The quantization with bit size", frac_bit, " is not appliable for sample",
                              real_sample_index)
                        break
                    else:
                        pre_mul_qu_lb_deepPoly_set.append(pre_mul_qu_lb_deepPoly)
                        pre_mul_qu_ub_deepPoly_set.append(pre_mul_qu_ub_deepPoly)

                if ifNextBit:
                    ifNextBit = False
                    continue
                else:
                    ifFound = True

                print("We find a quantization configuration [ Q , F ] for the Layer", cur_layer.layer_index,
                      "as: [", all_bit, ",", frac_bit, '].')

                cur_layer.frac_bit = frac_bit

                qu_frac_list.append(cur_layer.frac_bit)
                qu_int_list.append(cur_layer.int_bit)
                qu_list.append(all_bit)

                self.update_quantized_weights_affine_multi(in_layer, cur_layer, all_bit, frac_bit, frac_bit,
                                                           in_layer_index, success_samples)

                # if hidden layer, then update next relu's algebra
                rela_index = -1
                for real_sample_index in success_samples:
                    rela_index += 1
                    if cur_layer.layer_index < (len(self.dense_layers) + 1):
                        for out_index in range(cur_layer.layer_size):
                            lb_new = pre_mul_qu_lb_deepPoly_set[rela_index][out_index]
                            ub_new = pre_mul_qu_ub_deepPoly_set[rela_index][out_index]
                            cur_neuron = \
                                self.deepPolyNets_DNN_set[real_sample_index].layers[2 * (in_layer_index + 1)].neurons[
                                    out_index]
                            if lb_new >= 0:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_lower[out_index] = 1
                                cur_neuron.algebra_upper[out_index] = 1
                            elif ub_new <= 0:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                            elif lb_new + ub_new <= 0:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                k_new = ub_new / (ub_new - lb_new)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper[out_index] = k_new
                                cur_neuron.algebra_upper[-1] = - k_new * lb_new
                            else:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_lower[out_index] = 1
                                k_new = ub_new / (ub_new - lb_new)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper[out_index] = k_new
                                cur_neuron.algebra_upper[-1] = - k_new * lb_new
                    else:
                        self.output_layer.qu_lb_set[real_sample_index] = pre_mul_qu_lb_deepPoly_set[
                            rela_index]
                        self.output_layer.qu_ub_set[real_sample_index] = pre_mul_qu_ub_deepPoly_set[
                            rela_index]

            if not ifFound:
                print("Cannot find a quantization strategy for the cur_layer with index as: ",
                      cur_layer.layer_index)
                return False, None, None, None

        return True, qu_list, qu_frac_list, qu_int_list


    # update deepPoly model's quantized weights
    def update_quantized_weights_affine_multi(self, in_layer, out_layer, num_bit, frac_bit_weights, frac_bit_bias,
                                              in_layer_index, success_samples):
        print("This is propagate deepPoly range")
        min_fp_weight, max_fp_weight = int_get_min_max(num_bit, frac_bit_weights)
        min_fp_bias, max_fp_bias = int_get_min_max(num_bit, frac_bit_bias)
        for out_index in range(out_layer.layer_size):
            weight_row = out_layer.layer_paras[0][out_index]
            bias = out_layer.layer_paras[1][out_index]
            weight_row_int = quantize_int(np.asarray(weight_row), num_bit, frac_bit_weights)
            weight_row_fp = np.clip(weight_row_int / (2 ** frac_bit_weights),
                                    min_fp_weight, max_fp_weight)
            weight_row_fp = weight_row_int / (2 ** frac_bit_weights)
            bias_fp = quantize_int(bias, num_bit, frac_bit_bias) / (2 ** frac_bit_bias)

            # update weight and bias parameters for affine layer
            for sample_index in success_samples:
                self.deepPolyNets_DNN_set[sample_index].layers[2 * (in_layer_index + 1) - 1].neurons[
                    out_index].weight = weight_row_fp
                self.deepPolyNets_DNN_set[sample_index].layers[2 * (in_layer_index + 1) - 1].neurons[
                    out_index].bias = bias_fp

                # update algebra parameters for affine layer
                self.deepPolyNets_DNN_set[sample_index].layers[2 * (in_layer_index + 1) - 1].neurons[
                    out_index].algebra_lower = np.append(weight_row_fp, [bias_fp])
                self.deepPolyNets_DNN_set[sample_index].layers[2 * (in_layer_index + 1) - 1].neurons[
                    out_index].algebra_upper = np.append(weight_row_fp, [bias_fp])