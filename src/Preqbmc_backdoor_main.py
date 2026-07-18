import argparse
from utils.deep_models import *
from src.utils.preqbmc_encoding_backdoor import *
from src.utils.preqbmc_utils import *
from gurobipy import GRB
import gurobipy as gp
import random

bigM = GRB.MAXINT

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--arch", default="1blk_100")
parser.add_argument("--loc_row", type=int, default=1)  #
parser.add_argument("--loc_col", type=int, default=1)  #
parser.add_argument("--stamp_size", type=int, default=3)  #
parser.add_argument("--bit_lb", type=int, default=1)
parser.add_argument("--bit_ub", type=int, default=16)  #
parser.add_argument("--outputPath", default="")
parser.add_argument("--targetCls", type=int, default=0)
parser.add_argument("--originalCls", type=int, default=10)
parser.add_argument("--ifRelax", type=int, default=0)

parser.add_argument("--K", type=int, default=1)
parser.add_argument("--theta", type=float, default=0.9)
parser.add_argument("--delta", type=float, default=0.05)
parser.add_argument("--scaleFactor_threshold", type=float, default=0.01)

# Preimage Computing Mode: MILP-based ('milp'), Abstr-based ('abstr')
parser.add_argument("--preimg_mode", default="milp")

args = parser.parse_args()

if args.targetCls == args.originalCls:
    print("Not valid parameters: same target class and ground truth class!")
    exit(0)

if args.dataset == "fashion-mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
elif args.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    raise ValueError("Unknown dataset '{}'".format(args.dataset))

y_train = y_train.flatten()
y_test = y_test.flatten()

x_train = x_train.reshape([-1, 28 * 28]).astype(np.float32)
x_test = x_test.reshape([-1, 28 * 28]).astype(np.float32)

archMnist = args.arch.split('_')
numBlk = archMnist[0][:-3]
arch = [784]
blkset = list(map(int, archMnist[1:]))
blkset.append(10)
arch += blkset

assert int(numBlk) == len(blkset) - 1

model = DeepModel(
    blkset,
    last_layer_signed=True,
)

weight_path = "benchmark/{}/{}_{}_weight.h5".format(args.dataset, args.dataset, args.arch)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.build((None, 28 * 28))

model.load_weights(weight_path)  # input: 0~255

backdoorIndices = []
for stamp_index_row in range(args.stamp_size):
    for stamp_index_col in range(args.stamp_size):
        backdoorIndices.append(28 * (args.loc_row - 1 + stamp_index_row) + args.loc_col - 1 + stamp_index_col)


fileName = args.outputPath + "/Backdoor_row-" + str(args.loc_row) + "_col-" + str(args.loc_col) + "_K_" + str(
        args.K) + "_targetCls_" + str(args.targetCls) + "_originalCls_" + str(args.originalCls) + "_stampSize_" + str(
        args.stamp_size) + ".txt"

backDoor_color = 1

n = 0
z = 0

start_time = time.time()
dnn_gurobi_model = GPEncoding_multi(arch, model, args)
alpha = args.delta
beta = args.delta
success_X_n = []

while True:
    real_sample_input, real_sample_ID, real_sample_label = backdoor_random(x_test, y_test, args.K, args.originalCls,
                                                                           args.targetCls)
    x_low_set = []
    x_high_set = []
    x_set = []
    for index, x_input in enumerate(real_sample_input):
        predict_index = model.predict(np.expand_dims(x_input, 0))[0]
        predict_clsx = np.argmax(model.predict(np.expand_dims(x_input, 0))[0])
        print("The predict_clsx is: ", predict_clsx)
        x_new_low = x_input / 255 + 0
        x_new_high = x_input / 255 + 0
        x_set.append(x_input / 255)
        for backIndex in backdoorIndices:
            x_new_low[backIndex] = 0
            x_new_high[backIndex] = backDoor_color

        x_low_set.append(x_new_low)
        x_high_set.append(x_new_high)

    dnn_gurobi_model.update_scale_factors()
    forward_DNN_multi(x_set, dnn_gurobi_model)  # real_value：append
    dnn_gurobi_model.set_backward_input_bounds(x_low_set, x_high_set)  # append input_gp_vars
    dnn_gurobi_model.add_deepPolyNets_DNN_set(args.K)  # initialize K deepPoly models

    # append new inputs' bounds, and update newly-added deepPoly bounds
    dnn_gurobi_model.assert_input_box_multi(x_low_set, x_high_set, n)

    dnn_gurobi_model.symbolic_propagate_multi(n)

    ifSuccess, maxIndex = dnn_gurobi_model.backward_preimage_computation_multi(n)

    if ifSuccess:
        z = z + 1
        success_X_n.append(n)

    n = n + 1

    # compute left and right one for comparision
    p0 = 1 - math.pow(args.theta, args.K) + args.delta
    p1 = 1 - math.pow(args.theta, args.K) - args.delta

    left = math.pow(p1 / p0, z) * math.pow((1 - p1) / (1 - p0), n - z)
    right_H0 = beta / (1 - alpha)
    right_H1 = (1 - beta) / alpha

    if left <= right_H0:
        backward_end_time = time.time()
        ifSucc, qu_list, qu_frac_list, qu_int_list = dnn_gurobi_model.forward_quantization_with_esbmc()

        forward_end_time = time.time()

        print("Backward Time is: ", backward_end_time - start_time)
        print("Forward time is: ", forward_end_time - backward_end_time)
        dnn_gurobi_model._stats["backward_time"] = backward_end_time - start_time
        dnn_gurobi_model._stats["forward_time"] = forward_end_time - backward_end_time
        dnn_gurobi_model._stats["total_time"] = dnn_gurobi_model._stats["backward_time"] + dnn_gurobi_model._stats[
            "forward_time"]
        print("\n ******************** Total running time is: ", dnn_gurobi_model._stats["total_time"],
              " ********************")

        if ifSucc:
            fo = open(fileName, "w")
            fo.write("Solving Result: True\n")
            fo.write("We found a quantization strategy to hold the backdoor property.\n")
            fo.write("The all quantization bit sizes for each layer are:" + str(qu_list) + "\n")
            fo.write("The frac quantization bit sizes for each layer are:" + str(qu_frac_list) + "\n")
            fo.write("The int quantization bit sizes for each layer are:" + str(qu_int_list) + "\n")
            fo.write("Backward Time: " + str(dnn_gurobi_model._stats["backward_time"]) + "\n")
            fo.write("Forward Time: " + str(dnn_gurobi_model._stats["forward_time"]) + "\n")
            fo.write("Total Time: " + str(dnn_gurobi_model._stats["total_time"]) + "\n")
            numAllVars = dnn_gurobi_model.gp_model.getAttr("NumVars")
            numIntVars = dnn_gurobi_model.gp_model.getAttr("NumIntVars")
            numBinVars = dnn_gurobi_model.gp_model.getAttr("NumBinVars")
            numConstrs = dnn_gurobi_model.gp_model.getAttr("NumConstrs")
            #
            fo.write("The num of vars: " + str(numAllVars) + "\n")
            fo.write("The num of numIntVars: " + str(numIntVars) + "\n")
            fo.write("The num of numBinVars: " + str(numBinVars) + "\n")
            fo.write("The num of Constraints: " + str(numConstrs) + "\n")

            # write accuracy
            loss_DNN, accu_DNN = model.evaluate(x_test, y_test)

            for i in range(len(blkset)):
                Q = qu_list[i]
                I_W = qu_int_list[i]
                F_W = qu_frac_list[i]
                F_B = F_W
                assert Q == I_W + F_W

                paras = model.layers[i].get_weights()  # list of two array
                new_weight = []
                for j in range(len(paras[0])):
                    weight_j = paras[0][j].tolist()
                    weight_j = list(map(lambda a: real_round(a * (2 ** F_W)) / (2 ** F_W), weight_j))
                    new_weight.append(weight_j)

                new_weight = np.asarray(new_weight)

                bias = paras[1].tolist()
                bias = list(map(lambda a: real_round(a * (2 ** F_B)) / (2 ** F_B), bias))
                bias = np.asarray(bias)
                model.layers[i].set_weights([new_weight, bias])

            loss_QNN, accu_QNN = model.evaluate(x_test, y_test)

            outputMessage_QNN = "\nThe accuracy of QNN we got is: {}".format(accu_QNN)
            outputMessage_DNN = "\nThe accuracy of DNN is: {}".format(accu_DNN)
            fo.write(outputMessage_DNN)
            fo.write(outputMessage_QNN)

            fo.close()
            break
        else:
            fo = open(fileName, "w")
            fo.write("Solving Result: False\n")
            fo.write("We cannot found a quantization strategy to hold the backdoor property. The result is H0\n")
            fo.write("Backward Time: " + str(dnn_gurobi_model._stats["backward_time"]) + "\n")
            fo.write("Forward Time: " + str(dnn_gurobi_model._stats["forward_time"]) + "\n")
            fo.write("Total Time: " + str(dnn_gurobi_model._stats["total_time"]) + "\n")
            numAllVars = dnn_gurobi_model.gp_model.getAttr("NumVars")
            numIntVars = dnn_gurobi_model.gp_model.getAttr("NumIntVars")
            numBinVars = dnn_gurobi_model.gp_model.getAttr("NumBinVars")
            numConstrs = dnn_gurobi_model.gp_model.getAttr("NumConstrs")
            #
            fo.write("The num of vars: " + str(numAllVars) + "\n")
            fo.write("The num of numIntVars: " + str(numIntVars) + "\n")
            fo.write("The num of numBinVars: " + str(numBinVars) + "\n")
            fo.write("The num of Constraints: " + str(numConstrs) + "\n")
            fo.close()
            break

    elif left >= right_H1:
        fo = open(fileName, "w")
        fo.write("Solving Result: False\n")
        fo.write("We cannot found a quantization strategy to hold the backdoor property. The result is H1\n")
        fo.write("Backward Time: " + str(dnn_gurobi_model._stats["backward_time"]) + "\n")
        fo.write("Forward Time: " + str(dnn_gurobi_model._stats["forward_time"]) + "\n")
        fo.write("Total Time: " + str(dnn_gurobi_model._stats["total_time"]) + "\n")
        numAllVars = dnn_gurobi_model.gp_model.getAttr("NumVars")
        numIntVars = dnn_gurobi_model.gp_model.getAttr("NumIntVars")
        numBinVars = dnn_gurobi_model.gp_model.getAttr("NumBinVars")
        numConstrs = dnn_gurobi_model.gp_model.getAttr("NumConstrs")
        #
        fo.write("The num of vars: " + str(numAllVars) + "\n")
        fo.write("The num of numIntVars: " + str(numIntVars) + "\n")
        fo.write("The num of numBinVars: " + str(numBinVars) + "\n")
        fo.write("The num of Constraints: " + str(numConstrs) + "\n")
        fo.close()
        break