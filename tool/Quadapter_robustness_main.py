import argparse
import os
from utils.deep_models import *
from utils.quadapter_encoding_robustness import *
from utils.quadapter_utils import *
from abstractions.onnx_export import save_keras_to_onnx

from gurobipy import GRB

bigM = GRB.MAXINT

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--arch", default="1blk_100")
parser.add_argument("--sample_id", type=int, default=0)
parser.add_argument("--bit_lb", type=int, default=1)
parser.add_argument("--bit_ub", type=int, default=16)
parser.add_argument("--eps", type=int, default=2)
parser.add_argument("--outputPath", default="")
parser.add_argument("--save_onnx", type=int, default=1)
parser.add_argument("--onnx_dir", default="")
parser.add_argument("--ifRelax", type=int, default=0)

# Preimage Computing Mode: MILP-based ('milp'), Abstr-based ('abstr'), Composed ('comp')
parser.add_argument("--preimg_mode", default="milp")

args = parser.parse_args()

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

x_input = x_test[args.sample_id]
model_out = model.predict(np.expand_dims(x_test[args.sample_id], 0))[0]
model_predict = np.argmax(model_out)
original_prediction = y_test[args.sample_id]

print("\nModel output is: ", model_out)
print("\nModel prediction is: ", model_predict)

assert model_predict == original_prediction

print("original_prediction is: ", original_prediction, '\n')

x_low, x_high = np.clip(x_input - args.eps, 0, 255), np.clip(x_input + args.eps, 0, 255)

dnn_gurobi_model = GPEncoding(arch, model, args, original_prediction, x_low / 255, x_high / 255)

res = forward_DNN(x_input / 255, dnn_gurobi_model)

start_time = time.time()

ifSucc, qu_list, qu_frac_list, qu_int_list = dnn_gurobi_model.verified_quant(
    np.float32(x_low / 255),
    np.float32(x_high / 255))
finish_time = time.time()
running_time = finish_time - start_time

print("\n******************** Total running time is: ", running_time, " ********************")

fileName = args.outputPath + "/" + "Attack_" + str(args.eps) + "_ID_" + str(args.sample_id) + "_" + str(args.preimg_mode)+ ".txt"

if ifSucc:
    vad_res = dnn_gurobi_model.write_result(qu_frac_list, fileName)
    # obtain the accuracy of quantized network
    loss_DNN, accu_DNN = model.evaluate(x_test, y_test)
    for i in range(len(blkset)):
        Q = qu_list[i]
        I_W = qu_int_list[i]
        F_W = qu_frac_list[i]
        F_B = F_W
        assert Q == I_W + F_W

        paras = model.layers[i].get_weights()
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

    outputMessage_QNN = "\nThe accuracy of QNN got from {} method is: {}".format(args.preimg_mode, accu_QNN)
    outputMessage_DNN = "\nThe accuracy of DNN got from {} method is: {}".format(args.preimg_mode, accu_DNN)
    fo = open(fileName, "a")
    fo.write(outputMessage_DNN)
    fo.write(outputMessage_QNN)
    # Export quantized model to ONNX if requested
    try:
        if args.save_onnx:
            # Determine output directory for ONNX exports
            onnx_dir = args.onnx_dir.strip() if args.onnx_dir else os.path.join(args.outputPath or "output", "onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            onnx_filename = f"quantized_{args.dataset}_{args.arch}_id{args.sample_id}_eps{args.eps}.onnx"
            onnx_path = os.path.join(onnx_dir, onnx_filename)
            save_keras_to_onnx(model, onnx_path)
            print(f"Saved quantized ONNX model to: {onnx_path}")
            fo.write(f"\nSaved quantized ONNX model to: {onnx_path}")
    except Exception as _export_err:
        print(f"Failed to export quantized model to ONNX: {_export_err}")
        fo.write(f"\nFailed to export quantized model to ONNX: {_export_err}")
    fo.close()
else:
    print("Currently, we cannot find a quantization strategy to make the property hold.")
    fo = open(fileName, "w")
    fo.write("Solving Result: False\n")
    fo.write("Currently, we cannot find a quantization strategy to make the property hold.\n")
    if args.preimg_mode == "milp":
        fo.write("Encoding Time: " + str(dnn_gurobi_model._stats["encoding_time"]) + "\n")
        fo.write("Solving Time: " + str(dnn_gurobi_model._stats["solving_time"]) + "\n")
        fo.write("Total Time: " + str(dnn_gurobi_model._stats["total_time"]) + "\n")
    else:
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
