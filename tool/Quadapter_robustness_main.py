# ==================== IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS ====================
import argparse
from pathlib import Path
import numpy as np
from utils.deep_models import *
from utils.quadapter_encoding_robustness import *
from utils.quadapter_utils import *
from utils.data.iris import load_train_test_data
from utils.data.seeds import load_train_test_data_seeds
from utils.data.mnist_64 import load_train_test_data_mnist64
from utils.data.load_onnx import *
from gurobipy import GRB
import time
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure basic logging to the console
logging.basicConfig(filename='logs/quadapter_robustness_main.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Define o valor máximo para restrições (Big-M method usado em programação linear inteira)
bigM = GRB.MAXINT

# ==================== CONFIGURAÇÃO DOS ARGUMENTOS DE LINHA DE COMANDO ====================
# ==================== CONFIGURAÇÃO DOS ARGUMENTOS DE LINHA DE COMANDO ====================
parser = argparse.ArgumentParser()

# Dataset a ser usado (mnist ou fashion-mnist)
parser.add_argument("--dataset", default="mnist")

# Arquitetura da rede neural (ex: 1blk_100 = 1 bloco com 100 neurônios)
parser.add_argument("--arch", default="1blk_100")

# ID da amostra a ser verificada
parser.add_argument("--sample_id", type=int, default=0)

# Limite inferior de bits para quantização (mínimo de bits por parâmetro)
parser.add_argument("--bit_lb", type=int, default=1)

# Limite superior de bits para quantização (máximo de bits por parâmetro)
parser.add_argument("--bit_ub", type=int, default=16)

# Epsilon: raio de perturbação para verificação de robustez (norma L∞)
parser.add_argument("--eps", type=float, default=1)

# Caminho para salvar os resultados da verificação
parser.add_argument("--outputPath", default="")

# Flag para relaxação: 0=sem relaxação, 1=com relaxação
parser.add_argument("--ifRelax", type=int, default=0)

# Modo de computação de pré-imagem para análise backward:
# 'milp' = baseado em MILP (Mixed Integer Linear Programming)
# 'abstr' = baseado em abstração (interval analysis)
# 'comp' = composição dos dois métodos
parser.add_argument("--preimg_mode", default="milp")

parser.add_argument("--verify_mode", default="milp")
# Processa os argumentos fornecidos pela linha de comando
args = parser.parse_args()

# ==================== CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ====================


def _infer_dense_arch_from_h5(weight_file: str | Path) -> list[int]:
    """Infer [input_dim, hidden..., output_dim] from a Keras weight .h5 file."""
    weight_path = Path(weight_file)
    if not weight_path.exists() or weight_path.suffix.lower() != ".h5":
        return []
    try:
        import h5py  # type: ignore
    except ImportError:
        return []

    kernel_shapes: list[tuple[int, int]] = []
    try:
        with h5py.File(weight_path, "r") as h5f:
            layer_names = h5f.attrs.get("layer_names", [])
            if isinstance(layer_names, np.ndarray):
                layer_names = layer_names.tolist()
            for raw_name in layer_names:
                layer_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else raw_name
                if layer_name not in h5f:
                    continue
                layer_group = h5f[layer_name]
                weight_names = layer_group.attrs.get("weight_names", [])
                if isinstance(weight_names, np.ndarray):
                    weight_names = weight_names.tolist()
                for raw_weight in weight_names:
                    weight_name = raw_weight.decode("utf-8") if isinstance(raw_weight, bytes) else raw_weight
                    dataset_key = weight_name.split("/", maxsplit=1)[-1]
                    if not dataset_key.endswith("kernel:0"):
                        continue
                    if dataset_key in layer_group:
                        dataset = layer_group[dataset_key]
                    elif weight_name in layer_group:
                        dataset = layer_group[weight_name]
                    else:
                        continue
                    kernel_shapes.append(tuple(int(d) for d in dataset.shape))
    except OSError:
        return []

    if not kernel_shapes:
        return []

    inferred = [kernel_shapes[0][0]]
    inferred.extend(shape[1] for shape in kernel_shapes)
    return inferred

# Carrega o dataset apropriado com base no argumento fornecido
input_scale = 255.0
if args.dataset == "fashion-mnist":
    logging.info("Loading Fashion-MNIST dataset...")
    # Fashion-MNIST: dataset de roupas e acessórios (28x28, 10 classes)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
elif args.dataset == "mnist" or args.dataset == "mnist_onnx":
    # MNIST: dataset de dígitos manuscritos (28x28, 10 classes)
    logging.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
elif args.dataset == "mnist64":
    logging.info("Loading MNIST 64x64 dataset...")
    # MNIST 64x64: versão redimensionada do MNIST (64x64, 10 classes)
    (x_train, x_test), (y_train, y_test) = load_train_test_data_mnist64()
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
elif "iris" in args.dataset:
    # Iris: dataset clássico de flores Iris (4 features, 3 classes)
    (x_train, x_test), (y_train, y_test) = load_train_test_data()
    input_scale = 1.0
    logging.debug(f"x_train data shape:{x_train.shape}")
    logging.debug(f"y_train data shape: {y_train.shape}") 
    logging.debug(f"x_test data shape: {x_test.shape}")
    logging.debug(f"y_test data shape: {y_test.shape}")
    #input()
elif "seeds" in args.dataset:
    # Seeds: dataset de sementes de trigo (7 features, 3 classes)
    (x_train, x_test), (y_train, y_test) = load_train_test_data_seeds()
    input_scale = 1.0
    logging.debug(f"x_train data shape:{x_train.shape}")
    logging.debug(f"y_train data shape: {y_train.shape}") 
    logging.debug(f"x_test data shape: {x_test.shape}")
    logging.debug(f"y_test data shape: {y_test.shape}")
    #input()
else:
    raise ValueError("Unknown dataset '{}'".format(args.dataset))

# Converte dados para arrays NumPy (simplifica uso posterior)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# Achata as labels de formato (n, 1) para (n,) - remove dimensão extra
y_train = y_train.flatten()
y_test = y_test.flatten()

logging.debug(f"y_train data shape after flatten: {y_train.shape}")
logging.debug(f"y_test data shape after flatten: {y_test.shape}" )

# Redimensiona as imagens 28x28 para vetores 784x1 e converte para float32
# Mantém valores no intervalo [0, 255] (sem normalização ainda)
try:
    if len(x_train.shape) > 2 and np.prod(x_train.shape[1:]) == 28 * 28:
        x_train = x_train.reshape([-1, 28 * 28]).astype(np.float32)
except Exception as e: 
    print(e)
try:
    if len(x_test.shape) > 2 and np.prod(x_test.shape[1:]) == 28 * 28:
        x_test = x_test.reshape([-1, 28 * 28]).astype(np.float32)
except Exception as e: 
    print(e)

# Determina dimensionalidade de entrada e número de classes da tarefa
if x_train.ndim == 1:
    input_dim = x_train.size
else:
    input_dim = x_train.shape[-1]

num_classes = int(np.max(y_train)) + 1

if "iris" in args.dataset or "seeds" in args.dataset or args.dataset == "mnist64":
    weight_path = Path(f"benchmark/{args.dataset.split('_')[0]}/{args.dataset}_weight.h5")

else:
    weight_path = Path(f"benchmark/{args.dataset}/{args.dataset}_{args.arch}_weight.h5")

inferred_arch = _infer_dense_arch_from_h5(weight_path)
blkset_override: list[int] = []
if inferred_arch:
    inferred_input = inferred_arch[0]
    blkset_override = inferred_arch[1:]
    if inferred_input != input_dim:
        print(
            f"Aviso: input_dim inferido ({inferred_input}) difere do dataset ({input_dim}). "
            "Usando valor do arquivo de pesos."
        )
        input_dim = inferred_input
    inferred_classes = blkset_override[-1]
    if inferred_classes != num_classes:
        print(
            f"Aviso: número de classes inferido ({inferred_classes}) difere do dataset ({num_classes}). "
            "Usando valor do arquivo de pesos."
        )
        num_classes = inferred_classes

# ==================== CONSTRUÇÃO DA ARQUITETURA DA REDE ====================
logging.info("Starting Quadapter Robustness Main")

# Parseia a string de arquitetura (ex: "1blk_100" -> ["1blk", "100"])
archMnist = args.arch.split('_')

# Extrai o número de blocos (remove "blk" da string)
# Ex: "1blk" -> "1"
numBlk = archMnist[0][:-3]

# Converte os tamanhos das camadas ocultas para inteiros
# Ex: ["100", "50"] -> [100, 50]
blkset = list(map(int, archMnist[1:]))

# Adiciona a camada de saída com o número de classes da tarefa
blkset.append(num_classes)

if blkset_override:
    blkset = blkset_override

# Combina entrada + camadas ocultas + saída
# Ex: arch = [input_dim, 100, num_classes] para uma rede com 1 camada oculta
arch = [input_dim] + blkset

# Verifica se o número de blocos especificado corresponde à arquitetura parseada
# (número de camadas - 1, excluindo a camada de saída)
try:
    expected_blocks = int(numBlk)
except ValueError:
    expected_blocks = len(blkset) - 1
if expected_blocks != len(blkset) - 1:
    print(
        f"Aviso: número de blocos inferido ({len(blkset) - 1}) "
        f"diferente do fornecido via --arch ({expected_blocks}). Usando configuração dos pesos."
    )
# ==================== CRIAÇÃO E CARREGAMENTO DO MODELO ====================
logging.debug(f"Architecture is: {arch}")
logging.debug(f"Number of hidden blocks: {len(blkset) - 1}")
logging.debug(f"Layer widths (excluding input): {blkset}")
# Cria o modelo de rede neural profunda com a arquitetura especificada
model = DeepModel(
    blkset,  # Lista com tamanhos das camadas [camadas_ocultas..., num_classes]
    last_layer_signed=True,  # Última camada usa valores com sinal (logits)
    input_scale=input_scale,
)

# Compila o modelo especificando:
# - Otimizador: Adam com learning rate baixo
# - Função de perda: Cross-entropy categórica esparsa (logits não normalizados)
# - Métrica: Acurácia categórica esparsa
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
#if args.dataset == "iris":
#    scaler = MinMaxScaler()
#    x_train = scaler.fit_transform(x_train)
#    x_test = scaler.transform(x_test)
#    x_train = x_train.astype(np.float32)
#    x_test = x_test.astype(np.float32)

#model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=0)

# Constrói o modelo com o formato de entrada correto (None, input_dim)
# None permite batch size variável

print("input dim is: ", input_dim)
model.build((None, input_dim))
# Carrega os pesos pré-treinados do arquivo .h5
_ = model(tf.zeros((1, input_dim), dtype=tf.float32))
model.load_weights(str(weight_path))  # input: 0~255


# ==================== VERIFICAÇÃO DA PREDIÇÃO ORIGINAL ====================

# Seleciona a amostra de teste com base no sample_id fornecido
print("Sample ID is: ", args.sample_id)
logging.debug(f"get weight form the model {model.get_weights()}")

for i in range(30):
    logging.debug(f"x_test[{i}] is: {x_train[i]}")
    model_out = model.predict(np.expand_dims(x_train[i], 0))[0]
    # Obtém a classe predita (índice do maior logit)
    model_predict = np.argmax(model_out)

    # Obtém a classe verdadeira (ground truth) da amostra
    original_prediction = y_train[i]

    logging.debug(f"\nModel output is: {model_out}")
    logging.debug(f"\nModel prediction is: { model_predict}")

    # Verifica se a predição está correta
    # Só verificamos amostras que o modelo classifica corretamente
    logging.debug(f"\nOriginal label is: {original_prediction}")
    logging.debug("Checking whether the original prediction is correct...")
    logging.debug("If correct, then proceed to verified quantization...")
    try:
        assert model_predict == original_prediction
    except AssertionError:
        logging.debug("The prediction is incorrect. Skip to the next one.")
        continue
x_input = x_test[args.sample_id]

# Faz a predição para a amostra selecionada (adiciona dimensão batch)
# Retorna array com logits para cada classe
print("x_input is: ", x_input)
model_out = model.predict(np.expand_dims(x_test[args.sample_id], 0))[0]

# Obtém a classe predita (índice do maior logit)
model_predict = np.argmax(model_out)

# Obtém a classe verdadeira (ground truth) da amostra
original_prediction = y_test[args.sample_id]

print("\nModel output is: ", model_out)
print("\nModel prediction is: ", model_predict)

# Verifica se a predição está correta
# Só verificamos amostras que o modelo classifica corretamente
print("\nOriginal label is: ", original_prediction)
print("Checking whether the original prediction is correct...")
print("If correct, then proceed to verified quantization...")
assert model_predict == original_prediction

print("original_prediction is: ", original_prediction, '\n')
# ==================== DEFINIÇÃO DA REGIÃO DE ENTRADA (INPUT BOX) ====================
# Define os limites inferior e superior da região de perturbação L∞
# x_low: entrada original - epsilon (limitado ao intervalo [0, 255])
# x_high: entrada original + epsilon (limitado ao intervalo [0, 255])
# np.clip garante que os valores permaneçam no domínio válido das imagens
clip_low = 0.0
clip_high = input_scale if input_scale not in (None, 0) else np.max(x_train)
x_low, x_high = np.clip(x_input - args.eps, clip_low, clip_high), np.clip(x_input + args.eps, clip_low, clip_high)

# ==================== CRIAÇÃO DO MODELO DE VERIFICAÇÃO (GUROBI) ====================

# Cria o modelo de codificação Gurobi para verificação de quantização
# Normaliza os valores de entrada dividindo por 255 (converte [0,255] -> [0,1])
# Isso é necessário porque a rede foi treinada com entradas normalizadas
dnn_gurobi_model = GPEncoding(arch, model, args, original_prediction, x_low / input_scale, x_high / input_scale)

# ==================== PROPAGAÇÃO FORWARD NA DNN ====================

# Executa forward pass na rede neural para obter os valores reais de saída
# Normaliza a entrada dividindo por 255 (mesmo que durante o treinamento)
# Isso estabelece os valores de referência para a verificação
res = forward_DNN(x_input / input_scale, dnn_gurobi_model)

# ==================== VERIFICAÇÃO DE QUANTIZAÇÃO ====================

# Marca o tempo de início da verificação
start_time = time.time()

# Executa o algoritmo de quantização verificada
# Este é o núcleo do Quadapter: encontra estratégia de quantização que preserva robustez
# Retorna:
# - ifSucc: flag indicando se encontrou uma estratégia de quantização válida
# - qu_list: lista de bits totais por camada (bits inteiros + fracionários)
# - qu_frac_list: lista de bits fracionários por camada (para representar decimais)
# - qu_int_list: lista de bits inteiros por camada (para representar parte inteira)
ifSucc, qu_list, qu_frac_list, qu_int_list = dnn_gurobi_model.verified_quant(
    np.float32(x_low / input_scale),    # Limite inferior normalizado da região de entrada
    np.float32(x_high / input_scale))   # Limite superior normalizado da região de entrada

# Marca o tempo de término e calcula o tempo total de execução
finish_time = time.time()
running_time = finish_time - start_time

print("\n******************** Total running time is: ", running_time, " ********************")

# ==================== GRAVAÇÃO DOS RESULTADOS ====================

# Define o nome do arquivo de saída com base nos parâmetros de execução
# Formato: Attack_{epsilon}_ID_{sample_id}_{método}.txt
fileName = args.outputPath + "/" + "Attack_" + str(args.eps) + "_ID_" + str(args.sample_id) + "_" + str(args.preimg_mode)+ "_"+str(args.verify_mode)+ "_"+args.dataset+".txt"

# ==================== CASO DE SUCESSO: QUANTIZAÇÃO VÁLIDA ENCONTRADA ====================
if ifSucc:
    # Escreve os resultados da quantização no arquivo especificado
    vad_res = dnn_gurobi_model.write_result(qu_frac_list, fileName)
    
    # ==================== AVALIAÇÃO DA ACURÁCIA DA DNN ORIGINAL ====================
    # Avalia a rede neural original (ponto flutuante) no conjunto de teste
    # para estabelecer linha de base de performance
    loss_DNN, accu_DNN = model.evaluate(x_test, y_test)
    # ==================== APLICAÇÃO DA QUANTIZAÇÃO AO MODELO ====================
    # Itera sobre cada camada para aplicar a quantização de ponto fixo encontrada
    for i in range(len(blkset)):
        # Extrai os parâmetros de quantização para a camada atual:
        Q = qu_list[i]      # Número total de bits (inteiro + fracionário)
        I_W = qu_int_list[i]  # Bits para parte inteira dos pesos
        F_W = qu_frac_list[i] # Bits para parte fracionária dos pesos
        F_B = F_W           # Bits para parte fracionária dos biases (mesmo que pesos)
        
        # Verifica a consistência: total = inteiro + fracionário
        assert Q == I_W + F_W

        # Obtém os pesos e biases originais (ponto flutuante) da camada
        paras = model.dense_layers[i].get_weights()
        
        # ==================== QUANTIZAÇÃO DOS PESOS ====================
        new_weight = []
        # Para cada neurônio de entrada (itera sobre colunas da matriz de pesos)
        for j in range(len(paras[0])):
            # Extrai os pesos do neurônio j para todos os neurônios da camada seguinte
            weight_j = paras[0][j].tolist()
            
            # Aplica quantização de ponto fixo:
            # 1. Multiplica por 2^F_W para escalar para inteiros
            # 2. Arredonda para o inteiro mais próximo (real_round)
            # 3. Divide por 2^F_W para voltar ao domínio de ponto fixo
            # Isso simula aritmética de ponto fixo com F_W bits fracionários
            weight_j = list(map(lambda a: real_round(a * (2 ** F_W)) / (2 ** F_W), weight_j))
            new_weight.append(weight_j)

        # Converte de volta para numpy array
        new_weight = np.asarray(new_weight)

        # ==================== QUANTIZAÇÃO DOS BIASES ====================
        bias = paras[1].tolist()
        # Aplica a mesma quantização de ponto fixo aos biases
        bias = list(map(lambda a: real_round(a * (2 ** F_B)) / (2 ** F_B), bias))
        bias = np.asarray(bias)
        
        # Atualiza a camada com os pesos e biases quantizados
        model.dense_layers[i].set_weights([new_weight, bias])
    # ==================== AVALIAÇÃO DA ACURÁCIA DA QNN ====================
    # Avalia a rede quantizada no conjunto de teste
    # Compara com a acurácia original para medir degradação
    loss_QNN, accu_QNN = model.evaluate(x_test, y_test)

    # ==================== GRAVAÇÃO DAS ACURÁCIAS ====================
    # Prepara mensagens com os resultados de acurácia
    outputMessage_QNN = "\nThe accuracy of QNN got from {} method is: {}".format(args.preimg_mode, accu_QNN)
    outputMessage_DNN = "\nThe accuracy of DNN got from {} method is: {}".format(args.preimg_mode, accu_DNN)
    
    # Abre o arquivo em modo append e adiciona as mensagens de acurácia
    fo = open(fileName, "a")
    fo.write(outputMessage_DNN)
    fo.write(outputMessage_QNN)
    fo.close()
# ==================== CASO DE FALHA: QUANTIZAÇÃO NÃO ENCONTRADA ====================
else:
    print("Currently, we cannot find a quantization strategy to make the property hold.")
    
    # Abre arquivo para escrever mensagem de falha
    fo = open(fileName, "w")
    fo.write("Solving Result: False\n")
    fo.write("Currently, we cannot find a quantization strategy to make the property hold.\n")
    
    # ==================== ESTATÍSTICAS ESPECÍFICAS DO MÉTODO ====================
    # Para método MILP: tempo de codificação e resolução separados
    if args.preimg_mode == "milp":
        # Para MILP: registra tempos de codificação e otimização
        fo.write("Encoding Time: " + str(dnn_gurobi_model._stats["encoding_time"]) + "\n")
        fo.write("Solving Time: " + str(dnn_gurobi_model._stats["solving_time"]) + "\n")
        fo.write("Total Time: " + str(dnn_gurobi_model._stats["total_time"]) + "\n")
    else:
        # Para outros métodos (abstr, comp): registra tempos de análise backward e forward
        fo.write("Backward Time: " + str(dnn_gurobi_model._stats["backward_time"]) + "\n")
        fo.write("Forward Time: " + str(dnn_gurobi_model._stats["forward_time"]) + "\n")
        fo.write("Total Time: " + str(dnn_gurobi_model._stats["total_time"]) + "\n")
    
    # ==================== ESTATÍSTICAS DO MODELO GUROBI ====================
    # Coleta informações sobre o tamanho do problema de otimização
    numAllVars = dnn_gurobi_model.gp_model.getAttr("NumVars")      # Total de variáveis
    numIntVars = dnn_gurobi_model.gp_model.getAttr("NumIntVars")    # Variáveis inteiras
    numBinVars = dnn_gurobi_model.gp_model.getAttr("NumBinVars")    # Variáveis binárias
    numConstrs = dnn_gurobi_model.gp_model.getAttr("NumConstrs")    # Número de restrições
    
    # Escreve as estatísticas do problema no arquivo
    fo.write("The num of vars: " + str(numAllVars) + "\n")
    fo.write("The num of numIntVars: " + str(numIntVars) + "\n")
    fo.write("The num of numBinVars: " + str(numBinVars) + "\n")
    fo.write("The num of Constraints: " + str(numConstrs) + "\n")
    fo.close()
