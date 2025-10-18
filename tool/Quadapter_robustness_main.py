# ==================== IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS ====================
import argparse
from utils.deep_models import *
from utils.quadapter_encoding_robustness import *
from utils.quadapter_utils import *
from utils.data.iris import load_train_test_data
from utils.data.load_onnx import *
from gurobipy import GRB

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
parser.add_argument("--eps", type=int, default=2)

# Caminho para salvar os resultados da verificação
parser.add_argument("--outputPath", default="")

# Flag para relaxação: 0=sem relaxação, 1=com relaxação
parser.add_argument("--ifRelax", type=int, default=0)

# Modo de computação de pré-imagem para análise backward:
# 'milp' = baseado em MILP (Mixed Integer Linear Programming)
# 'abstr' = baseado em abstração (interval analysis)
# 'comp' = composição dos dois métodos
parser.add_argument("--preimg_mode", default="milp")

# Processa os argumentos fornecidos pela linha de comando
args = parser.parse_args()

# ==================== CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ====================

# Carrega o dataset apropriado com base no argumento fornecido
if args.dataset == "fashion-mnist":
    # Fashion-MNIST: dataset de roupas e acessórios (28x28, 10 classes)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
elif args.dataset == "mnist":
    # MNIST: dataset de dígitos manuscritos (28x28, 10 classes)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("x_train data shape:", x_train.shape)
    print("y_train data shape:", y_train.shape) 
    print("x_test data shape:", x_test.shape)
elif args.dataset == "iris":
    # Iris: dataset clássico de flores Iris (4 features, 3 classes)
    (x_train, x_test), (y_train, y_test) = load_train_test_data()
    #input()
else:
    raise ValueError("Unknown dataset '{}'".format(args.dataset))

# Achata as labels de formato (n, 1) para (n,) - remove dimensão extra


y_train = y_train.flatten()
y_test = y_test.flatten()

print("y_train data shape after flatten:", y_train.shape)
print("y_test data shape after flatten:", y_test.shape)

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
# ==================== CONSTRUÇÃO DA ARQUITETURA DA REDE ====================

# Parseia a string de arquitetura (ex: "1blk_100" -> ["1blk", "100"])
archMnist = args.arch.split('_')

# Extrai o número de blocos (remove "blk" da string)
# Ex: "1blk" -> "1"
numBlk = archMnist[0][:-3]

# Inicializa a arquitetura com a camada de entrada (784 = 28*28 pixels)
arch = [784]

# Converte os tamanhos das camadas ocultas para inteiros
# Ex: ["100", "50"] -> [100, 50]
blkset = list(map(int, archMnist[1:]))

# Adiciona a camada de saída (10 classes para MNIST/Fashion-MNIST)
blkset.append(10)

# Combina entrada + camadas ocultas + saída
# Ex: arch = [784, 100, 10] para uma rede com 1 camada oculta de 100 neurônios
arch += blkset

# Verifica se o número de blocos especificado corresponde à arquitetura parseada
# (número de camadas - 1, excluindo a camada de saída)
assert int(numBlk) == len(blkset) - 1

# ==================== CRIAÇÃO E CARREGAMENTO DO MODELO ====================

# Cria o modelo de rede neural profunda com a arquitetura especificada
model = DeepModel(
    blkset,  # Lista com tamanhos das camadas [camadas_ocultas..., 10]
    last_layer_signed=True,  # Última camada usa valores com sinal (logits)
)

# Define o caminho dos pesos pré-treinados baseado no dataset e arquitetura

if args.dataset == "iris":
    weight_path = "benchmark/{}/{}_weight.h5".format(args.dataset, args.dataset)
else:
    weight_path = "benchmark/{}/{}_{}_weight.h5".format(args.dataset, args.dataset, args.arch)

# Compila o modelo especificando:
# - Otimizador: Adam com learning rate baixo
# - Função de perda: Cross-entropy categórica esparsa (logits não normalizados)
# - Métrica: Acurácia categórica esparsa
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Constrói o modelo com o formato de entrada correto (None, 784)
# None permite batch size variável
model.build((None, 28 * 28))

# Carrega os pesos pré-treinados do arquivo .h5
model.load_weights(weight_path)  # input: 0~255

# ==================== VERIFICAÇÃO DA PREDIÇÃO ORIGINAL ====================

# Seleciona a amostra de teste com base no sample_id fornecido
x_input = x_test[args.sample_id]

# Faz a predição para a amostra selecionada (adiciona dimensão batch)
# Retorna array com logits para cada classe
model_out = model.predict(np.expand_dims(x_test[args.sample_id], 0))[0]

# Obtém a classe predita (índice do maior logit)
model_predict = np.argmax(model_out)

# Obtém a classe verdadeira (ground truth) da amostra
original_prediction = y_test[args.sample_id]

print("\nModel output is: ", model_out)
print("\nModel prediction is: ", model_predict)

# Verifica se a predição está correta
# Só verificamos amostras que o modelo classifica corretamente
assert model_predict == original_prediction

print("original_prediction is: ", original_prediction, '\n')

# ==================== DEFINIÇÃO DA REGIÃO DE ENTRADA (INPUT BOX) ====================

# Define os limites inferior e superior da região de perturbação L∞
# x_low: entrada original - epsilon (limitado ao intervalo [0, 255])
# x_high: entrada original + epsilon (limitado ao intervalo [0, 255])
# np.clip garante que os valores permaneçam no domínio válido das imagens
x_low, x_high = np.clip(x_input - args.eps, 0, 255), np.clip(x_input + args.eps, 0, 255)

# ==================== CRIAÇÃO DO MODELO DE VERIFICAÇÃO (GUROBI) ====================

# Cria o modelo de codificação Gurobi para verificação de quantização
# Normaliza os valores de entrada dividindo por 255 (converte [0,255] -> [0,1])
# Isso é necessário porque a rede foi treinada com entradas normalizadas
dnn_gurobi_model = GPEncoding(arch, model, args, original_prediction, x_low / 255, x_high / 255)

# ==================== PROPAGAÇÃO FORWARD NA DNN ====================

# Executa forward pass na rede neural para obter os valores reais de saída
# Normaliza a entrada dividindo por 255 (mesmo que durante o treinamento)
# Isso estabelece os valores de referência para a verificação
res = forward_DNN(x_input / 255, dnn_gurobi_model)

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
    np.float32(x_low / 255),    # Limite inferior normalizado da região de entrada
    np.float32(x_high / 255))   # Limite superior normalizado da região de entrada

# Marca o tempo de término e calcula o tempo total de execução
finish_time = time.time()
running_time = finish_time - start_time

print("\n******************** Total running time is: ", running_time, " ********************")

# ==================== GRAVAÇÃO DOS RESULTADOS ====================

# Define o nome do arquivo de saída com base nos parâmetros de execução
# Formato: Attack_{epsilon}_ID_{sample_id}_{método}.txt
fileName = args.outputPath + "/" + "Attack_" + str(args.eps) + "_ID_" + str(args.sample_id) + "_" + str(args.preimg_mode)+ ".txt"

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
        paras = model.layers[i].get_weights()
        
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
        model.layers[i].set_weights([new_weight, bias])

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