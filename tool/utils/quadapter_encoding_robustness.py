# ==================== IMPORTAÇÕES E DEPENDÊNCIAS ====================
from symbolic_pp.DeepPoly_quadapter import *  # DeepPoly para análise de intervalos simbólicos
from utils.quadapter_utils import *           # Utilitários específicos do Quadapter
from utils.abstract import *                  # Funções para geração de código C abstrato
import math                                   # Operações matemáticas
from gurobipy import GRB                     # Constantes e tipos do Gurobi
import gurobipy as gp                        # Solver MILP (Mixed Integer Linear Programming)
import time                                  # Medição de tempo de execução
import numpy as np                           # Operações numéricas e arrays
import pdb
import logging
from utils.logs import LogFile

# Configuração do logger para o módulo
log_instance = LogFile(log_file_path='logs/quadapter_esbmc_output.log', log_name='quadapter_esbmc_output')
log_esbmc = log_instance.get_logger()

log_instance_parameters = LogFile(log_file_path='logs/quadapater_parameters.log', log_name='quadapater_parameters')
log_parameters = log_instance_parameters.get_logger()

logging.basicConfig(filename='logs/quadapter_encoding_robustness.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# ==================== CLASSE PARA CODIFICAÇÃO DE CAMADAS ====================
class LayerEncoding:
    """
    Classe que codifica uma camada da rede neural para verificação de quantização.
    Mantém informações sobre bounds, quantização e variáveis do modelo Gurobi.
    """
    def __init__(
            self,
            gp_model,      # Modelo Gurobi para otimização
            preimg_mode,   # Modo de computação de pré-imagem ('milp', 'abstr', 'comp')
            layer_index,   # Índice da camada na rede
            layer_size,    # Número de neurônios na camada
            layer_paras,   # Parâmetros da camada (pesos e biases)
            bit_lb,        # Limite inferior de bits para quantização
            bit_ub,        # Limite superior de bits para quantização
            if_hid,        # Flag indicando se é camada oculta (True) ou saída/entrada (False)
    ):
        # ==================== INICIALIZAÇÃO DOS ATRIBUTOS BÁSICOS ====================
        # Armazena informações básicas da camada
        self.layer_index = layer_index    # Índice da camada na rede
        self.layer_size = layer_size      # Número de neurônios
        self.layer_paras = layer_paras    # Pesos e biases [weight_matrix, bias_vector]
        self.bit_lb = bit_lb             # Limite mínimo de bits para quantização
        self.bit_ub = bit_ub             # Limite máximo de bits para quantização
        self.frac_bit = None             # Bits fracionários (será determinado depois)
        self.grad = None                 # Gradiente (não usado nesta implementação)
        self.realVal = None              # Valores reais de ativação

        # ==================== CONFIGURAÇÃO DE LIMITES DE NEURÔNIOS ====================
        # Define limites para variáveis do Gurobi baseado no tipo de camada
        if if_hid:
            # Camadas ocultas: após ReLU, valores ≥ 0
            neuron_lb_after = 0
        else:
            # Camadas de entrada/saída: valores podem ser negativos
            neuron_lb_after = -GRB.MAXINT

        # Antes da ativação ReLU, valores podem ser negativos em qualquer camada
        neuron_lb_before = -GRB.MAXINT

        # ==================== INICIALIZAÇÃO DOS ARRAYS DE BOUNDS ====================
        # Arrays para armazenar limites concretos (obtidos via DeepPoly)
        self.lb = np.zeros(layer_size, dtype=np.float32)           # Limite inferior concreto
        self.ub = np.zeros(layer_size, dtype=np.float32)           # Limite superior concreto
        self.clipped_lb = np.zeros(layer_size, dtype=np.float32)   # Limite inferior após ReLU
        self.clipped_ub = np.zeros(layer_size, dtype=np.float32)   # Limite superior após ReLU

        # Arrays para limites da rede quantizada (QNN)
        self.qu_lb = np.zeros(layer_size, dtype=np.float32)        # Limite inferior da QNN
        self.qu_ub = np.zeros(layer_size, dtype=np.float32)        # Limite superior da QNN
        self.qu_clipped_lb = np.zeros(layer_size, dtype=np.float32) # Limite inferior da QNN após ReLU
        self.qu_clipped_ub = np.zeros(layer_size, dtype=np.float32) # Limite superior da QNN após ReLU

        # ==================== VARIÁVEIS BINÁRIAS PARA CODIFICAÇÃO DE BITS ====================
        # Cria variáveis binárias para codificar o número de bits usado na quantização
        # Uma variável para cada possível número de bits no intervalo [bit_lb, bit_ub]
        self.bit_vars = [gp_model.addVar(vtype=GRB.BINARY) for i in range(self.bit_ub - self.bit_lb + 1)]

        # ==================== CÁLCULO DOS BITS INTEIROS NECESSÁRIOS ====================
        # Calcula o número mínimo de bits inteiros necessários para evitar overflow
        if layer_index > 0:
            # Para camadas não-entrada, analisa os parâmetros da camada
            self.max_weight = np.round(max(np.max(layer_paras[0]), np.max(layer_paras[1])))
            self.min_weight = np.round(min(np.min(layer_paras[0]), np.min(layer_paras[1])))
            self.max_int = max(abs(self.max_weight), abs(self.min_weight))
            
            # Determina bits inteiros baseado no maior valor absoluto
            if self.max_int == 0:
                self.int_bit = 1     # Mínimo 1 bit para representar zero
            elif self.max_int == 1:
                self.int_bit = 2     # 2 bits para representar ±1 (1 bit + sinal)
            else:
                # log2(max_value) + 1 bit para sinal
                self.int_bit = int(np.ceil(math.log(self.max_int, 2)) + 1)
        else:
            # Camada de entrada: não tem parâmetros próprios
            self.int_bit = None

        # ==================== INICIALIZAÇÃO DE BOUNDS RELAXADOS ====================
        # Arrays para armazenar bounds relaxados (usados na computação de pré-imagem)
        self.relaxed_lb = np.zeros(layer_size, dtype=np.float32)      # Limite inferior relaxado
        self.relaxed_lb_expression = [1 for i in range(layer_size)]   # Expressão simbólica do limite inferior
        self.relaxed_ub = np.zeros(layer_size, dtype=np.float32)      # Limite superior relaxado
        self.relaxed_ub_expression = [1 for i in range(layer_size)]   # Expressão simbólica do limite superior
        self.actMode = np.zeros(layer_size, dtype=np.float32)         # Modo de ativação (0=inativo, 1=ativo, 2=saturado)

        # ==================== CRIAÇÃO DE VARIÁVEIS GUROBI PARA PRÉ-IMAGEM ====================
        # Inicializa variáveis para codificação do template de pré-imagem
        # Diferentes conjuntos de variáveis dependem do modo de computação

        if preimg_mode == 'milp' or preimg_mode == 'comp':
            # Modo MILP ou composição: variáveis para codificação exata
            self.gp_vars_before = [gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for s in
                                      range(layer_size)]  # Valores antes da função ReLU
            self.gp_vars_after = [gp_model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS) for s in
                                      range(layer_size)]  # Valores após a função ReLU
            # Variáveis para controlar a relaxação da pré-imagem
            self.alpha = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
            self.beta = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]

        elif preimg_mode == 'abstr' or preimg_mode == 'comp':
            # Modo abstração ou composição: variáveis para análise de intervalos
            self.gp_vars_lb_before = [gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for s in
                                      range(layer_size)]  # Limite inferior antes do ReLU
            self.gp_vars_ub_before = [gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for s in
                                      range(layer_size)]  # Limite superior antes do ReLU
            # Variáveis alpha e beta para relaxação antes e depois do ReLU
            self.alpha_before = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
            self.alpha_after = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
            self.beta_before = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
            self.beta_after = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
        else:
            print("Wrong option for the preimage computation mode!")
            exit(0)

        # Atualiza o modelo Gurobi para registrar todas as novas variáveis
        gp_model.update()

        print("The quantization bit size for integer parts of Layer ", self.layer_index, " is: ", self.int_bit)

    def set_input_bounds(self, low, high):
        """Define os limites de entrada para a camada."""
        self.lb = low
        self.ub = high

    def set_realVal(self, realVal):
        """Define os valores reais de ativação para a camada (usado como referência)."""
        self.realVal = realVal
        print("We set the real output values for the layer ", self.layer_index)


# ==================== CLASSE PRINCIPAL PARA CODIFICAÇÃO GUROBI ====================
class GPEncoding:
    """
    Classe principal que coordena a verificação de quantização usando Gurobi.
    Implementa o algoritmo Quadapter para encontrar estratégias de quantização robustas.
    """
    def __init__(self, arch, model, args, original_prediction, x_low_real, x_high_real):
        # ==================== CONFIGURAÇÃO DO MODELO GUROBI ====================
        # Inicializa o solver de otimização MILP
        self.gp_model = gp.Model("gp_encoding")
        self.tole = 1e-6                                    # Tolerância para comparações numéricas
        self.gp_model.Params.IntFeasTol = 1e-9             # Tolerância de factibilidade para inteiros
        self.gp_model.Params.FeasibilityTol = self.tole    # Tolerância geral de factibilidade
        self.gp_model.setParam(GRB.Param.Threads, 30)      # Usa até 30 threads para paralelização
        self.gp_model.setParam(GRB.Param.OutputFlag, 0)    # Suprime saída do Gurobi

        # ==================== PARÂMETROS DE CONFIGURAÇÃO ====================
        # Extrai parâmetros dos argumentos de linha de comando
        self.bit_lb = args.bit_lb                           # Limite inferior de bits para quantização
        self.bit_ub = args.bit_ub                           # Limite superior de bits para quantização
        self.preimg_mode = args.preimg_mode                 # Modo de computação de pré-imagem
        self.x_low_real = x_low_real                        # Limite inferior da região de entrada
        self.x_high_real = x_high_real                      # Limite superior da região de entrada
        self.sample_id = args.sample_id                     # ID da amostra sendo verificada
        self.eps = args.eps                                 # Raio de perturbação (epsilon)
        self.outputPath = args.outputPath                   # Caminho para salvar resultados
        self.ifRelax = args.ifRelax                         # Flag de relaxação
        self.scaleValueSet = []                             # Armazena valores de escala para cada camada
        self.verify_mode = args.verify_mode
        # ==================== ESTATÍSTICAS DE PERFORMANCE ====================
        # Dicionário para rastrear tempos de execução de diferentes fases
        self._stats = {
            "encoding_time": 0,      # Tempo para codificar o problema MILP
            "solving_time": 0,       # Tempo para resolver o problema MILP
            "backward_time": 0,      # Tempo para computação backward (pré-imagem)
            "forward_time": 0,       # Tempo para quantização forward
            "total_time": 0,         # Tempo total de execução
        }

        # ==================== INICIALIZAÇÃO DE ESTRUTURAS DA REDE ====================
        # Estruturas para armazenar informações das camadas
        self.dense_layers = []                              # Lista de camadas ocultas codificadas
        self.nnparas = []                                   # Parâmetros de cada camada (pesos/biases)
        self.deep_model = model                             # Referência ao modelo original
        self.layerNum = len(model.dense_layers)             # Número de camadas densas
        self.targetCls = original_prediction                # Classe alvo (predição original)
        self.deepPolyNets_DNN = DP_DNN_network(True)       # Rede DeepPoly para análise simbólica

        # ==================== EXTRAÇÃO DOS PARÂMETROS DA REDE ====================
        # Lista para variáveis de entrada no modelo Gurobi
        self.input_gp_vars = []
        
        # Extrai pesos e biases de cada camada densa do modelo
        for i, l in enumerate(model.dense_layers):
            tf_layer = model.dense_layers[i]
            w_cont, b_cont = tf_layer.get_weights()         # Obtém pesos e biases
            paras = [w_cont.T, b_cont]                      # Transpõe pesos para formato correto
            self.nnparas.append(paras)                      # Adiciona à lista de parâmetros

        # ==================== CRIAÇÃO DA CAMADA DE SAÍDA ====================
        # Cria codificação para a camada de saída (última camada)
        self.output_layer = LayerEncoding(self.gp_model, preimg_mode=self.preimg_mode,
                                          layer_index=len(self.nnparas),    # Índice após todas as camadas ocultas
                                          layer_size=arch[-1],              # Tamanho da camada de saída
                                          layer_paras=self.nnparas[-1], bit_lb=self.bit_lb, bit_ub=self.bit_ub,
                                          if_hid=False)                     # Não é camada oculta

        # ==================== CRIAÇÃO DAS CAMADAS OCULTAS ====================
        # Cria codificação para cada camada oculta
        for layer in range(len(arch) - 2):
            self.dense_layers.append(
                LayerEncoding(self.gp_model, preimg_mode=self.preimg_mode,
                              layer_index=layer + 1,                    # Índices 1, 2, 3, ...
                              layer_size=arch[layer + 1],               # Tamanho da camada
                              layer_paras=self.nnparas[layer],          # Parâmetros correspondentes
                              bit_lb=self.bit_lb, bit_ub=self.bit_ub,
                              if_hid=True)                              # É camada oculta
            )
            self.scaleValueSet.append(0)                               # Inicializa valor de escala

        # ==================== CRIAÇÃO DA CAMADA DE ENTRADA ====================
        # Cria codificação para a camada de entrada
        input_size = arch[0]                                           # Tamanho da entrada (784 para MNIST)

        self.input_layer = LayerEncoding(self.gp_model, preimg_mode=self.preimg_mode,
                                         layer_index=0,                 # Primeira camada (índice 0)
                                         layer_size=input_size,         # 784 neurônios para MNIST
                                         layer_paras=None,              # Entrada não tem parâmetros próprios
                                         bit_lb=self.bit_lb, bit_ub=self.bit_ub,
                                         if_hid=False)                  # Não é camada oculta

        # ==================== CONFIGURAÇÃO DO DEEPPOLY ====================
        # Carrega o modelo na rede DeepPoly para análise simbólica
        self.deepPolyNets_DNN.load_dnn(model)

        # ==================== CRIAÇÃO DE VARIÁVEIS DE ENTRADA NO GUROBI ====================
        # Adiciona variáveis de entrada com restrições de bounds
        for input_index in range(self.input_layer.layer_size):
            x_lb = x_low_real[input_index]                             # Limite inferior da entrada
            x_ub = x_high_real[input_index]                           # Limite superior da entrada
            # Cria variável contínua com bounds específicos
            cur_var = self.gp_model.addVar(lb=x_lb, ub=x_ub, vtype=GRB.CONTINUOUS)
            self.input_gp_vars.append(cur_var)

    def verified_quant(self, lb, ub):
        """
        Método principal para verificação de quantização.
        Executa o algoritmo Quadapter para encontrar estratégia de quantização robusta.
        """
        # ==================== CONFIGURAÇÃO DA REGIÃO DE ENTRADA ====================
        # Define a caixa de entrada (input box) para a verificação
        self.assert_input_box(lb, ub)

        # ==================== PROPAGAÇÃO SIMBÓLICA ====================
        # Executa análise DeepPoly para obter bounds simbólicos
        self.symbolic_propagate()

        # ==================== VERIFICAÇÃO DA PROPRIEDADE NA DNN ORIGINAL ====================
        # A DNN deve satisfazer a propriedade de robustez antes da quantização
        out_bounds_lb = self.output_layer.lb                          # Limites inferiores da saída
        out_bounds_ub = self.output_layer.ub                          # Limites superiores da saída
        other_max = -1000                                              # Máximo limite superior das outras classes
        # Encontra o máximo limite superior entre todas as classes exceto a alvo
        for i, v in enumerate(self.output_layer.ub):
            if i == self.targetCls:
                continue                                               # Pula a classe alvo
            else:
                other_max = max(other_max, v)                         # Atualiza máximo das outras classes

        print("The lower bound of the target class: ", out_bounds_lb[self.targetCls])
        print("The maximal upper bound of other classes: ", other_max)

        # ==================== VERIFICAÇÃO DA CONDIÇÃO DE ROBUSTEZ ====================
        # Verifica se o limite inferior da classe alvo é maior que o máximo das outras
        # Isso garante que a propriedade de classificação é robusta na região de entrada
        logging.debug(f"Target class lower bound: {out_bounds_lb[self.targetCls]} >= {other_max}")
        if (out_bounds_lb[self.targetCls] >= other_max):
            # ==================== COMPUTAÇÃO BACKWARD (PRÉ-IMAGEM) ====================
            # Calcula pré-imagens relaxadas para todas as camadas
            backward_start_time = time.time()
            self.backward_preimage_computation()
            backward_end_time = time.time()
            print("Backward Time is: ", backward_end_time - backward_start_time)

            if self.verify_mode == "esbmc":
                ifSucc, qu_list, qu_frac_list, qu_int_list = self.forward_quantization_with_esbmc()
            else:
                ifSucc, qu_list, qu_frac_list, qu_int_list = self.forward_quantization()
            
            forward_end_time = time.time()

            # ==================== QUANTIZAÇÃO FORWARD COM ESBMC ====================
            # Busca estratégia de quantização usando verificação ESBMC
            print("Forward time is: ", forward_end_time - backward_end_time)

            # ==================== ATUALIZAÇÃO DAS ESTATÍSTICAS ====================
            # Registra tempos de execução para análise de performance
            self._stats["backward_time"] = backward_end_time - backward_start_time
            self._stats["forward_time"] = forward_end_time - backward_end_time
            self._stats["total_time"] = self._stats["backward_time"] + self._stats["forward_time"]

            return ifSucc, qu_list, qu_frac_list, qu_int_list

        else:
            # ==================== PROPRIEDADE NÃO SATISFEITA ====================
            # Se a DNN original não satisfaz a propriedade, não há como quantizar robustamente
            print("The property does not hold in DNN!")
            exit(0)

    def assert_input_box(self, x_lb, x_ub):
        """
        Inicializa a região de entrada (input box) para verificação.
        Define os limites da região onde a propriedade deve ser verificada.
        """
        low, high = x_lb, x_ub

        input_size = self.input_layer.layer_size

        # Ensure low is a vector
        low = np.array(low, dtype=np.float32) * np.ones(input_size, dtype=np.float32)
        high = np.array(high, dtype=np.float32) * np.ones(input_size, dtype=np.float32)

        self.input_layer.set_input_bounds(low, high)

        self.deepPolyNets_DNN.property_region = 1

        for i in range(self.deepPolyNets_DNN.layerSizes[0]):
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_lower = low[i]
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_upper = high[i]
            self.deepPolyNets_DNN.property_region *= (high[i] - low[i])
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_lower = np.array([low[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_upper = np.array([high[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].algebra_lower = np.array([low[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].algebra_upper = np.array([high[i]])

    # Conduct DeepPoly on DNN
    def symbolic_propagate(self):
        self.deepPolyNets_DNN.deeppoly()

        for i, l in enumerate(self.dense_layers):

            for out_index in range(l.layer_size):
                lb = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].concrete_lower_noClip
                ub = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].concrete_upper_noClip

                lb_clipped = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[
                    out_index].concrete_lower
                ub_clipped = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[
                    out_index].concrete_upper

                l.lb[out_index] = lb
                l.ub[out_index] = ub

                l.clipped_lb[out_index] = max(lb_clipped, 0)
                l.clipped_ub[out_index] = max(ub_clipped, 0)

                # record activation pattern for activation-based preimage computation method
                if self.preimg_mode == 'abstr' or self.preimg_mode == 'comp':
                    act_mode = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].actMode
                    l.actMode[out_index] = act_mode

        for out_index in range(self.output_layer.layer_size):
            lb = self.deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_lower_noClip
            ub = self.deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_upper_noClip
            self.output_layer.lb[out_index] = lb
            self.output_layer.ub[out_index] = ub
            logging.debug(f"self.output_layer.lb[{out_index}] = {lb}, self.output_layer.ub[{out_index}] = {ub}")

    # TODO: Design a composition method (Currently Not in use)
    # if abstraction-based method compute a relatively tight preimage, we turn to use MILP-based method
    def backward_preimage_computation_composed(self):
        cur_layer = self.output_layer
        in_layer_index = len(self.dense_layers)

        metric = 0.1
        # add input_layer's region constraint

        for in_layer in reversed(self.dense_layers):

            in_layer_index -= 1
            scaleValue = self.underPreImageMILP(in_layer_index, in_layer, cur_layer)

            if scaleValue == 0:  # abstraction of current layer is better, hence use abstr-based method
                scaleValue = self.underPreImageAbstr(in_layer_index, in_layer, cur_layer)

            self.scaleValueSet[in_layer.layer_index - 1] = scaleValue
            cur_layer = in_layer


    # Two backward preimage computation: MILP-based, abstraction-bassed
    def backward_preimage_computation(self):
        cur_layer = self.output_layer
        in_layer_index = len(self.dense_layers)

        for in_layer in reversed(self.dense_layers):

            in_layer_index -= 1
            scaleValue = 0

            if self.preimg_mode == 'milp' or self.preimg_mode == 'comp':
                scaleValue = self.underPreImageMILP(in_layer_index, in_layer, cur_layer)

            if self.preimg_mode == 'abstr' or (self.preimg_mode == 'comp' and scaleValue <= 0):
                scaleValue = self.underPreImageAbstr(in_layer_index, in_layer, cur_layer)

            self.scaleValueSet[in_layer.layer_index - 1] = scaleValue
            cur_layer = in_layer

    # MILP-based method for computing preimage
    def underPreImageMILP(self, in_layer_index, in_layer, cur_layer):
        enc_start_time = time.time()

        var_ll = []
        prop_cstr_ll = []
        model_cstr_ll = []
        w = cur_layer.layer_paras[0]
        b = cur_layer.layer_paras[1]

        relaxScale = self.gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS)

        relaxScale_LL = []
        relaxScale_LL.append(relaxScale)

        for in_index in range(in_layer.layer_size):
            neuron_val = in_layer.realVal[in_index]
            neuron_lb = in_layer.lb[in_index]
            neuron_ub = in_layer.ub[in_index]

            alpha_K = max(neuron_val - neuron_lb, 1e-3)
            beta_K = max(neuron_ub - neuron_val, 1e-3)

            # alpha_K = (neuron_ub - neuron_lb)/2
            # beta_K = (neuron_ub - neuron_lb)/2

            model_cstr_ll.append(
                self.gp_model.addConstr(in_layer.alpha[in_index] == (alpha_K * relaxScale)))
            model_cstr_ll.append(
                self.gp_model.addConstr(in_layer.beta[in_index] == (beta_K * relaxScale)))

            model_cstr_ll.append(
                self.gp_model.addConstr(
                    in_layer.ub[in_index] + beta_K * relaxScale >= in_layer.lb[in_index] - alpha_K * relaxScale))

            # get symbolic lower bounds w.r.t. input vars
            in_lb_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                in_index].concrete_algebra_lower
            in_ub_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                in_index].concrete_algebra_upper

            relaxed_lb_bias = in_lb_algebra[-1] - in_layer.alpha[in_index]
            relaxed_ub_bias = in_ub_algebra[-1] + in_layer.beta[in_index]

            # get symbolic upper bounds w.r.t. input vars
            symbolic_lb_expression = np.dot(in_lb_algebra[:-1], self.input_gp_vars)
            symbolic_lb_expression = symbolic_lb_expression + relaxed_lb_bias

            symbolic_ub_expression = np.dot(in_ub_algebra[:-1], self.input_gp_vars)
            symbolic_ub_expression = symbolic_ub_expression + relaxed_ub_bias

            model_cstr_ll.append(
                self.gp_model.addConstr(in_layer.gp_vars_before[in_index] <= symbolic_ub_expression))
            model_cstr_ll.append(
                self.gp_model.addConstr(in_layer.gp_vars_before[in_index] >= symbolic_lb_expression))

            model_cstr_ll.append(
                self.gp_model.addGenConstrMax(in_layer.gp_vars_after[in_index],
                                              [in_layer.gp_vars_before[in_index], 0]))

        self.gp_model.update()

        # encoding cur_layer's computation
        for out_index in range(cur_layer.layer_size):
            weights = w[out_index]
            bias = b[out_index]

            accumulation = np.dot(weights, in_layer.gp_vars_after)
            accumulation = accumulation + bias

            model_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_before[out_index] == accumulation))

        enc_finish_time = time.time()

        prop_start_time = time.time()

        # encoding cur_layer's property, compute concrete bounds
        if cur_layer.layer_index == (len(self.dense_layers) + 1):
            # output layer: not argmax
            other_vars = []
            for i in range(cur_layer.layer_size):
                if i == int(self.targetCls):
                    continue
                else:
                    other_vars.append(cur_layer.gp_vars_before[i])

            other_maximal = self.gp_model.addVar(lb=-1000, vtype=GRB.CONTINUOUS)
            prop_cstr_ll.append(self.gp_model.addGenConstrMax(other_maximal, other_vars))
            prop_cstr_ll.append(
                self.gp_model.addConstr(other_maximal >= cur_layer.gp_vars_before[self.targetCls] + self.tole))

        else:
            # other hidden layer: exist one neuron s.t. after relu, it is not included in the clipped_relaxed_ub
            bigM = 1000
            sumOfK = 0
            for i in range(cur_layer.layer_size):
                k_i_lb = self.gp_model.addVar(vtype=GRB.BINARY)
                relaxScale_LL.append(k_i_lb)

                prop_cstr_ll.append(self.gp_model.addConstr(
                    cur_layer.gp_vars_before[i] <= cur_layer.relaxed_lb_expression[i] - bigM * (
                            k_i_lb - 1) - 2 * self.tole))
                prop_cstr_ll.append(self.gp_model.addConstr(
                    cur_layer.gp_vars_before[i] >= cur_layer.relaxed_lb_expression[
                        i] - bigM * k_i_lb + 2 * self.tole))
                sumOfK = sumOfK + k_i_lb
                #
                # k_i encodes: is not included
                # for upper bounds
                k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                relaxScale_LL.append(k_i_ub)
                prop_cstr_ll.append(self.gp_model.addConstr(
                    cur_layer.gp_vars_before[i] >= cur_layer.relaxed_ub_expression[i] + bigM * (
                            k_i_ub - 1) + 2 * self.tole))
                prop_cstr_ll.append(self.gp_model.addConstr(
                    cur_layer.gp_vars_before[i] <= cur_layer.relaxed_ub_expression[
                        i] + bigM * k_i_ub - 2 * self.tole))

                sumOfK = sumOfK + k_i_ub

            prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= 1))

        prop_finish_time = time.time()
        prop_encoding_time = prop_finish_time - prop_start_time

        self.gp_model.update()
        self.gp_model.setObjective(relaxScale, GRB.MINIMIZE)
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

        # print("ifgpINF_OR_UNBD: ", ifgpINF_OR_UNBD)
        # print("ifgpINFEASIBLE: ", ifgpINFEASIBLE)
        # print("ifgpOptimize: ", ifgpOptimize)
        # print("ifgpUNBOUNDED: ", ifgpUNBOUNDED)

        scaleValue = -10000

        if ifgpOptimize:
            scaleValue = relaxScale.X
            print("\n########################### scaleValue for Relax layer index using MILP-based method: ",
                  in_layer_index, " is: ",
                  scaleValue, " ###########################")

            for in_index in range(in_layer.layer_size):
                alpha = in_layer.alpha[in_index].X
                beta = in_layer.beta[in_index].X

                relaxed_ub = in_layer.ub[in_index] + beta
                relaxed_lb = in_layer.lb[in_index] - alpha
                in_layer.relaxed_ub[in_index] = relaxed_ub
                in_layer.relaxed_lb[in_index] = relaxed_lb

                in_lb_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                    in_index].concrete_algebra_lower
                in_ub_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                    in_index].concrete_algebra_upper

                relaxed_lb_bias = in_lb_algebra[-1] - alpha
                relaxed_ub_bias = in_ub_algebra[-1] + beta

                # get symbolic upper bounds w.r.t. input vars
                relaxed_symbolic_lb_expression = np.dot(in_lb_algebra[:-1], self.input_gp_vars)
                relaxed_symbolic_lb_expression = relaxed_symbolic_lb_expression + relaxed_lb_bias

                relaxed_symbolic_ub_expression = np.dot(in_ub_algebra[:-1], self.input_gp_vars)
                relaxed_symbolic_ub_expression = relaxed_symbolic_ub_expression + relaxed_ub_bias

                in_layer.relaxed_lb_expression[in_index] = relaxed_symbolic_lb_expression
                in_layer.relaxed_ub_expression[in_index] = relaxed_symbolic_ub_expression

                if relaxed_ub <= 0:
                    in_layer.relaxed_ub_expression[in_index] = 0

            self.gp_model.remove(prop_cstr_ll)
            self.gp_model.remove(model_cstr_ll)
            self.gp_model.remove(relaxScale_LL)
            self.gp_model.remove(var_ll)
            self.gp_model.update()

        return scaleValue

    def underPreImageAbstr(self, in_layer_index, in_layer, cur_layer):

        enc_start_time = time.time()
        relaxScale_LL = []

        var_ll = []
        prop_cstr_ll = []
        model_cstr_ll = []
        w = cur_layer.layer_paras[0]
        b = cur_layer.layer_paras[1]

        relaxScale = self.gp_model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS)
        relaxScale_LL.append(relaxScale)

        # define relaxed_region for gp_vars_after (ReLU is approximated by abstract transformers)
        # We require that activation pattern remains unchanged in the preimage to avoid combinatorial explosion problem
        for in_index in range(in_layer.layer_size):

            neuron_val = in_layer.realVal[in_index]
            actMode = in_layer.actMode[in_index]

            neuron_lb = in_layer.lb[in_index]
            neuron_ub = in_layer.ub[in_index]

            if actMode == 1:
                alpha_K = neuron_val - neuron_lb
                beta_K = neuron_ub - neuron_val

                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.alpha_before[in_index] == (alpha_K * relaxScale)))
                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.beta_after[in_index] == (beta_K * relaxScale)))

                model_cstr_ll.append(self.gp_model.addGenConstrMin(in_layer.alpha_after[in_index],
                                                                   [in_layer.alpha_before[in_index],
                                                                    in_layer.lb[in_index]]))
            elif actMode == 2:
                continue
            else:  # actMode == 3 or 4:
                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.alpha_after[in_index] == (-neuron_lb * relaxScale)))
                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.beta_after[in_index] == (neuron_ub * relaxScale)))

        self.gp_model.update()

        # compute relaxed accumulated bounds instead of exactly encoding cur_layer's computation
        for out_index in range(cur_layer.layer_size):
            weights = w[out_index]
            tmp_add_lower = 0
            tmp_add_upper = 0

            # get new added biases
            for in_index in range(in_layer.layer_size):
                actMode = in_layer.actMode[in_index]
                if actMode == 1:
                    if weights[in_index] >= 0:
                        tmp_add_lower -= weights[in_index] * in_layer.alpha_after[in_index]
                        tmp_add_upper += weights[in_index] * in_layer.beta_after[in_index]
                    else:
                        tmp_add_lower += weights[in_index] * in_layer.beta_after[in_index]
                        tmp_add_upper -= weights[in_index] * in_layer.alpha_after[in_index]
                elif actMode == 2:
                    continue

                elif actMode == 3:
                    # update bounds
                    K = in_layer.ub[in_index] / (in_layer.ub[in_index] - in_layer.lb[in_index])
                    if weights[in_index] >= 0:
                        tmp_add_upper += weights[in_index] * K * (
                                in_layer.beta_after[in_index] + in_layer.alpha_after[in_index])
                    else:
                        tmp_add_lower += weights[in_index] * K * (
                                in_layer.beta_after[in_index] + in_layer.alpha_after[in_index])

                else:  # actMode == 4
                    K = in_layer.ub[in_index] / (in_layer.ub[in_index] - in_layer.lb[in_index])
                    if weights[in_index] >= 0:
                        tmp_add_lower -= weights[in_index] * in_layer.alpha_after[in_index]
                        tmp_add_upper += weights[in_index] * K * (
                                in_layer.beta_after[in_index] + in_layer.alpha_after[in_index])
                    else:
                        tmp_add_lower += weights[in_index] * K * (
                                in_layer.beta_after[in_index] + in_layer.alpha_after[in_index])
                        tmp_add_upper -= weights[in_index] * in_layer.alpha_after[in_index]

            model_cstr_ll.append(self.gp_model.addConstr(
                (tmp_add_lower + cur_layer.lb[out_index]) == cur_layer.gp_vars_lb_before[out_index]))
            model_cstr_ll.append(self.gp_model.addConstr(
                (tmp_add_upper + cur_layer.ub[out_index]) == cur_layer.gp_vars_ub_before[out_index]))

            self.gp_model.update()

        enc_finish_time = time.time()
        model_encoding_time = enc_finish_time - enc_start_time

        prop_start_time = time.time()

        # encoding cur_layer's property, the preimage propagated should be concluded in the next layer's preimage computed before
        if cur_layer.layer_index == (len(self.dense_layers) + 1):
            for var_index, var in enumerate(cur_layer.gp_vars_ub_before):
                if var_index == self.targetCls:
                    continue
                elif var_index < self.targetCls:
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_lb_before[self.targetCls] >= (var + 2 * self.tole)))
                else:
                    prop_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_lb_before[self.targetCls] >= var))
        else:
            for var_index, var in enumerate(cur_layer.gp_vars_lb_before):
                if cur_layer.actMode[var_index] == 1:
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_ub_before[var_index] <= cur_layer.relaxed_ub[var_index]))
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_lb_before[var_index] >= cur_layer.relaxed_lb[var_index]))
                elif cur_layer.actMode[var_index] == 2:
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_ub_before[var_index] <= 0))  # relaxed_ub>=0
                else:
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_ub_before[var_index] <= cur_layer.relaxed_ub[var_index]))  # relaxed_ub>=0
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_lb_before[var_index] >= cur_layer.relaxed_lb[var_index]))

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

        # print("ifgpINF_OR_UNBD: ", ifgpINF_OR_UNBD)
        # print("ifgpINFEASIBLE: ", ifgpINFEASIBLE)
        # print("ifgpOptimize: ", ifgpOptimize)
        # print("ifgpUNBOUNDED: ", ifgpUNBOUNDED)

        if ifgpOptimize:
            scaleValue = relaxScale.X
            print("\n########################### scaleValue for Relax layer index using abstraction-based method: ",
                  in_layer_index, " is: ",
                  scaleValue, " ###########################")

            for in_index in range(in_layer.layer_size):

                alpha_after = in_layer.alpha_after[in_index].X
                beta_after = in_layer.beta_after[in_index].X

                if in_layer.ub[in_index] <= 0:  # Case B
                    in_layer.relaxed_ub[in_index] = 0
                    in_layer.relaxed_lb[in_index] = in_layer.lb[in_index] - alpha_after

                else:  # Case A,C,D
                    in_layer.relaxed_ub[in_index] = np.float32(in_layer.ub[in_index] + beta_after)
                    in_layer.relaxed_lb[in_index] = np.float32(in_layer.lb[in_index] - alpha_after)

                in_lb_algebra = deepcopy(self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                                             in_index].concrete_algebra_lower)
                in_ub_algebra = deepcopy(self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                                             in_index].concrete_algebra_upper)

                relaxed_lb_bias = in_lb_algebra[-1] - alpha_after
                relaxed_ub_bias = in_ub_algebra[-1] + beta_after

                relaxed_symbolic_lb_expression = np.dot(in_lb_algebra[:-1], self.input_gp_vars)
                relaxed_symbolic_lb_expression = relaxed_symbolic_lb_expression + relaxed_lb_bias
                relaxed_symbolic_ub_expression = np.dot(in_ub_algebra[:-1], self.input_gp_vars)
                relaxed_symbolic_ub_expression = relaxed_symbolic_ub_expression + relaxed_ub_bias

                in_layer.relaxed_lb_expression[in_index] = relaxed_symbolic_lb_expression
                in_layer.relaxed_ub_expression[in_index] = relaxed_symbolic_ub_expression

                if in_layer.ub[in_index] <= 0:
                    in_layer.relaxed_ub_expression[in_index] = 0

            self.gp_model.remove(prop_cstr_ll)
            self.gp_model.remove(model_cstr_ll)
            self.gp_model.remove(relaxScale_LL)
            self.gp_model.remove(var_ll)
            self.gp_model.update()

            return scaleValue

    # forward quantization procedure
    def forward_quantization_with_esbmc(self):
        print("\nNow we begin ESBMC-based forward quantization!")
        qu_list = []
        qu_frac_list = []
        qu_int_list = []

        nonInputLayers = self.dense_layers.copy()
        nonInputLayers.append(self.output_layer)
        in_layer_index = -1
        previous_layer = self.input_layer
        #.set_trace()
        for cur_layer in nonInputLayers:

            logging.info(f"This neural network has {nonInputLayers} hidden layers and the current in_layer_index is {cur_layer}")
            in_layer_index += 1
            
            if cur_layer.layer_index == 1:
                in_layer = self.input_layer
            else:
                in_layer = self.dense_layers[cur_layer.layer_index - 2]

            w = cur_layer.layer_paras[0]
            b = cur_layer.layer_paras[1]

            lower_bit = self.bit_lb
            upper_bit = self.bit_ub
            ifFound = False
            for rela_bit in range(upper_bit - lower_bit + 1):
                if ifFound:
                    break

                frac_bit = rela_bit + lower_bit
                int_bit = cur_layer.int_bit
                all_bit = frac_bit + int_bit

                # Get quantized parameters (same as original)
                logging.debug(f"Weights before quantization: {w}")
                qu_w_int = quantize_int(w, all_bit, frac_bit) #/ (2 ** frac_bit)
                qu_b_int = quantize_int(b, all_bit, frac_bit) #/ (2 ** frac_bit)
                logging.debug(f"Quantized weights: {qu_w_int}")
                logging.debug(f"Quantized biases: {qu_b_int}")
                  # Para uso posterior (DeepPoly), converte de volta para float
                qu_w_float = qu_w_int / (2 ** frac_bit)
                qu_b_float = qu_b_int / (2 ** frac_bit)

                logging.debug(f"about to gen the C code for ESBMC verification for Layer {cur_layer.layer_index} with Q={all_bit}, F={frac_bit}")
                logging.debug(f"cur_layer: {cur_layer}, in_layer: {in_layer}, qu_w: {qu_w_float}, qu_b: {qu_b_float}, frac_bit: {frac_bit}, all_bit: {all_bit}, in_layer_index: {in_layer_index}")
                # === ESBMC VERIFICATION REPLACEMENT ===
                # Instead of Gurobi optimization, call ESBMC

                log_parameters.info(f"===========================================")
                log_parameters.info(f"ESBMC Verification for Layer {cur_layer.layer_index-1}")
                log_parameters.debug(f"For the layer {cur_layer.layer_index-1}, preimage bounds are: {cur_layer.lb}")
                # Calculate values without scaling

                esbmc_result = self.verify_layer_with_esbmc(
                    cur_layer, in_layer, qu_w_int, qu_b_int, 
                    frac_bit, all_bit, in_layer_index
                )
                
                
                if esbmc_result == "VERIFIED":
                    print(f"ESBMC verified quantization [Q={all_bit}, F={frac_bit}] for Layer {cur_layer.layer_index}")
                    
                    cur_layer.frac_bit = frac_bit
                    qu_frac_list.append(cur_layer.frac_bit)
                    qu_int_list.append(cur_layer.int_bit)
                    qu_list.append(all_bit)
                    ifFound = True
                    last_layer = cur_layer

                    # Update weights and algebra (same as original)
                    self.update_quantized_weights_affine(in_layer, cur_layer, all_bit, frac_bit, frac_bit, in_layer_index)
                    

            if not ifFound:
                print(f"ESBMC cannot verify any quantization for layer {cur_layer.layer_index}")
                return False, None, None, None

        return True, qu_list, qu_frac_list, qu_int_list
    
    def verify_layer_with_esbmc(self, cur_layer, in_layer, qu_w, qu_b, frac_bit, all_bit, layer_index):
        """Replace Gurobi optimization with ESBMC verification"""
        
        # Generate ESBMC verification code for this layer
        esbmc_code = self.generate_esbmc_verification_code(
            cur_layer, in_layer, qu_w, qu_b, frac_bit, all_bit, layer_index
        )
        if esbmc_code is None:
            print("Output layer ESBMC verification code generation not implemented.")
            return "UNVERIFIED"

        # save layer in ouput/layers
        with open(f"output/layers/layer_{layer_index}_Q{all_bit}_F{frac_bit}.c", 'w') as f:
            f.write(esbmc_code)
            f.close()
        # Write to temporary file
        temp_file = f"esbmc_verify_layer_{layer_index}_Q{all_bit}_F{frac_bit}.c"
        with open(temp_file, 'w') as f:
            f.write(esbmc_code)

        # Run ESBMC
        result = self.run_esbmc_verification(temp_file, layer_index)
        
        # Clean up
        import os
        os.remove(temp_file)
        
        return result
    
    def generate_esbmc_verification_code(self, cur_layer, in_layer, qu_w_int, qu_b_int,
                                        frac_bit, all_bit, layer_index):
        """
        Gera código C para verificação ESBMC usando aritmética de ponto fixo (inteiros).
        
        Estratégia de ponto fixo:
        - Representa cada valor v como V = round(v * SCALE), onde SCALE = 2^frac_bit
        - Camadas ocultas: verifica se saída afim fica dentro do intervalo de pré-imagem (escalado)
        - Camada de saída: verifica se classe alvo é estritamente maximal (saídas escaladas)
        """
        # ==================== CONFIGURAÇÃO DO FATOR DE ESCALA ====================
        # Calcula o fator de escala para conversão para ponto fixo
        SCALE = 1 << int(frac_bit)  # SCALE = 2^frac_bit

        # ==================== QUANTIZAÇÃO DOS PARÂMETROS PARA INTEIROS ====================
        # Converte pesos e biases para representação de ponto fixo (inteiros)
        weights_c_int = self.numpy_to_c_int_array(qu_w_int)  # ← Já é inteiro!
        biases_c_int  = self.numpy_to_c_int_array(qu_b_int)  # ← Já é inteiro!
        
        # ==================== CONFIGURAÇÃO DOS BOUNDS DE PRÉ-IMAGEM ====================
        # Usa bounds relaxados se disponíveis, senão usa bounds concretos
        # Escala conservadoramente para evitar problemas de precisão
        if hasattr(cur_layer, "relaxed_lb") and hasattr(cur_layer, "relaxed_ub") and \
           cur_layer.relaxed_lb is not None and cur_layer.relaxed_ub is not None:
            # Usa bounds relaxados (computados pela análise de pré-imagem)
            pre_lo = np.array(cur_layer.relaxed_lb, dtype=np.float64)
            pre_hi = np.array(cur_layer.relaxed_ub, dtype=np.float64)
        else:
            # Usa bounds concretos (obtidos via DeepPoly)
            pre_lo = np.array(cur_layer.lb, dtype=np.float64)
            pre_hi = np.array(cur_layer.ub, dtype=np.float64)
            
        # Converte para inteiros de forma conservadora (floor/ceil para ampliar intervalo)
        pre_lo_int = np.floor(pre_lo * SCALE).astype(np.int64)  # Floor para limite inferior
        pre_hi_int = np.ceil(pre_hi * SCALE).astype(np.int64)   # Ceil para limite superior

        # Converte para strings C
        preimage_low_c_int  = self.numpy_to_c_int_array(pre_lo_int)
        preimage_high_c_int = self.numpy_to_c_int_array(pre_hi_int)

        # ==================== CONFIGURAÇÃO DOS BOUNDS DE ENTRADA ====================
        # Converte bounds da região de entrada para inteiros escalados
        # Alarga ligeiramente para ser conservativo
        if cur_layer.layer_index == 1:
            # Primeira camada: usa bounds da entrada original
            x_lo = np.array(self.x_low_real, dtype=np.float64)  # Limite inferior da entrada
            x_hi = np.array(self.x_high_real, dtype=np.float64)  # Limite superior da entrada
        else:
            x_lo = np.array(in_layer.clipped_lb, dtype=np.float64)        # Limite inferior da entrada
            x_hi = np.array(in_layer.clipped_ub, dtype=np.float64)       # Limite superior da entrada
    

        x_lo_int = np.floor(x_lo * SCALE).astype(np.int64)        # Floor para ser conservativo
        x_hi_int = np.ceil(x_hi * SCALE).astype(np.int64)         # Ceil para ser conservativo
        input_bounds_low_int  = self.numpy_to_c_int_array(x_lo_int)
        input_bounds_high_int = self.numpy_to_c_int_array(x_hi_int)

        targetCls = self.targetCls

        # ==================== SELEÇÃO DO TEMPLATE DE VERIFICAÇÃO ====================
        # Determina se é camada de saída ou camada oculta
        is_output_layer = (cur_layer.layer_index == len(self.dense_layers) + 1)
        if is_output_layer:

            # ==================== PRIMEIRA CAMADA OCULTA ====================
            # Verifica se saída afim fica dentro da pré-imagem relaxada
            return outerlayer_fixed_int_multiclass(in_layer.layer_size,
                cur_layer.layer_size,   # Tamanhos das camadas
                weights_c_int, biases_c_int,                 # Parâmetros quantizados
                input_bounds_low_int, input_bounds_high_int, [0,1,2], # Bounds da região de entrada
                SCALE )  
            return outerlayer_fixed_int(in_layer.layer_size,
                cur_layer.layer_size,   # Tamanhos das camadas
                weights_c_int, biases_c_int,                 # Parâmetros quantizados
                input_bounds_low_int, input_bounds_high_int, targetCls, # Bounds da região de entrada
                SCALE )                                       # Fator de escala

        return innerlayer_fixed_int_bounds_only(
                cur_layer.layer_size, in_layer.layer_size,   # Tamanhos das camadas
                weights_c_int, biases_c_int,                 # Parâmetros quantizados
                preimage_low_c_int, preimage_high_c_int,     # Pré-imagem relaxada
                input_bounds_low_int, input_bounds_high_int, # Bounds da região de entrada
                SCALE                                        # Fator de escala
            )
        if is_output_layer:
            
            # ==================== CAMADA DE SAÍDA ====================
            # Verifica propriedade de classificação: classe alvo deve ser máxima
            return None
            return outerlayer_fixed_int(
                in_layer.layer_size, cur_layer.layer_size,   # Tamanhos das camadas
                weights_c_int, biases_c_int,                 # Parâmetros quantizados
                input_bounds_low_int, input_bounds_high_int, # Bounds da região de entrada
                self.targetCls, SCALE                        # Classe alvo e fator de escala
            )
        else:
            # ==================== CAMADA OCULTA ====================
            # Verifica se saída afim fica dentro da pré-imagem relaxada
            return innerlayer_fixed_int_bounds_only(
                cur_layer.layer_size, in_layer.layer_size,   # Tamanhos das camadas
                weights_c_int, biases_c_int,                 # Parâmetros quantizados
                preimage_low_c_int, preimage_high_c_int,     # Pré-imagem relaxada
                input_bounds_low_int, input_bounds_high_int, # Bounds da região de entrada
                SCALE                                        # Fator de escala
            )


    def numpy_to_c_array(self, np_array):
        """
        Converte array numpy para string de inicialização C (ponto flutuante).
        Usado para gerar código C com valores em ponto flutuante.
        """
        if np_array.ndim == 1:
            # Array 1D: {val1, val2, val3}
            return "{" + ", ".join([f"{x:.6f}f" for x in np_array]) + "}"
        else:
            # Array 2D: {{row1}, {row2}, {row3}}
            rows = []
            for row in np_array:
                rows.append("{" + ", ".join([f"{x:.6f}f" for x in row]) + "}")
            return "{" + ", ".join(rows) + "}"
            
    def numpy_to_c_int_array(self, np_array):
        """
        Converte array numpy para string de inicialização C (inteiros).
        Usado para gerar código C com valores inteiros (ponto fixo).
        """
        if np_array.ndim == 1:
            # Array 1D: {val1, val2, val3}
            return "{" + ", ".join([str(int(x)) for x in np_array]) + "}"
        else:
            # Array 2D: {{row1}, {row2}, {row3}}
            rows = []
            for row in np_array:
                rows.append("{" + ", ".join([str(int(x)) for x in row]) + "}")
            return "{" + ", ".join(rows) + "}"
        

    def update_quantized_weights_affine(self, in_layer, out_layer, num_bit, frac_bit_weights, frac_bit_bias,
                                        in_layer_index):
        """
        Atualiza os pesos quantizados no modelo DeepPoly.
        Aplica quantização de ponto fixo aos parâmetros da camada.
        """
        # ==================== CÁLCULO DOS LIMITES DE QUANTIZAÇÃO ====================
        # Obtém os valores mínimos e máximos representáveis com a quantização especificada
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
            self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[out_index].weight = weight_row_fp
            self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[out_index].bias = bias_fp

            # update algebra parameters for affine layer
            self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[out_index].algebra_lower = np.append(
                weight_row_fp, [bias_fp])
            self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[out_index].algebra_upper = np.append(
                weight_row_fp, [bias_fp])

    def write_result(self, qu_frac_list, fileName):
        real_qu_list = []
        frac_qu_list = []
        int_qu_list = []
        for i, l in enumerate(self.dense_layers):
            real_qu = qu_frac_list[i] + self.dense_layers[i].int_bit
            real_qu_list.append(real_qu)
            frac_qu_list.append(qu_frac_list[i])
            int_qu_list.append(self.dense_layers[i].int_bit)

        real_qu_list.append(qu_frac_list[-1] + self.output_layer.int_bit)
        frac_qu_list.append(qu_frac_list[-1])
        int_qu_list.append(self.output_layer.int_bit)

        print("\n******************** The quantization strategy holds the property ********************\n")
        fo = open(fileName, "w")
        fo.write("Solving Result: True\n")
        fo.write("We found a quantization strategy to hold the robustness property.\n")
        fo.write("The all quantization bit sizes for each layer are:" + str(real_qu_list) + "\n")
        fo.write("The frac quantization bit sizes for each layer are:" + str(frac_qu_list) + "\n")
        fo.write("The int quantization bit sizes for each layer are:" + str(int_qu_list) + "\n")
        fo.write("Backward Time: " + str(self._stats["backward_time"]) + "\n")
        fo.write("Forward Time: " + str(self._stats["forward_time"]) + "\n")
        fo.write("Total Time: " + str(self._stats["total_time"]) + "\n")
        numAllVars = self.gp_model.getAttr("NumVars")
        numIntVars = self.gp_model.getAttr("NumIntVars")
        numBinVars = self.gp_model.getAttr("NumBinVars")
        numConstrs = self.gp_model.getAttr("NumConstrs")
        #
        fo.write("The num of vars: " + str(numAllVars) + "\n")
        fo.write("The num of numIntVars: " + str(numIntVars) + "\n")
        fo.write("The num of numBinVars: " + str(numBinVars) + "\n")
        fo.write("The num of Constraints: " + str(numConstrs) + "\n")
        fo.close()

    def run_esbmc_verification(self, c_file, layer_index):
        """
        Executa o verificador ESBMC e analisa os resultados.
        
        ESBMC (Efficient SMT-based Bounded Model Checker) é usado para verificar
        se a propriedade de quantização é preservada em cada camada.
        """


        # Configuração otimizada para verificação de redes neurais
        import subprocess
        import re
        
        # ==================== CONFIGURAÇÃO DO COMANDO ESBMC ====================
        # Importante: garantir unwind suficiente para a prova ser sound.
        # Inferimos INPUT_SIZE/LAYER_SIZE do código C gerado e usamos --unwind >= max(iter).
        unwind = 0
        try:
            with open(c_file, "r", encoding="utf-8", errors="ignore") as f:
                csrc = f.read()
            m_in = re.search(r"#define\s+INPUT_SIZE\s+(\d+)", csrc)
            m_la = re.search(r"#define\s+LAYER_SIZE\s+(\d+)", csrc)
            if m_in:
                unwind = max(unwind, int(m_in.group(1)))
            if m_la:
                unwind = max(unwind, int(m_la.group(1)))
        except Exception:
            unwind = 0
        unwind = max(unwind, 1) + 1
        esbmc_cmd = [
            "esbmc", c_file,                    # Arquivo C a ser verificado
            "--loop-invariant",                 # Usa invariantes de loop para melhor verificação
            "--function", "main",
            "--interval-analysis",
            "--unwind", str(unwind),
            "--incremental-bmc",               # BMC incremental para melhor performance
            "--state-hashing",                 # Hashing de estados para reduzir exploração
            "--force-malloc-success",          # Assume que malloc sempre sucede
            "--no-bounds-check",               # Desabilita verificação de bounds de arrays
            "--no-div-by-zero-check",          # Desabilita verificação de divisão por zero
            "--no-pointer-check",              # Desabilita verificação de ponteiros
            "--timeout", "900",                # Timeout de 15 minutos por verificação
            "--verbosity", "10"                # Nível máximo de verbosidade para debug
            , "--print-stack-traces",             # Imprime stack traces para facilitar debug
            "--ir",                         # Análise de intervalos para otimização
        ]
        
        try:
            # ==================== EXECUÇÃO DO ESBMC ====================
            print(f"Running ESBMC for layer {layer_index}...")
            result = subprocess.run(
                esbmc_cmd,                     # Comando e argumentos
                stdout=subprocess.PIPE,        # Captura saída padrão
                stderr=subprocess.PIPE,        # Captura saída de erro
                text=True,                     # Decodifica como texto
                timeout=1200,                  # Timeout de 20 minutos (safety margin)
                encoding="utf-8",             # Codificação de caracteres
                errors="replace",             # Substitui caracteres inválidos
            )            
            # ==================== DEBUG E LOG DA EXECUÇÃO ====================
            # Imprime informações de debug para monitoramento
            print(f"Result stdout: {result.stdout[-500:]}")  # Últimos 500 chars para debug
            print("ESBMC return code:", result.returncode)
            print("--- STDOUT (tail) ---\n", (result.stdout or "")[-8000:])
            print("--- STDERR (tail) ---\n", (result.stderr or "")[-8000:])
            log_esbmc.debug(f"Result stdout: {result.stdout[-5000:]}")  # Últimos 500 chars para debug
            log_esbmc.debug(f"ESBMC return code: {result.returncode}")
            log_esbmc.debug(f"--- STDOUT (tail) ---\n {(result.stdout or '')[-20000:]}")
            log_esbmc.debug(f"--- STDERR (tail) ---\n {(result.stderr or '')[-20000:]}")
        

            # ==================== ANÁLISE DOS RESULTADOS ====================
            # Parseia a saída do ESBMC para determinar o resultado da verificação
            if "VERIFICATION SUCCESSFUL" in result.stderr:
                print(f"ESBMC verification PASSED for layer {layer_index}")
                return "VERIFIED"                          # Propriedade verificada com sucesso
            elif "VERIFICATION FAILED" in result.stdout:
                print(f"ESBMC verification FAILED for layer {layer_index}")
                # Extrai contraexemplo se disponível
                if "Counterexample:" in result.stdout:
                    print("Counterexample found - quantization violates preimage")
                return "FAILED"                           # Propriedade violada
            else:
                print(f"ESBMC verification UNKNOWN for layer {layer_index}")
                print(f"ESBMC output: {result.stdout[-500:]}")  # Últimos 500 chars
                return "UNKNOWN"                          # Resultado inconclusivo
                
        except subprocess.TimeoutExpired:
            # ==================== TRATAMENTO DE TIMEOUT ====================
            print(f"ESBMC timeout for layer {layer_index}")
            return "TIMEOUT"                              # Timeout na verificação
        except Exception as e:
            # ==================== TRATAMENTO DE ERROS ====================
            print(f"ESBMC error for layer {layer_index}: {e}")
            return "ERROR"                                # Erro na execução
        
    
    # forward quantization procedure
    def forward_quantization(self):
        print("\nNow we begin to do the forward quantization!")
        qu_list = []
        qu_frac_list = []
        qu_int_list = []

        nonInputLayers = self.dense_layers.copy()
        nonInputLayers.append(self.output_layer)

        in_layer_index = -1

        for cur_layer in nonInputLayers:
            in_layer_index += 1

            if cur_layer.layer_index == 1:
                in_layer = self.input_layer
            else:
                in_layer = self.dense_layers[cur_layer.layer_index - 2]

            w = cur_layer.layer_paras[0]
            b = cur_layer.layer_paras[1]

            # test for all bit:
            lower_bit = self.bit_lb
            upper_bit = self.bit_ub

            ifFound = False

            for rela_bit in range(upper_bit - lower_bit + 1):
                pre_mul_qu_lb_deepPoly = []
                pre_mul_qu_ub_deepPoly = []

                if ifFound:
                    break

                model_cstr_ll = []
                prop_cstr_ll = []
                var_ll = []

                frac_bit = rela_bit + lower_bit
                int_bit = cur_layer.int_bit
                all_bit = frac_bit + int_bit

                # get_quantized_paras
                qu_w = quantize_int(w, all_bit, frac_bit) / (2 ** frac_bit)
                qu_b = quantize_int(b, all_bit, frac_bit) / (2 ** frac_bit)

                # for last layer
                target_lb = 0
                other_ubs = []

                # quantized_concrete_algebra_lower
                quantized_concrete_algebra_lower = []
                quantized_concrete_algebra_upper = []

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
                        quantized_concrete_algebra_lower.append(cur_neuron_concrete_algebra_lower)
                        quantized_concrete_algebra_upper.append(cur_neuron_concrete_algebra_upper)

                    # reverse, from cur_layer's affine layer to input layer
                    for kk in range(2 * (in_layer_index + 1) - 1)[::-1]:
                        # size of input
                        tmp_lower = np.zeros(len(self.deepPolyNets_DNN.layers[kk].neurons[0].algebra_lower))
                        tmp_upper = np.zeros(len(self.deepPolyNets_DNN.layers[kk].neurons[0].algebra_lower))

                        assert (self.deepPolyNets_DNN.layers[kk].size + 1 == len(lower_bound))
                        assert (self.deepPolyNets_DNN.layers[kk].size + 1 == len(upper_bound))

                        for pp in range(self.deepPolyNets_DNN.layers[kk].size):
                            if lower_bound[pp] >= 0:
                                tmp_lower += np.float32(
                                    lower_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[
                                        pp].algebra_lower)
                            else:
                                tmp_lower += np.float32(
                                    lower_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[
                                        pp].algebra_upper)

                            if upper_bound[pp] >= 0:
                                tmp_upper += np.float32(
                                    upper_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[
                                        pp].algebra_upper)
                            else:
                                tmp_upper += np.float32(
                                    upper_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[
                                        pp].algebra_lower)

                        tmp_lower[-1] += lower_bound[-1]
                        tmp_upper[-1] += upper_bound[-1]
                        lower_bound = deepcopy(tmp_lower)
                        upper_bound = deepcopy(tmp_upper)
                        #
                        if kk == 1:
                            cur_neuron_concrete_algebra_lower = deepcopy(lower_bound)
                            cur_neuron_concrete_algebra_upper = deepcopy(upper_bound)
                            quantized_concrete_algebra_lower.append(cur_neuron_concrete_algebra_lower)
                            quantized_concrete_algebra_upper.append(cur_neuron_concrete_algebra_upper)

                    assert (len(lower_bound) == 1)
                    assert (len(upper_bound) == 1)

                    cur_neuron_concrete_lower = lower_bound[0]
                    cur_neuron_concrete_upper = upper_bound[0]

                    tmp_acc_lower += cur_neuron_concrete_lower
                    tmp_acc_upper += cur_neuron_concrete_upper

                    pre_mul_qu_lb_deepPoly.append(tmp_acc_lower)
                    pre_mul_qu_ub_deepPoly.append(tmp_acc_upper)

                    # generate property constraints
                    # get quantized_ub_expression from backward-procedure
                    quantized_lb_expression = np.dot(cur_neuron_concrete_algebra_lower[:-1],
                                                     self.input_gp_vars)
                    quantized_lb_expression = quantized_lb_expression + cur_neuron_concrete_algebra_lower[
                        -1]

                    quantized_ub_expression = np.dot(cur_neuron_concrete_algebra_upper[:-1],
                                                     self.input_gp_vars)
                    quantized_ub_expression = quantized_ub_expression + cur_neuron_concrete_algebra_upper[
                        -1]

                    # either lower or higher
                    if cur_layer.layer_index == (len(self.dense_layers) + 1):
                        if out_index == self.targetCls:
                            target_lb = quantized_lb_expression
                        else:
                            other_ubs.append(quantized_ub_expression)
                    else:
                        k_i_lb = self.gp_model.addVar(vtype=GRB.BINARY)
                        var_ll.append(k_i_lb)

                        if cur_layer.relaxed_ub[out_index] > 0:
                            prop_cstr_ll.append(self.gp_model.addConstr(
                                quantized_lb_expression <= cur_layer.relaxed_lb_expression[out_index] - 1000 * (
                                        k_i_lb - 1) - self.tole))
                            prop_cstr_ll.append(self.gp_model.addConstr(
                                quantized_lb_expression >= cur_layer.relaxed_lb_expression[
                                    out_index] - 1000 * k_i_lb + self.tole))
                            sumOfK = sumOfK + k_i_lb
                            numOfK += 1

                        # k_i encodes: is not included
                        # for upper bounds
                        k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                        var_ll.append(k_i_ub)
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            quantized_ub_expression >= cur_layer.relaxed_ub_expression[out_index] + 1000 * (
                                    k_i_ub - 1) + self.tole))
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            quantized_ub_expression <= cur_layer.relaxed_ub_expression[
                                out_index] + 1000 * k_i_ub - self.tole))

                        numOfK += 1
                        sumOfK = sumOfK + k_i_ub

                if len(other_ubs) > 0:  # output layer
                    # k_i encodes: is not included
                    # for upper bounds
                    for other_single_ub in other_ubs:
                        k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                        var_ll.append(k_i_ub)
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            other_single_ub >= target_lb + 1000 * (
                                    k_i_ub - 1) + self.tole))
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            other_single_ub <= target_lb + 1000 * k_i_ub - self.tole))

                        sumOfK = sumOfK + k_i_ub
                        numOfK += 1

                # for relaxed version of Quadapter
                if len(other_ubs) == 0 and self.ifRelax == 1:

                    scale = 0.25

                    # # for better performance, can try this relaxation
                    # if self.scaleValueSet[in_layer_index] <= 0.01 and in_layer_index > 0:
                    #     scale = 0.35

                    prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= int(numOfK * scale) + 1))
                else:
                    prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= 1))

                self.gp_model.update()
                self.gp_model.setParam('DualReductions', 0)

                self.gp_model.optimize()

                ifgpUNSat = self.gp_model.status == GRB.INFEASIBLE

                if ifgpUNSat:
                    print("We find a quantization configuration [ Q , F ] for the Layer", cur_layer.layer_index,
                          "as: [", all_bit, ",", frac_bit, '].')

                    cur_layer.frac_bit = frac_bit

                    qu_frac_list.append(cur_layer.frac_bit)
                    qu_int_list.append(cur_layer.int_bit)
                    qu_list.append(all_bit)

                    ifFound = True

                    self.gp_model.remove(model_cstr_ll)
                    self.gp_model.remove(prop_cstr_ll)

                    self.gp_model.remove(var_ll)

                    self.gp_model.update()

                    self.update_quantized_weights_affine(in_layer, cur_layer, all_bit, frac_bit, frac_bit,
                                                         in_layer_index)

                    # if hidden layer, then update next relu's algebra for the abstract element cf. DeepPoly
                    if cur_layer.layer_index < (len(self.dense_layers) + 1):
                        for out_index in range(cur_layer.layer_size):
                            lb_new = pre_mul_qu_lb_deepPoly[out_index]
                            ub_new = pre_mul_qu_ub_deepPoly[out_index]
                            cur_neuron = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1)].neurons[out_index]
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
                        self.output_layer.qu_lb = pre_mul_qu_lb_deepPoly
                        self.output_layer.qu_ub = pre_mul_qu_ub_deepPoly
            if not ifFound:
                print("Cannot find a quantization strategy for the cur_layer with index as: ", cur_layer.layer_index)
                return False, None, None, None

        return True, qu_list, qu_frac_list, qu_int_list
