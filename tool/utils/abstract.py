def outerlayer_fixed_int(in_layer_layer_size, cur_layer_layer_size, weights_c_int, biases_c_int,
                         input_bounds_low_int, input_bounds_high_int, targetCls, scale_factor):

    return f"""\
#include <stdint.h>
#include <limits.h>


#define INPUT_SIZE   {in_layer_layer_size}
#define LAYER_SIZE   {cur_layer_layer_size}
#define TARGET_CLASS {targetCls}
#define SCALE_FACTOR {scale_factor}LL

extern long long nondet_longlong(void);

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE]              = {biases_c_int};

long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

static inline long long llabs(long long x) {{
    return x < 0LL ? -x : x;
}}

/* Transformacao afim em ponto fixo: out[i] = (W[i] · in + b[i]) / SCALE */
static void affine_transform_fixed(const long long in_[INPUT_SIZE], long long out_[LAYER_SIZE])
{{
    for (int i = 0; i < LAYER_SIZE; ++i) {{
        long long acc = 0LL; /* acumulador em SCALE_FACTOR^2 */
        
        for (int j = 0; j < INPUT_SIZE; ++j) {{
            /* w * x ambos em SCALE_FACTOR → produto em SCALE_FACTOR^2 */
            long long prod = weights[i][j] * in_[j];
            acc += prod;
        }}
        
        /* Rescale para SCALE_FACTOR e adiciona bias (já em SCALE_FACTOR) */
        out_[i] = (acc / SCALE_FACTOR) + biases[i];
    }}
}}

/* verifica se o output condiz a classe esperada */
static int verify_classification(const long long out_[LAYER_SIZE])
{{
    const int T = TARGET_CLASS;
    const long long target = out_[T];
    long long max_other = LLONG_MIN / 2;
    
    int i = 0;
    
    while (i < LAYER_SIZE)
    {{
        __ESBMC_loop_invariant(0 <= i && i <= LAYER_SIZE && max_other <= target);
        if (i != T) {{
            const long long cand = out_[i];
            if (cand > max_other) {{
                max_other = cand;
            }}
        }}
        ++i;
    }}
    
    return max_other < target;
}}

int main(void)
{{
    long long input[INPUT_SIZE];
    long long output[LAYER_SIZE];
    
    /* Entrada nao-deterministica */
    for (int k = 0; k < INPUT_SIZE; ++k) {{
        input[k] = nondet_longlong();
        __ESBMC_assume(input[k] >= input_bounds_low[k] && 
                       input[k] <= input_bounds_high[k]);
    }}
    
    affine_transform_fixed(input, output);
    
    __ESBMC_assert(verify_classification(output),
                   "Classification property violated (output layer, fixed-point)");
    
    return 0;
}}
"""


def innerlayer_fixed_int(cur_layer_layer_size, in_layer_layer_size, weights_c_int, biases_c_int,
                         preimage_low_int, preimage_high_int, input_bounds_low_int, input_bounds_high_int,
                         scale_factor):
    return f"""\
#include <stdint.h>
#include <limits.h>


#define INPUT_SIZE   {in_layer_layer_size}
#define LAYER_SIZE   {cur_layer_layer_size}
#define SCALE_FACTOR {scale_factor}LL

extern long long nondet_longlong(void);

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE]              = {biases_c_int};

long long preimage_low[LAYER_SIZE]  = {preimage_low_int};
long long preimage_high[LAYER_SIZE] = {preimage_high_int};

long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

static inline long long llabs(long long x) {{
    return x < 0LL ? -x : x;
}}

/* Camada afim em ponto fixo sobre um box de entrada: s_lb <= s_out <= s_ub */
static void check_affine_bounds_fixed(const long long in_[INPUT_SIZE])
{{
    /* tolerancia para preimagem */
    const long long abs_tol = (long long)(1e-3 * SCALE_FACTOR);
    const long long rel_tol_num = 1; /* 1% = 1/100 */
    const long long rel_tol_den = 100;
    
    for (int i = 0; i < LAYER_SIZE; ++i) {{
        long long s_out = 0LL;  /* saída exata na entrada atual */
        long long s_lb  = 0LL;  /* limite inferior usando box */
        long long s_ub  = 0LL;  /* limite superior usando box */
        
        /* Tolerance ao redor do intervalo de preimagem */
        const long long pre_lo = preimage_low[i];
        const long long pre_hi = preimage_high[i];
        const long long range = llabs(pre_hi - pre_lo);
        const long long eps = abs_tol + (rel_tol_num * range) / rel_tol_den;
        
        int j = 0;
        
        while (j < INPUT_SIZE)
        {{
            //__ESBMC_loop_invariant(0 <= j && j <= INPUT_SIZE &&
            //                       s_lb <= s_out && s_out <= s_ub);
            
            const long long w  = weights[i][j];
            const long long lo = input_bounds_low[j];
            const long long hi = input_bounds_high[j];
            
            /* Passo exato na entrada nao determinística */
            s_out += w * in_[j];
            
            /* Contribuição baseada no sinal (imagem do box) */
            const long long cmin = (w >= 0LL) ? (w * lo) : (w * hi);
            const long long cmax = (w >= 0LL) ? (w * hi) : (w * lo);
            
            s_lb += cmin;
            s_ub += cmax;
            
            ++j;
        }}
        
        /* Rescale e adiciona bias */
        s_out = (s_out / SCALE_FACTOR) + biases[i];
        s_lb  = (s_lb  / SCALE_FACTOR) + biases[i];
        s_ub  = (s_ub  / SCALE_FACTOR) + biases[i];
        
        /* verifica se a saida esta dentro da preimagem esperada */
        __ESBMC_assert(s_out >= pre_lo - eps && s_out <= pre_hi + eps,
                       "affine output not within tolerated preimage");
    }}
}}

int main(void)
{{
    long long in_[INPUT_SIZE];
    
    /* Entrada nao deterministica dentro do conjunto de entrada */
    for (int j = 0; j < INPUT_SIZE; ++j) {{
        in_[j] = nondet_longlong();
        __ESBMC_assume(in_[j] >= input_bounds_low[j] && 
                       in_[j] <= input_bounds_high[j]);
    }}
    
    check_affine_bounds_fixed(in_);
    
    return 0;
}}
"""


def outerlayer_fixed_int_multiclass(in_layer_layer_size, cur_layer_layer_size, weights_c_int, biases_c_int,
                                   input_bounds_low_int, input_bounds_high_int, valid_classes, scale_factor):
    
    valid_classes_array = "{" + ", ".join(map(str, valid_classes)) + "}"
    num_valid_classes = len(valid_classes)
    
    return f"""\
#include <stdint.h>
#include <limits.h>
#include <stdbool.h>


#define INPUT_SIZE       {in_layer_layer_size}
#define LAYER_SIZE       {cur_layer_layer_size}
#define NUM_VALID_CLASSES {num_valid_classes}
#define SCALE_FACTOR     {scale_factor}LL

extern long long nondet_longlong(void);

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE]              = {biases_c_int};

long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

int valid_classes[NUM_VALID_CLASSES] = {valid_classes_array};

/* Verifica se uma classe está no conjunto de classes válidas */
static bool is_valid_class(int class_id) {{
    for (int i = 0; i < NUM_VALID_CLASSES; ++i) {{
        if (valid_classes[i] == class_id) {{
            return true;
        }}
    }}
    return false;
}}

/* Transformacao na funcao afim em ponto fixo */
static void affine_transform_fixed(const long long in_[INPUT_SIZE], long long out_[LAYER_SIZE])
{{
    for (int i = 0; i < LAYER_SIZE; ++i) {{
        long long acc = 0LL;
        
        for (int j = 0; j < INPUT_SIZE; ++j) {{
            long long prod = weights[i][j] * in_[j];
            acc += prod;
        }}
        
        out_[i] = (acc / SCALE_FACTOR) + biases[i];
    }}
}}

/* Verifica se a classificacao esta entre as classes validas */
static int verify_classification_multiclass(const long long out_[LAYER_SIZE])
{{
    long long max_valid = LLONG_MIN;
    long long max_invalid = LLONG_MIN;
    
    /* Encontra os valores maximos nas classes validas e invalidas */
    for (int i = 0; i < LAYER_SIZE; ++i) {{
        if (is_valid_class(i)) {{
            if (out_[i] > max_valid) {{
                max_valid = out_[i];
            }}
        }} else {{
            if (out_[i] > max_invalid) {{
                max_invalid = out_[i];
            }}
        }}
    }}
    
    /* (maior valor entre validas > maior valor entre invalidas) */
    return max_valid > max_invalid;
}}

int main(void)
{{
    long long input[INPUT_SIZE];
    long long output[LAYER_SIZE];
    
    /* Entrada nao-deterministica dentro dos bounds */
    for (int k = 0; k < INPUT_SIZE; ++k) {{
        input[k] = nondet_longlong();
        __ESBMC_assume(input[k] >= input_bounds_low[k] && 
                       input[k] <= input_bounds_high[k]);
    }}
    
    affine_transform_fixed(input, output);
    
    __ESBMC_assert(verify_classification_multiclass(output),
                   "Classification property violated - output not in valid classes");
    
    return 0;
}}
"""