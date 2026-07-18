#include <stdint.h>
#include <limits.h>
#include <stdbool.h>


#define INPUT_SIZE       10
#define LAYER_SIZE       3
#define NUM_VALID_CLASSES 3
#define SCALE_FACTOR     32768LL

extern long long nondet_longlong(void);

long long weights[LAYER_SIZE][INPUT_SIZE] = {{29717, -7006, -47581, 43340, -9874, 6848, 8704, -6950, -22322, 13735}, {-20569, -2835, 238, 30529, -5213, 16677, -386, 14704, 4388, -22172}, {-4114, 9671, 31239, -63978, 6783, -23409, 4480, 15754, 30736, -12179}};
long long biases[LAYER_SIZE]              = {-4695, 8264, -15361};

long long input_bounds_low[INPUT_SIZE]  = {66764, 0, 0, 207032, 0, 85844, 0, 3591, 0, 53844};
long long input_bounds_high[INPUT_SIZE] = {69230, 0, 0, 210931, 0, 87680, 0, 4462, 0, 55875};

int valid_classes[NUM_VALID_CLASSES] = {0, 1, 2};

/* Verifica se uma classe está no conjunto de classes válidas */
static bool is_valid_class(int class_id) {
    for (int i = 0; i < NUM_VALID_CLASSES; ++i) {
        if (valid_classes[i] == class_id) {
            return true;
        }
    }
    return false;
}

/* Transformacao afim em ponto fixo */
static void affine_transform_fixed(const long long in_[INPUT_SIZE], long long out_[LAYER_SIZE])
{
    for (int i = 0; i < LAYER_SIZE; ++i) {
        long long acc = 0LL;
        
        for (int j = 0; j < INPUT_SIZE; ++j) {
            long long prod = weights[i][j] * in_[j];
            acc += prod;
        }
        
        out_[i] = (acc / SCALE_FACTOR) + biases[i];
    }
}

/* Verifica se a classificacao está entre as classes validas */
static int verify_classification_multiclass(const long long out_[LAYER_SIZE])
{
    long long max_valid = LLONG_MIN;
    long long max_invalid = LLONG_MIN;
    
    /* Encontra os valores maximos nas classes validas e invalidas */
    for (int i = 0; i < LAYER_SIZE; ++i) {
        if (is_valid_class(i)) {
            if (out_[i] > max_valid) {
                max_valid = out_[i];
            }
        } else {
            if (out_[i] > max_invalid) {
                max_invalid = out_[i];
            }
        }
    }
    
    /* A classe predita deve ser valida (maior valor entre validas > maior valor entre invalidas) */
    return max_valid > max_invalid;
}

int main(void)
{
    long long input[INPUT_SIZE];
    long long output[LAYER_SIZE];
    
    /* Entrada nao-deterministica dentro dos bounds */
    for (int k = 0; k < INPUT_SIZE; ++k) {
        input[k] = nondet_longlong();
        __ESBMC_assume(input[k] >= input_bounds_low[k] && 
                       input[k] <= input_bounds_high[k]);
    }
    
    affine_transform_fixed(input, output);
    
    __ESBMC_assert(verify_classification_multiclass(output),
                   "Classification property violated - output not in valid classes");
    
    return 0;
}
