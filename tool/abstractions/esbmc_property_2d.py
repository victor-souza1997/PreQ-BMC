import numpy as np

from ceg4n.verifier.base import EquivalenceSpec


def export_2d(
    spec: EquivalenceSpec,
    lb: str,
    ub: str,
    pre_lb: str,
    pre_ub: str,
    check_preimage: str,
    pre_layers: str,
):
    width = str(spec.input_shape[-1])
    output_size = str(np.prod(np.array(spec.y).shape))

    equivalence = str(0 if spec.top else 1)
    epsilon = str(spec.epsilon if spec.epsilon else -1)

    return (
        _TEMPLATE.replace("@OUTPUT_SIZE", output_size)
        .replace("@WIDTH", width)
        .replace("@LOWER_BOUNDS", lb)
        .replace("@UPPER_BOUNDS", ub)
        .replace("@EPSILON", epsilon)
        .replace("@EQUIVALENCE", equivalence)
        .replace("@CHECK_PREIMAGE", check_preimage)
        .replace("@PREIMAGE_NUM", pre_layers)
        .replace("@PREIMAGE_LB", pre_lb)
        .replace("@PREIMAGE_UB", pre_ub)
    )


_TEMPLATE = """

#include <stdlib.h>
#include <stdio.h>

#include "original.h"
#include "quantized.h"

#ifndef EQUIVALENCE
#define EQUIVALENCE @EQUIVALENCE
#endif

#ifndef EPSILON
#define EPSILON @EPSILON
#endif

#define BATCH 1

#ifndef WIDTH
#define WIDTH @WIDTH
#endif

#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE @OUTPUT_SIZE
#endif

#ifndef CHECK_PREIMAGE
#define CHECK_PREIMAGE @CHECK_PREIMAGE
#endif

#ifndef PREIMAGE_NUM
#define PREIMAGE_NUM @PREIMAGE_NUM
#endif

float nondet_float();

const float lower_bounds[BATCH][WIDTH] = 
@LOWER_BOUNDS;

const float upper_bounds[BATCH][WIDTH] = 
@UPPER_BOUNDS;

#if PREIMAGE_NUM > 0
const float preimage_lower_bounds[PREIMAGE_NUM][OUTPUT_SIZE] = 
@PREIMAGE_LB;

const float preimage_upper_bounds[PREIMAGE_NUM][OUTPUT_SIZE] = 
@PREIMAGE_UB;
#endif

static inline void init_symbolic_input(float input[BATCH * WIDTH])
{
    for(size_t i = 0; i < BATCH * WIDTH; i++)
    {
        input[i] = nondet_float();            
    }
}

static inline void add_input_assumptions(float input[BATCH * WIDTH], float x[BATCH][WIDTH])
{
    size_t i = 0;
    for(size_t b = 0; b < BATCH; b++)
    {
	    for(size_t w = 0; w < WIDTH; w++)
        {
            x[b][w] = input[i];
		    __ESBMC_assume(lower_bounds[b][w] <= x[b][w] && x[b][w] <= upper_bounds[b][w]);   
            i++;
        }
    }
}

static inline int top1(const float output[BATCH][OUTPUT_SIZE])
{
    int top = 0;
    for(size_t b = 0; b < BATCH; b++)
    {
        for (size_t o = 0; o < OUTPUT_SIZE; o++)
        {
            if(output[b][o] <= output[b][top])
            {
                continue;
            }
            top = (int) o;
        }
    }
    return top;
}

static inline void epsilon(
    const float output_original[BATCH][OUTPUT_SIZE],
    const float output_quantized[BATCH][OUTPUT_SIZE],
    float output_diff[BATCH][OUTPUT_SIZE]
){
    for(size_t b = 0; b < BATCH; b++)
    {
        for (size_t O = 0; O < OUTPUT_SIZE; O++)
        {
            output_diff[b][O] = output_original[b][O] - output_quantized[b][O];
            if(output_diff[b][O] >= 0)
            {
                continue;
            }
            output_diff[b][O] *= -1.0; 
        }
    }
}

static inline void check_top(const float output_original[BATCH][OUTPUT_SIZE], const float output_quantized[BATCH][OUTPUT_SIZE])
{
    int original_prediction = top1(output_original);
	int quantized_prediction = top1(output_quantized);

    int property_holds = (original_prediction==quantized_prediction);
    __ESBMC_assert(property_holds, "Networks not equivalent.");
}

static inline void check_epsilon(const float output_original[BATCH][OUTPUT_SIZE], const float output_quantized[BATCH][OUTPUT_SIZE])
{
    float output_diff[BATCH][OUTPUT_SIZE];
    
    // Get output diff
    epsilon(output_original, output_quantized, output_diff);

    int property_holds = 1;
    for(size_t O = 0; O < OUTPUT_SIZE; O++)
    {
        if (output_diff[0][O] <= EPSILON)
        {
            continue;
        }

        property_holds = 0;
        break;
    }

    __ESBMC_assert(property_holds, "Networks not equivalent.");
}

int main()
{

    // Define input and output vectors
    float input[BATCH*WIDTH];
    float x[BATCH][WIDTH];
    float original_output[BATCH][OUTPUT_SIZE];
    float quantized_output[BATCH][OUTPUT_SIZE];


    // Init symbolic input vector
    init_symbolic_input(input);

    // Add input assumptions
    add_input_assumptions(input, x);

    // Call networks
    original(x, original_output);
    quantized(x, quantized_output);

    if(EQUIVALENCE == 0)
    {
        check_top(original_output, quantized_output);
    } else {
        check_epsilon(original_output, quantized_output);
    }

    if(CHECK_PREIMAGE == 1 && PREIMAGE_NUM > 0)
    {
        int property_holds = 1;
        for(size_t l = 0; l < PREIMAGE_NUM; l++)
        {
            for(size_t o = 0; o < OUTPUT_SIZE; o++)
            {
                if(preimage_lower_bounds[l][o] <= quantized_output[0][o] && quantized_output[0][o] <= preimage_upper_bounds[l][o])
                {
                    continue;
                }

                property_holds = 0;
                break;
            }

            if(property_holds == 0)
            {
                break;
            }
        }

        __ESBMC_assert(property_holds, "Quantized network output violates preimage bounds.");
    }
}

"""
