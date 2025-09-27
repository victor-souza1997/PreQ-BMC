import numpy as np

from ceg4n.verifier.base import EquivalenceSpec


def export_3d(spec: EquivalenceSpec, lb: str, ub: str):
    _, height, width = [str(v) for v in spec.input_shape]

    output_size = str(np.prod(np.array(spec.y).shape))
    equivalence = str(0 if spec.top else 1)
    epsilon = str(spec.epsilon if spec.epsilon else -1)

    return (
        _TEMPLATE.replace("@OUTPUT_SIZE", output_size)
        .replace("@HEIGHT", height)
        .replace("@WIDTH", width)
        # .replace("@CHANNELS", channels)
        .replace("@LOWER_BOUNDS", lb)
        .replace("@UPPER_BOUNDS", ub)
        .replace("@EPSILON", epsilon)
        .replace("@EQUIVALENCE", equivalence)
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

#ifndef HEIGHT
#define HEIGHT @HEIGHT
#endif

#ifndef WIDTH
#define WIDTH @WIDTH
#endif

#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE @OUTPUT_SIZE
#endif

float nondet_float();

const float lower_bounds[BATCH][HEIGHT][WIDTH] = 
@LOWER_BOUNDS;

const float upper_bounds[BATCH][HEIGHT][WIDTH] = 
@UPPER_BOUNDS;

static inline void init_symbolic_input(float input[BATCH * HEIGHT * WIDTH])
{
    for(size_t i = 0; i < BATCH * HEIGHT * WIDTH; i++)
    {
        input[i] = nondet_float();            
    }
}

static inline void add_input_assumptions(float input[BATCH * HEIGHT * WIDTH], float x[BATCH][HEIGHT][WIDTH])
{
    size_t i = 0;
    for(size_t b = 0; b < BATCH; b++)
    {
	    for(size_t h = 0; h < HEIGHT; h++)
        {
            for(size_t w = 0; w < WIDTH; w++)
            {
                x[b][h][w] = input[i];
                __ESBMC_assume(lower_bounds[b][h][w] <= x[b][h][w] && x[b][h][w] <= upper_bounds[b][h][w]);
                i++;   
            }
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
    float input[BATCH*HEIGHT*WIDTH];
    float x[BATCH][HEIGHT][WIDTH];
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
}
"""
