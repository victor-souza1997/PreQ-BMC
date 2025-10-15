

def outerlayer(in_layer, cur_layer, weights_c, biases_c, input_bounds_low, input_bounds_high, targetCls):

    return f"""\

            #ifndef __invariant
            #define __invariant(p) /* paper-style invariant marker (no-op for ESBMC) */
            #endif
            static inline float f_abs(float x){{ return x < 0.0f ? -x : x; }}


            #define INPUT_SIZE   {in_layer.layer_size}
            #define LAYER_SIZE   {cur_layer.layer_size}
            #define TARGET_CLASS {targetCls}

            extern float nondet_float(void);

            float weights[LAYER_SIZE][INPUT_SIZE] = {weights_c};
            float biases[LAYER_SIZE]              = {biases_c};

            float input_bounds_low[INPUT_SIZE]  = {input_bounds_low};
            float input_bounds_high[INPUT_SIZE] = {input_bounds_high};

            static void affine_transform(const float in_[INPUT_SIZE], float out_[LAYER_SIZE])
            {{
            for (int i = 0; i < LAYER_SIZE; ++i) {{
                out_[i] = biases[i];
                for (int j = 0; j < INPUT_SIZE; ++j) {{
                out_[i] += weights[i][j] * in_[j];
                }}
            }}
            }}

            /* Classification: argmax(out) == TARGET_CLASS */
            static int verify_classification(const float out_[LAYER_SIZE])
            {{
            const int   T      = TARGET_CLASS;
            const float target = out_[T];
            float max_other    = -INFINITY;

            int i = 0;
            __invariant(0 <= i && i <= LAYER_SIZE);
            __invariant(max_other <= target);
            while (i < LAYER_SIZE)
            {{
                __ESBMC_loop_invariant(0 <= i && i <= LAYER_SIZE && max_other <= target);
                if (i != T) {{
                const float cand = out_[i];
                if (cand > max_other) max_other = cand;
                }}
                ++i;
            }}
            return max_other < target;
            }}

            int main(void)
            {{
            float input[INPUT_SIZE];
            float output[LAYER_SIZE];

            for (int k = 0; k < INPUT_SIZE; ++k) {{
                input[k] =  nondet_float();
                __ESBMC_assume(input[k] >= input_bounds_low[k] &&
                            input[k] <= input_bounds_high[k]);
            }}

            affine_transform(input, output);
            __ESBMC_assert(verify_classification(output),
                            "Classification property violated (output layer)");
            return 0;
            }}
            """

def innerlayer(cur_layer_layer_size, in_layer_layer_size, weights_c, biases_c, preimage_low_c, preimage_high_c, input_bounds_low, input_bounds_high):

    return f"""\

            #ifndef __invariant
            #define __invariant(p) /* paper-style invariant marker (no-op for ESBMC) */
            #endif

            #define INPUT_SIZE {in_layer_layer_size}
            #define LAYER_SIZE {cur_layer_layer_size}

            extern float nondet_float(void);

            float weights[LAYER_SIZE][INPUT_SIZE] = {weights_c};
            float biases[LAYER_SIZE]              = {biases_c};

            float preimage_low[LAYER_SIZE]  = {preimage_low_c};
            float preimage_high[LAYER_SIZE] = {preimage_high_c};

            float input_bounds_low[INPUT_SIZE]  = {input_bounds_low};
            float input_bounds_high[INPUT_SIZE] = {input_bounds_high};

            /* Affine layer over an input box: maintain running enclosure s_lb <= s_out <= s_ub */
            static void affine_transform_and_check(const float in_[INPUT_SIZE])
            {{
            const float abs_tol = 1e-3f;
            const float rel_tol = 1e-2f;

            for (int i = 0; i < LAYER_SIZE; ++i) {{
                float s_out = biases[i];   /* exact partial sum on actual input   */
                float s_lb  = biases[i];   /* running lower bound using box       */
                float s_ub  = biases[i];   /* running upper bound using box       */

                /* tolerance around the (relaxed) preimage interval */
                const float pre_lo = preimage_low[i];
                const float pre_hi = preimage_high[i];
                const float eps    = abs_tol + rel_tol * f_abs(pre_hi - pre_lo);

                int j = 0;
                __invariant(0 <= j && j <= INPUT_SIZE);
                __invariant(s_lb <= s_out && s_out <= s_ub);
                while (j < INPUT_SIZE)
                {{
                //__ESBMC_loop_invariant(0 <= j && j <= INPUT_SIZE &&
                //    s_lb <= s_out && s_out <= s_ub);
                const float w  = weights[i][j];
                const float lo = input_bounds_low[j];
                const float hi = input_bounds_high[j];

                /* exact step on actual (nondet) input */
                s_out += w * in_[j];

                /* sign-aware contribution bounds (box image) */
                const float cmin = (w >= 0.0f) ? (w * lo) : (w * hi);
                const float cmax = (w >= 0.0f) ? (w * hi) : (w * lo);
                s_lb += cmin;
                s_ub += cmax;

                ++j;
                }}

                /* final postcondition (affine, before any ReLU): inside tolerated preimage */
                __ESBMC_assert(s_out >= pre_lo - eps && s_out <= pre_hi + eps,
                            "Affine output not within tolerated preimage (hidden layer)");
            }}
            }}

            int main(void)
            {{
            /* nondet input constrained by box */
            float in_[INPUT_SIZE];
            for (int j = 0; j < INPUT_SIZE; ++j) {{
                in_[j] = nondet_float();
                __ESBMC_assume(in_[j] >= input_bounds_low[j] &&
                            in_[j] <= input_bounds_high[j]);
            }}

            affine_transform_and_check(in_);
            return 0;
        }}
        """


def outerlayer_fixed_int(in_layer_layer_size, cur_layer_layer_size, weights_c_int, biases_c_int,
                         input_bounds_low_int, input_bounds_high_int, targetCls, scale_factor):

    return f"""\

            #include <stdint.h>
            #include <math.h>

            #ifndef __invariant
            #define __invariant(p) /* paper-style invariant marker (no-op for ESBMC) */
            #endif

            #define INPUT_SIZE   {in_layer_layer_size}
            #define LAYER_SIZE   {cur_layer_layer_size}
            #define TARGET_CLASS {targetCls}
            #define SCALE_FACTOR {scale_factor}

            extern long long nondet_longlong(void);

            long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
            long long biases[LAYER_SIZE]              = {biases_c_int};

            long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
            long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

            static void affine_transform_fixed(const long long in_[INPUT_SIZE], long long out_[LAYER_SIZE])
            {{
                for (int i = 0; i < LAYER_SIZE; ++i) {{
                    long long acc = 0; /* scaled by SCALE_FACTOR^2 during accumulation */
                    for (int j = 0; j < INPUT_SIZE; ++j) {{
                        acc += weights[i][j] * in_[j];
                    }}
                    /* rescale to SCALE_FACTOR and add bias (already at SCALE_FACTOR) */
                    out_[i] = (acc / SCALE_FACTOR) + biases[i];
                }}
            }}

            /* Classification: argmax(out) == TARGET_CLASS */
            static int verify_classification(const long long out_[LAYER_SIZE])
            {{
                const int T   = TARGET_CLASS;
                const long long target = out_[T];
                long long max_other    = LLONG_MIN / 4;
                int i = 0;
                __invariant(0 <= i && i <= LAYER_SIZE);
                __invariant(max_other <= target);
                while (i < LAYER_SIZE)
                {{
                    __ESBMC_loop_invariant(0 <= i && i <= LAYER_SIZE && max_other <= target);
                    if (i != T) {{
                        const long long cand = out_[i];
                        if (cand > max_other) max_other = cand;
                    }}
                    ++i;
                }}
                return max_other < target;
            }}

            int main(void)
            {{
                long long input[INPUT_SIZE];
                long long output[LAYER_SIZE];
                for (int k = 0; k < INPUT_SIZE; ++k) {{
                    input[k] = nondet_float(); /* nondet within box */
                    __ESBMC_assume(input[k] >= input_bounds_low[k] && input[k] <= input_bounds_high[k]);
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

            #ifndef __invariant
            #define __invariant(p) /* paper-style invariant marker (no-op for ESBMC) */
            #endif

            #define INPUT_SIZE  {in_layer_layer_size}
            #define LAYER_SIZE  {cur_layer_layer_size}
            #define SCALE_FACTOR {scale_factor}

            extern long long nondet_longlong(void);

            long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
            long long biases[LAYER_SIZE]              = {biases_c_int};

            long long preimage_low[LAYER_SIZE]  = {preimage_low_int};
            long long preimage_high[LAYER_SIZE] = {preimage_high_int};

            long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
            long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

            static void affine_transform_and_check_fixed(const long long in_[INPUT_SIZE])
            {{
                for (int i = 0; i < LAYER_SIZE; ++i) {{
                    long long acc = 0; /* accumulate at SCALE_FACTOR^2 */
                    for (int j = 0; j < INPUT_SIZE; ++j) {{
                        acc += weights[i][j] * in_[j];
                    }}
                    long long s_out = (acc / SCALE_FACTOR) + biases[i]; /* back to SCALE_FACTOR */

                    /* exact postcondition (affine, before any ReLU): inside preimage */
                    __ESBMC_assert(s_out >= preimage_low[i] && s_out <= preimage_high[i],
                                   "Affine output not within preimage (hidden layer, fixed-point)");
                }}
            }}

            int main(void)
            {{
                long long in_[INPUT_SIZE];
                for (int j = 0; j < INPUT_SIZE; ++j) {{
                    in_[j] = nondet_float(); /* nondet within box */
                    __ESBMC_assume(in_[j] >= input_bounds_low[j] && in_[j] <= input_bounds_high[j]);
                }}
                affine_transform_and_check_fixed(in_);
                return 0;
            }}
        """
