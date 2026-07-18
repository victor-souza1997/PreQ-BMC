#include <stdint.h>
#include <limits.h>

#ifndef __invariant
#define __invariant(p) /* paper-style invariant marker (no-op for ESBMC) */
#endif

#define INPUT_SIZE   15
#define LAYER_SIZE   15
#define SCALE_FACTOR 65536LL

extern long long nondet_longlong(void);

long long weights[LAYER_SIZE][INPUT_SIZE] = {{-12823, 13630, -45886, -13784, -12653, -27612, 1333, -9964, 13449, -16503, -6041, 38324, -22087, -3677, 131071}, {-14143, 6685, -35509, -8662, 3763, -23738, -953, 13066, 6239, -8161, -21904, 15028, -8135, 15540, 46358}, {20181, -6811, 52086, -9836, 24736, 38533, -10068, -8997, -8011, -15774, 43357, 32702, 18505, 4919, -63702}, {-11526, 5804, 7037, 9486, 5248, 13162, -15042, 11992, 16315, 3479, -11183, -12297, 4160, 2231, 8113}, {16628, 16723, -12266, -6007, -15354, 8606, 7895, 4140, 4473, -1515, -7834, -3791, 12225, -14798, 2159}, {14120, -16118, 5075, 8731, -16889, 11856, -7883, -2677, -2337, 11516, 10774, -11368, -12337, 11234, -16125}, {9227, 14214, 53388, 9218, -5548, 17565, -4630, -12271, 7907, -8513, 50888, 24338, 38441, 3398, -42606}, {3955, 15792, -3251, -2989, 11076, -2248, 7315, 12731, 8099, -3103, -9104, -11483, 1331, 4399, 10346}, {10634, 12373, -4645, -10605, -10382, -15588, 13981, 7087, 1216, 9434, 179, 13625, 324, -5321, -6959}, {-15708, -11906, -9483, -15975, 9817, -3565, -14312, -8953, 16893, 2561, -7913, -11913, 11285, -14414, 7460}, {15987, 13930, 598, 12468, -14936, 16956, 12273, 1735, -1912, 15683, 2754, 257, -10844, 10788, -6637}, {-11177, -13515, 15522, -7409, 1367, -12663, -7354, -8129, -6159, 3616, 14668, 13854, 19761, -8702, -14065}, {-15744, -15606, -13415, -1263, -14308, 5035, -2067, -12253, 14684, -16616, 63, 9985, 4078, 523, -6362}, {-12473, -7064, -14044, 15119, -11419, -8932, 16016, -150, -4283, -11302, -312, 12706, 10926, -14299, -14590}, {-614, -4370, -5397, 2353, -7520, 8325, -14302, -4075, -15386, 1897, -5195, 10524, 8576, -325, -8720}};
long long biases[LAYER_SIZE]              = {71861, 8346, 12687, -12176, 1318, -13210, 27702, -15054, -8120, -11611, 11917, 35363, -9974, -8096, 13196};

long long preimage_low[LAYER_SIZE]  = {287623, 53481, -174387, -31393, -11663, -87508, -90671, -24210, -20887, -36220, -12795, 3603, -25387, -49611, -7826};
long long preimage_high[LAYER_SIZE] = {642890, 200136, 34170, 7275, 2921, -32561, 74551, 9984, 6801, 1408, 9860, 51558, -8129, -11533, 8005};

long long input_bounds_low[INPUT_SIZE]  = {26651, 53957, 1566, 2075, 0,0,0,0,0,0,0,0,0,0,0};
long long input_bounds_high[INPUT_SIZE] = {27963, 55269, 2877, 3387, 0,0,0,0,0,0,0,0,0,0,0};

static inline long long llabs(long long x) {
    return x < 0LL ? -x : x;
}
static inline long long div_floor_ll(long long x, long long k) {
  // k > 0 no nosso caso
  if (x >= 0) return x / k;
  // floor para negativo
  return -(( -x + k - 1 ) / k);
}
static inline long long div_ceil_ll(long long x, long long k) {
  if (x >= 0) return (x + k - 1) / k; // ceil para positivo
  return x / k;                        // trunc(a/b) já é ceil quando x<0
}

/* Camada afim em ponto fixo sobre um box de entrada: mantém o invólucro s_lb <= s_out <= s_ub */
static void check_affine_bounds_fixed(const long long in_[INPUT_SIZE])
{
    /* tolerancia para preimagem */
    const long long abs_tol = (long long)(1e-3 * SCALE_FACTOR);
    const long long rel_tol_num = 1; /* 1% = 1/100 */
    const long long rel_tol_den = 100;
    
    for (int i = 0; i < LAYER_SIZE; ++i) {
        long long s_out = 0LL;  /* saída exata na entrada atual */
        long long s_lb  = 0LL;  /* limite inferior usando box */
        long long s_ub  = 0LL;  /* limite superior usando box */
        
        /* Tolerância ao redor do intervalo de pré-imagem */
        const long long pre_lo = preimage_low[i];
        const long long pre_hi = preimage_high[i];
        const long long range = llabs(pre_hi - pre_lo);
        const long long eps = abs_tol + (rel_tol_num * range) / rel_tol_den;
        
        int j = 0;
        //__invariant(0 <= j && j <= INPUT_SIZE);
        //__invariant(s_lb <= s_out && s_out <= s_ub);
        
        //__ESBMC_loop_invariant(0 <= j && j <= INPUT_SIZE && s_lb <= s_out && s_out <= s_ub);
        loop_invariant(0 <= j && j <= INPUT_SIZE);
        loop_invariant(s_lb <= s_out && s_out <= s_ub);
        while (j < INPUT_SIZE)
        {
        _decreases(INPUT_SIZE - j);
            
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
        }
        
        /* Rescale e adiciona bias */
         const long long s_out_q = (s_out / SCALE_FACTOR) + biases[i];
    const long long s_lb_q  = div_floor_ll(s_lb, SCALE_FACTOR) + biases[i];
    const long long s_ub_q  = div_ceil_ll (s_ub, SCALE_FACTOR) + biases[i];

        
        /* verifica se a saida esta dentro da preimagem esperada */
        __ESBMC_assert(s_lb_q <= s_out_q && s_out_q <= s_ub_q,
                   "internal box image invariant broken after rescale");
        __ESBMC_assert(s_out_q >= pre_lo - eps && s_out_q <= pre_hi + eps,
                   "affine output not within tolerated preimage");
      
                    }

}

int main(void)
{
    long long in_[INPUT_SIZE];
    
    /* Entrada nao determinística dentro do box */
    for (int j = 0; j < INPUT_SIZE; ++j) {
        in_[j] = nondet_longlong();
        __ESBMC_assume(in_[j] >= input_bounds_low[j] && 
                       in_[j] <= input_bounds_high[j]);
    }
    
    check_affine_bounds_fixed(in_);
    
    return 0;
}
