

# At verification Level,
We should separete the verification of hidden layers from the output layer.
For hidden layers, we only need to check if the affine output is within the preimage bounds
// For output layer, we need to check if the classification is correct
```
int verify_layer_esbmc_optimized(long long int affine_output[LAYER_SIZE], 
                                int layer_index, 
                                PreimageBounds* bounds) {
    if (layer_index == output_layer_index) {
        // Output layer: check classification
        return affine_output[target_class] > max_other_classes(affine_output);
    } else {
        // Hidden layer: check affine output against preimage
        // No ReLU application needed - preimage accounts for it
        for (int i = 0; i < LAYER_SIZE; i++) {
            if (affine_output[i] < bounds->low[i] || 
                affine_output[i] > bounds->high[i]) {
                return 0;
            }
        }
        return 1;
    }
}
```

# Future ideas
Independent Bit-Accurate Verification: You could use Quadapter to synthesize a promising quantization strategy $\Xi$. Then, you could use an SMT solver with a bit-vector theory to perform a more rigorous, bit-exact verification of the resulting QNN $\hat{\mathcal{N}}$ for the property $\langle \mathcal{I}, \mathcal{O} \rangle$. This would provide an even stronger guarantee.

Targeting a Different Problem: The paper focuses on per-layer quantization. A fascinating extension would be to use SMT to synthesize or verify mixed-precision quantization at the per-channel or even per-weight level. The expressiveness of SMT could help manage the increased complexity of this finer-grained strategy.

Verifying Different Properties: The paper focuses on robustness and backdoor-freeness. Your SMT-based approach could be tailored to verify other properties that are more naturally expressed in first-order logic, which is the native language of SMT solvers.