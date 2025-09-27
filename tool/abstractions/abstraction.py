from collections import OrderedDict
from decimal import Decimal

import sympy as sp
import torch
import torch.nn as nn

from ceg4n.constants import ABSTRACTIONS_PATH
from ceg4n.nn import original_model_provider, quantized_model_provider


class Abstraction:
    def get_network_graph(self, params):
        layers_sequence = [
            (name, layer)
            for name, layer in params.items()
            if not isinstance(layer, nn.Flatten)
        ]
        previous_layer = "input"
        network_graph = []
        for i, (current_layer, layer) in enumerate(layers_sequence):
            if isinstance(layer, nn.ReLU):
                continue
            next_layer = (
                layers_sequence[i + 2][0] if i + 2 < len(layers_sequence) else "y"
            )
            is_activated = (i + 1 < len(layers_sequence)) and isinstance(
                layers_sequence[i + 1][1], nn.ReLU
            )
            network_graph.append(
                (previous_layer, current_layer, next_layer, is_activated)
            )
            previous_layer = current_layer
        return network_graph

    def write_weights_and_bias(self, fp, weights_and_bias, suffix=""):
        def get_lines(data):
            var_names, var_values = data
            var_names = var_names.flatten().tolist()
            var_values = var_values.flatten().tolist()
            data = zip(var_names, var_values)
            data = [(str(name), "{}f".format(Decimal(value))) for name, value in data]
            return [
                f"const float {suffix}{variable} = {value};\n"
                for variable, value in data
            ]

        for _, (weight, bias) in weights_and_bias.items():
            weight_lines = get_lines(weight)
            fp.writelines(weight_lines + ["\n"])
            bias_lines = get_lines(bias)
            fp.writelines(bias_lines + ["\n"])

    def get_layers(self, params):
        def get_data(prefix, tensor):
            symbolic_tensor = sp.symarray(prefix=prefix, shape=tensor.shape)
            return symbolic_tensor, tensor.detach().cpu().numpy()

        layers_data = []
        for (name, layer) in params.items():
            layers_data.append(
                (
                    name,
                    (
                        get_data(f"{name}_weight", layer.weight.data),
                        get_data(f"{name}_bias", layer.bias.data),
                    ),
                )
            )
        return OrderedDict(layers_data)

    def write_affine(self, fp, network_graph, weights_and_bias, suffix=""):
        layers = list(zip(network_graph, weights_and_bias.items()))
        fp.writelines("\n")
        (previous_layer, current_layer, next_layer, is_activated), (
            name,
            (weight, bias),
        ) = layers[0]
        weight_vars, _ = weight
        output_shape, input_shape = weight_vars.shape
        fp.writelines(
            "%s = {0};\n" % f"float {suffix}tensor_{previous_layer}[{input_shape}]"
        )
        for (previous_layer, current_layer, next_layer, is_activated), (
            name,
            (weight, bias),
        ) in layers:
            weight_vars, _ = weight
            bias_vars, _ = bias
            output_shape, input_shape = weight_vars.shape
            fp.writelines(
                "%s = {0};\n" % f"float {suffix}tensor_{current_layer}[{output_shape}]"
            )

        for (previous_layer, current_layer, next_layer, is_activated), (
            name,
            (weight, bias),
        ) in layers:
            weight_vars, _ = weight
            bias_vars, _ = bias
            output_shape, input_shape = weight_vars.shape
            x = sp.symarray(
                prefix=f"{suffix}tensor_{previous_layer}", shape=(input_shape,)
            )
            x_vars = [str(x_var) for x_var in x]
            x_replaces = {
                x_var: f"{suffix}tensor_{previous_layer}[{i}]"
                for i, x_var in enumerate(x_vars)
            }
            dot_product_sum = x @ weight_vars.T + bias_vars

            def replace_names(output):
                for e, v in x_replaces.items():
                    output = output.replace(e, v)
                return output.replace("+", "\n\t\t\t\t+")

            outputs = []
            for output in dot_product_sum:
                if is_activated:
                    output = sp.Max(0.0, output)
                outputs.append(replace_names(str(output)))

            fp.writelines(f"\nstatic inline void {suffix}{current_layer}()")
            fp.writelines("\n{\n")
            for i, ouptus in enumerate(outputs):
                fp.writelines(
                    f"\t{suffix}tensor_{current_layer}[{i}] = {outputs[i]};\n"
                )
            fp.writelines("\n};\n")

    def write_network(self, fp, weights_and_bias, suffix=""):
        layers = list(weights_and_bias.items())
        (first_layer_name, (weight, bias)) = layers[0]
        weight_vars, _ = weight
        _, x_size = weight_vars.shape

        (last_layer_name, (weight, bias)) = layers[-1]
        weight_vars, _ = weight
        y_size, _ = weight_vars.shape

        fp.writelines(
            "%s\n{"
            % f"\nstatic inline void {suffix}network(const float x[{x_size}], float y[{y_size}])"
        )

        fp.writelines(
            "\n\t%s\n\t{\n\t\t%s\n\t}\n"
            % (
                f"for(size_t i = 0; i < {x_size}; i++)",
                f"tensor_{first_layer_name}[i] = x[i];",
            )
        )

        for (layer_name, _) in layers:
            fp.writelines(f"\n\t{suffix}{layer_name}();")

        fp.writelines(
            "\n\n\t%s\n\t{\n\t\t%s\n\t}\n"
            % (
                f"for(size_t i = 0; i < {x_size}; i++)",
                f"y[i] = tensor_{last_layer_name}[i];",
            )
        )
        fp.writelines("}\n\n")

    def write_network(self, model_params, filename, suffix):
        with open(filename, "w") as fp:
            fp.writelines(
                "\n#ifndef Max\n#define Max(X,Y) ( X > Y ? X : Y)\n#endif\n\n"
            )
            network_graph = self.get_network_graph(model_params.params)
            weights_and_bias = self.get_layers(model_params.quantizable_params)
            self.write_weights_and_bias(fp, weights_and_bias, suffix=suffix)
            self.write_affine(fp, network_graph, weights_and_bias, suffix=suffix)
            self.write_network(fp, weights_and_bias, suffix=suffix)

    def generate(self, benchmark, properties):
        benchmark = benchmark.replace("-", "_")
        abs_path = ABSTRACTIONS_PATH.joinpath(benchmark.lower())
        abs_path.mkdir(exist_ok=True)

        model_params = original_model_provider(benchmark)
        filename = str(abs_path.joinpath("network.h"))
        self.write_network(model_params, filename, suffix="")

        filename = str(abs_path.joinpath("q_network.h"))
        model_params = quantized_model_provider(benchmark)
        self.write_network(model_params, filename, suffix="q_")

    def write_test_cases(self, path, properties):
        raise RuntimeError("Should not be calles")
