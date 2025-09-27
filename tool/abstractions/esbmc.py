from asyncio.subprocess import PIPE
import re
from decimal import Decimal
from pathlib import Path
from subprocess import Popen

import numpy as np

from ceg4n.abstractions.esbmc_property_2d import export_2d
from ceg4n.abstractions.esbmc_property_3d import export_3d
from ceg4n.abstractions.esbmc_property_4d import export_4d
from ceg4n.nn import original_model_provider, quantized_model_provider
from ceg4n.verifier.base import EquivalenceSpec


def _process(lines):
    should_remove_curly = False
    for line in lines:
        if re.match(r"union[\w|\s|\d]+{", line):
            # fp.writelines(line)
            should_remove_curly = True
            continue
        elif re.match("static union[\w|\s|\d]+;", line):
            # fp.writelines(line)
            continue
        elif should_remove_curly and re.match(r"};", line):
            # fp.writelines(line)
            should_remove_curly = False
            continue
        elif re.findall(r"tu[\d]+.tensor", line):
            line = re.sub(r"tu[\d]+.tensor", "tensor", line)
            # line = re.sub(r"tu[\d]+.tensor", "tensor", line)
            # continue
        yield line


def _process_file_content(file_content: str):
    lines = file_content.split("\n")
    return list(_process(lines))


def _run_onnx2c(onnx_path: Path):
    import os
    onnx2c = f'{os.environ["ONNX2C_PATH"]}/onnx2c'
    command = [onnx2c, onnx_path]
    print(command)
    process = Popen(command, stdout=PIPE, stderr=PIPE, shell=False)
    output, error = process.communicate()
    return output.decode("utf-8")


def onnx_to_c(onnx_path: Path) -> str:
    file_content = _run_onnx2c(onnx_path)
    file_content = _process_file_content(file_content)
    file_content = "\n".join(file_content)
    return file_content


def save_abstraction(benchmark: str, abstraction_path: Path, spec: EquivalenceSpec):
    _save_original(benchmark, abstraction_path)
    _save_quantized(benchmark, abstraction_path)
    _save_property(abstraction_path, spec)


def _save_original(benchmark: str, abstraction_path: Path):
    filename = original_model_provider.model_file(benchmark)
    original_content = onnx_to_c(filename)
    original_net = _ORIGINAL_TEMPLATE.format(original_content).replace(
        "entry", "original"
    )

    filename = str(abstraction_path.joinpath("original.h"))
    with open(filename, "w") as fp:
        fp.writelines(original_net)
        fp.flush()
        fp.close()


def _save_quantized(benchmark: str, abstraction_path: Path):
    filename = quantized_model_provider.model_file(benchmark)
    quantized_content = onnx_to_c(filename)
    quantized_net = (
        __QUANTIZED_TEMPLATE.format(quantized_content)
        .replace("entry", "quantized")
        .replace("tensor", "quantized_tensor")
        .replace("node", "quantized_node")
    )
    filename = str(abstraction_path.joinpath("quantized.h"))
    print(filename)
    with open(filename, "w") as fp:
        fp.writelines(quantized_net)
        fp.flush()
        fp.close()


def _save_property(abstraction_path: Path, spec: EquivalenceSpec):
    def array2string(arr):
        shape = arr.shape
        arr = arr.flatten().tolist()
        arr = ["{}".format(Decimal(x)) for x in arr]
        arr = str(np.array(arr).reshape(shape).tolist())
        arr = re.sub(r"\s+", " ", arr.replace("[", "{").replace("]", "}")).replace("'", "").replace(", ", ",\n\t")#.replace(" ", ", ")
        return arr
    
    lb = array2string(np.array(spec.input_lower_bounds))
    ub = array2string(np.array(spec.input_upper_bounds))
    # ub = np.array2string(
        # ub, formatter={"float_kind": lambda x: "{}".format(Decimal(x))}
    # )
    # ub = re.sub(r"\s+", " ", ub.replace("[", "{").replace("]", "}")).replace(" ", ", ")

    if len(spec.input_shape) == 2:
        content = export_2d(spec, lb, ub)
    elif len(spec.input_shape) == 3:
        content = export_3d(spec, lb, ub)
    elif len(spec.input_shape) == 4:
        content = export_4d(spec, lb, ub)
    else:
        raise RuntimeError("Invalid specification")

    prop_id = spec.spec_id.replace(".", "_")
    filename = str(abstraction_path.joinpath(f"main_{prop_id}.c"))
    with open(filename, "w") as fp:
        fp.writelines(content)
        

_ORIGINAL_TEMPLATE = """
#ifndef ORIGINAL_H
#define ORIGINAL_H
{}
#endif // ORIGINAL_H

"""

__QUANTIZED_TEMPLATE = """
#ifndef QUANTIZED_H
#define QUANTIZED_H
{}
#endif // QUANTIZED_H

"""
