from asyncio.subprocess import PIPE
import os
import re
from decimal import Decimal
from pathlib import Path
from subprocess import Popen

import numpy as np
from typing import Optional, Tuple

# Lazy/optional imports to avoid import-time errors when only converting ONNX.
try:
    from .esbmc_property_2d import export_2d  # type: ignore
    from .esbmc_property_3d import export_3d  # type: ignore
    from .esbmc_property_4d import export_4d  # type: ignore
except Exception:
    export_2d = export_3d = export_4d = None  # type: ignore
try:
    from .verifier.base import EquivalenceSpec  # type: ignore
except Exception:
    # Minimal fallback so type checking continues; functions won't need it.
    class EquivalenceSpec:  # type: ignore
        pass


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
    """Run onnx2c binary to translate ONNX to C.

    Uses $ONNX2C_PATH if set; otherwise tries to infer from constants.ONNX2C_PATH,
    falling back to a local "bin" folder next to the repo root.
    """
    onnx2c_path = os.environ.get("ONNX2C_PATH")
    if not onnx2c_path:
        try:
            from constants import ONNX2C_PATH as CONST_ONNX2C  # type: ignore
            onnx2c_path = str(CONST_ONNX2C)
        except Exception:
            onnx2c_path = str(Path.cwd().parent.joinpath("bin"))
        os.environ["ONNX2C_PATH"] = onnx2c_path

    onnx2c_bin = f"{onnx2c_path}/onnx2c"
    command = [onnx2c_bin, str(onnx_path)]
    process = Popen(command, stdout=PIPE, stderr=PIPE, shell=False)
    output, error = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"onnx2c failed ({process.returncode}): {error.decode('utf-8')}")
    return output.decode("utf-8")


def onnx_to_c(onnx_path: Path) -> str:
    file_content = _run_onnx2c(onnx_path)
    file_content = _process_file_content(file_content)
    file_content = "\n".join(file_content)
    return file_content


def convert_onnx_to_esbmc_header(
    onnx_path: Path,
    out_header: Path,
    *,
    function_name: str = "network",
    rename_prefix: Optional[str] = None,
    header_guard: Optional[str] = None,
) -> Path:
    """Convert an ONNX file to a sanitized C header suitable for ESBMC.

    - Translates using onnx2c and normalizes output for ESBMC ingestion.
    - Renames the default entry function to `function_name` (default: network).
    - Optionally prefixes common symbols (tensor/node) via `rename_prefix` to avoid
      collisions when including multiple networks.
    - Wraps output in a header guard (auto-generated if not provided).

    Returns the path to the written header.
    """
    onnx_path = Path(onnx_path)
    out_header = Path(out_header)
    out_header.parent.mkdir(parents=True, exist_ok=True)

    content = onnx_to_c(onnx_path)
    content = content.replace("entry", function_name)
    if rename_prefix:
        content = (
            content.replace("tensor", f"{rename_prefix}_tensor")
            .replace("node", f"{rename_prefix}_node")
        )

    guard = header_guard or f"{function_name.upper()}_H"
    header_text = f"""
#ifndef {guard}
#define {guard}
{content}
#endif // {guard}
""".lstrip()

    with open(out_header, "w") as fp:
        fp.write(header_text)

    return out_header


def convert_pair_to_esbmc_headers(
    original_onnx: Path,
    quantized_onnx: Path,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Convert original and quantized ONNX models to ESBMC headers.

    Produces `original.h` and `quantized.h` under `out_dir`, renaming symbols in
    the quantized header to avoid clashes (`quantized_tensor`, `quantized_node`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_h = out_dir.joinpath("original.h")
    q_h = out_dir.joinpath("quantized.h")

    convert_onnx_to_esbmc_header(original_onnx, orig_h, function_name="original")
    convert_onnx_to_esbmc_header(
        quantized_onnx,
        q_h,
        function_name="quantized",
        rename_prefix="quantized",
    )
    return orig_h, q_h


def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Convert ONNX to ESBMC-ready C headers using onnx2c")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--onnx", dest="onnx", help="Path to a single ONNX model to convert")
    mode.add_argument("--pair", dest="pair", action="store_true", help="Convert original+quantized ONNX pair")

    # Single conversion
    p.add_argument("--out", dest="out", help="Output header path for single conversion")
    p.add_argument("--function_name", default="network", help="Rename entry function to this name (single)")
    p.add_argument("--rename_prefix", default=None, help="Optional prefix for tensor/node symbols (single)")
    p.add_argument("--guard", default=None, help="Optional header guard name (single)")

    # Pair conversion
    p.add_argument("--onnx_original", help="Original ONNX path (pair mode)")
    p.add_argument("--onnx_quantized", help="Quantized ONNX path (pair mode)")
    p.add_argument("--out_dir", help="Output directory for original.h and quantized.h (pair mode)")

    # onnx2c location (optional override)
    p.add_argument("--onnx2c_path", help="Directory containing onnx2c binary; overrides $ONNX2C_PATH")
    return p


def _main_cli(argv=None):
    p = _build_arg_parser()
    args = p.parse_args(argv)

    if args.onnx2c_path:
        os.environ["ONNX2C_PATH"] = args.onnx2c_path

    if args.onnx:
        if not args.out:
            p.error("--out is required when using --onnx")
        out = convert_onnx_to_esbmc_header(
            Path(args.onnx),
            Path(args.out),
            function_name=args.function_name,
            rename_prefix=args.rename_prefix,
            header_guard=args.guard,
        )
        print(f"Wrote header: {out}")
        return 0

    # Pair mode
    if not (args.onnx_original and args.onnx_quantized and args.out_dir):
        p.error("--onnx_original, --onnx_quantized and --out_dir are required in --pair mode")
    orig_h, q_h = convert_pair_to_esbmc_headers(
        Path(args.onnx_original), Path(args.onnx_quantized), Path(args.out_dir)
    )
    print(f"Wrote headers: {orig_h}, {q_h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_cli())


def save_abstraction(benchmark: str, abstraction_path: Path, spec: EquivalenceSpec):
    _save_original(benchmark, abstraction_path)
    _save_quantized(benchmark, abstraction_path)
    _save_property(abstraction_path, spec)


def _save_original(benchmark: str, abstraction_path: Path):
    from tool.nn import original_model_provider  # lazy import
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
    from tool.nn import quantized_model_provider  # lazy import
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
