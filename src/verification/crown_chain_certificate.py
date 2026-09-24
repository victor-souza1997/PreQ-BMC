"""Render ESBMC obligations for untrusted exact-integer CROWN chains."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

SCHEMA = "crown_chain_certificate_pilot_v1"


def _ints(values: Iterable[Any], name: str) -> list[int]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must contain integers")
        result.append(int(value))
    return result


def _sparse(payload: dict[str, Any], name: str, size: int) -> dict[int, int]:
    indices = _ints(payload.get("indices", []), f"{name}.indices")
    values = _ints(payload.get("values", []), f"{name}.values")
    if len(indices) != len(values) or len(set(indices)) != len(indices):
        raise ValueError(f"{name} is not canonical sparse data")
    if any(index < 0 or index >= size for index in indices):
        raise ValueError(f"{name} has an out-of-range index")
    if any(value == 0 for value in values):
        raise ValueError(f"{name} stores an explicit zero")
    return dict(zip(indices, values, strict=True))


def _csr_payload(layer: dict[str, Any]) -> dict[str, Any]:
    csr = layer["lowered_csr"]
    rows, cols = int(csr["rows"]), int(csr["cols"])
    indptr = _ints(csr["indptr"], "csr.indptr")
    indices = _ints(csr["indices"], "csr.indices")
    data = _ints(csr["data"], "csr.data")
    bias = _ints(csr["bias"], "csr.bias")
    if (
        rows <= 0 or cols <= 0 or len(indptr) != rows + 1
        or indptr[0] != 0 or indptr[-1] != len(data)
        or any(left > right for left, right in zip(indptr, indptr[1:]))
        or len(indices) != len(data) or len(bias) != rows
        or any(index < 0 or index >= cols for index in indices)
    ):
        raise ValueError("Invalid lowered CSR")
    hashed = {
        "rows": rows, "cols": cols, "indptr": indptr,
        "indices": indices, "data": data, "bias": bias,
    }
    digest = hashlib.sha256(
        json.dumps(hashed, sort_keys=True).encode()
    ).hexdigest()
    if digest != csr.get("sha256"):
        raise ValueError(f"Lowered CSR hash mismatch for layer {layer['layer_index']}")
    return {**hashed, "sha256": digest}


def load_crown_chain_certificate(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported CROWN chain schema: {data.get('schema')!r}")
    scale_bits = data.get("scale_bits")
    if not isinstance(scale_bits, int) or not 0 <= scale_bits <= 62:
        raise ValueError("Unsupported scale_bits")
    qif = data.get("qif", {})
    total_bits, fractional_bits = qif.get("total_bits"), qif.get("fractional_bits")
    if not isinstance(total_bits, int) or not 2 <= total_bits <= 62:
        raise ValueError("Unsupported total_bits")
    if not isinstance(fractional_bits, int) or not 0 <= fractional_bits < total_bits:
        raise ValueError("Unsupported fractional_bits")
    low = _ints(data["input_box"]["low"], "input_box.low")
    high = _ints(data["input_box"]["high"], "input_box.high")
    if not low or len(low) != len(high) or any(a > b for a, b in zip(low, high)):
        raise ValueError("Invalid input box")

    layers = {}
    for layer in data.get("layers", []):
        index = layer.get("layer_index")
        if not isinstance(index, int) or index in layers:
            raise ValueError("Invalid or duplicate layer index")
        layers[index] = _csr_payload(layer)
    if set(layers) != set(range(len(layers))):
        raise ValueError("Layers must be consecutively indexed")
    if layers[0]["cols"] != len(low):
        raise ValueError("Input box does not match layer zero")

    bounds = {}
    for record in data.get("bound_records", []):
        bound_id = record.get("bound_id")
        if not isinstance(bound_id, str) or bound_id in bounds:
            raise ValueError("Invalid or duplicate bound ID")
        layer, neuron = record.get("layer_index"), record.get("neuron_index")
        lower, upper = record.get("lower"), record.get("upper")
        box = _ints(record.get("post_clamp_pre_relu_box", []), f"{bound_id}.box")
        if (
            not isinstance(layer, int) or layer not in layers
            or not isinstance(neuron, int) or not 0 <= neuron < layers[layer]["rows"]
            or not isinstance(lower, int) or not isinstance(upper, int)
            or lower > upper or len(box) != 2 or box[0] > box[1]
            or record.get("source") not in {"box_only", "chain_intersect_box"}
        ):
            raise ValueError(f"Invalid bound record {bound_id}")
        bounds[bound_id] = record

    chains = data.get("chains", [])
    identifiers = [chain.get("certificate_id") for chain in chains]
    if any(not isinstance(value, str) for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Invalid or duplicate chain ID")
    return {
        **data,
        "input_box": {"low": low, "high": high},
        "layers_by_index": layers,
        "bounds_by_id": bounds,
    }


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _header() -> list[str]:
    return [
        "#include <stdint.h>",
        "void __ESBMC_assume(_Bool);",
        "void __ESBMC_assert(_Bool, const char *);",
        "long long nondet_long_long(void);",
        "int nondet_int(void);",
        "static __int128 floor_div_i128(__int128 n, __int128 d) {",
        "  __int128 q = n / d, r = n % d;",
        "  return q - (r != 0 && n < 0);",
        "}",
        "",
    ]


def _bound_source_value(
    certificate: dict[str, Any],
    layer_index: int,
    neuron: int,
) -> tuple[int, dict[str, Any]]:
    bound_id = f"bound/L{layer_index}/n{neuron}"
    record = certificate["bounds_by_id"].get(bound_id)
    if record is None:
        raise ValueError(f"Missing {bound_id}")
    if record["layer_index"] != layer_index or record["neuron_index"] != neuron:
        raise ValueError(f"Mismatched {bound_id}")
    return max(int(record["lower"]), 0), record


def _validate_bound_record(
    record: dict[str, Any],
    q_low: int,
    q_high: int,
) -> list[str]:
    lower, upper = int(record["lower"]), int(record["upper"])
    chain_low, chain_high = int(record["chain_lower"]), int(record["chain_upper"])
    box_low, box_high = map(int, record["post_clamp_pre_relu_box"])
    label = record["bound_id"]
    lines = [
        f'  __ESBMC_assert({lower} <= {upper}, "{label} ordered");',
    ]
    if record["source"] == "box_only":
        lines.extend([
            f'  __ESBMC_assert({lower} == {box_low} && {upper} == {box_high}, "{label} box source");',
            f'  __ESBMC_assert({box_low} > {q_low} && {box_high} < {q_high}, "{label} clamp inactive from interior box");',
        ])
    else:
        lines.extend([
            f'  __ESBMC_assert({chain_low} >= {q_low} && {chain_high} <= {q_high}, "{label} chain clamp inactivity");',
            f'  __ESBMC_assert({lower} == ({chain_low} > {box_low} ? {chain_low} : {box_low}), "{label} lower intersection");',
            f'  __ESBMC_assert({upper} == ({chain_high} < {box_high} ? {chain_high} : {box_high}), "{label} upper intersection");',
        ])
    return lines


def _affine_contributions(
    csr: dict[str, Any],
    coefficients: dict[int, int],
) -> dict[int, list[tuple[int, int]]]:
    result: dict[int, list[tuple[int, int]]] = {}
    for row, coefficient in coefficients.items():
        if not 0 <= row < csr["rows"]:
            raise ValueError("Affine input coefficient index is outside layer rows")
        for position in range(csr["indptr"][row], csr["indptr"][row + 1]):
            column = csr["indices"][position]
            result.setdefault(column, []).append((csr["data"][position], coefficient))
    return result


def _chunk_columns(
    columns: list[int],
    contributions: dict[int, list[tuple[int, int]]],
    max_terms: int,
) -> list[list[int]]:
    chunks, current, terms = [], [], 0
    for column in columns:
        count = len(contributions.get(column, ()))
        if current and terms + count > max_terms:
            chunks.append(current)
            current, terms = [], 0
        current.append(column)
        terms += count
    if current:
        chunks.append(current)
    return chunks


def render_affine_step(
    certificate: dict[str, Any],
    chain: dict[str, Any],
    step: dict[str, Any],
    *,
    max_terms: int = 25_000,
) -> list[tuple[str, str, dict[str, Any]]]:
    layer_index = int(step["layer_index"])
    csr = certificate["layers_by_index"][layer_index]
    input_coeff = _sparse(step["input_coefficients"], "input_coefficients", csr["rows"])
    output_coeff = _sparse(step["output_coefficients"], "output_coefficients", csr["cols"])
    residual = _sparse(step["residual"], "residual", csr["cols"])
    residual_lower = _sparse(
        step["residual_lower_values"], "residual_lower_values", csr["cols"]
    )
    if any(index not in residual for index in residual_lower):
        raise ValueError(f"{step['step_id']} lower value without residual")
    denominator = 1 << certificate["qif"]["fractional_bits"]
    q_low = -(1 << (certificate["qif"]["total_bits"] - 1))
    q_high = (1 << (certificate["qif"]["total_bits"] - 1)) - 1
    contributions = _affine_contributions(csr, input_coeff)
    columns = sorted(set(contributions) | set(output_coeff) | set(residual) | set(residual_lower))
    chunks = _chunk_columns(columns, contributions, max_terms)
    rendered = []
    expected_partial_total = 0

    for chunk_index, columns_chunk in enumerate(chunks):
        lines = _header()
        lines.append("int main(void) {")
        partial = 0
        term_count = 0
        for ordinal, column in enumerate(columns_chunk):
            terms = contributions.get(column, [])
            term_count += len(terms)
            expression = " + ".join(
                f"((__int128)({weight}) * ({coefficient}))"
                for weight, coefficient in terms
            ) or "0"
            expected_output = output_coeff.get(column, 0)
            expected_residual = residual.get(column, 0)
            if expected_residual != 0 or column in residual_lower:
                if layer_index == 0:
                    source_lower = certificate["input_box"]["low"][column]
                    source_lines: list[str] = []
                else:
                    source_lower, record = _bound_source_value(
                        certificate, layer_index - 1, column
                    )
                    source_lines = _validate_bound_record(record, q_low, q_high)
            else:
                source_lower = 0
                source_lines = []
            exported_lower = residual_lower.get(column, 0)
            partial += expected_residual * exported_lower
            lines.extend([
                f"  __int128 g_{ordinal} = {expression};",
                f"  __int128 r_{ordinal} = g_{ordinal} - (__int128)({expected_output}) * {denominator};",
                f'  __ESBMC_assert(r_{ordinal} == {expected_residual}, "{step["step_id"]} residual {column}");',
                f'  __ESBMC_assert(r_{ordinal} >= 0 && r_{ordinal} < {denominator}, "{step["step_id"]} floor {column}");',
                f'  __ESBMC_assert(g_{ordinal} >= -(__int128){int(step["max_abs_g"])} && g_{ordinal} <= (__int128){int(step["max_abs_g"])}, "{step["step_id"]} g envelope {column}");',
                *source_lines,
                f'  __ESBMC_assert({exported_lower} == {source_lower}, "{step["step_id"]} residual lower {column}");',
            ])
        lines.extend(["  return 0;", "}", ""])
        expected_partial_total += partial
        name = f"{_sanitize(step['step_id'])}_affine_chunk_{chunk_index}.c"
        rendered.append((name, "\n".join(lines), {
            "chain_id": chain["certificate_id"],
            "step_id": step["step_id"],
            "kind": "affine_chunk",
            "chunk_index": chunk_index,
            "multiply_terms": term_count,
            "output_coordinates": len(columns_chunk),
            "expected_partial": partial,
        }))

    bias_dot = sum(
        coefficient * csr["bias"][row] for row, coefficient in input_coeff.items()
    )
    abs_sum = sum(abs(value) for value in input_coeff.values())
    lines = _header()
    lines.extend(["int main(void) {", "  __int128 bias_dot = 0;"])
    for row, coefficient in input_coeff.items():
        lines.append(
            f"  bias_dot += (__int128)({coefficient}) * ({csr['bias'][row]});"
        )
    lines.append("  __int128 residual_dot = 0;")
    for column, residual_value in residual.items():
        exported_lower = residual_lower.get(column, 0)
        lines.append(
            f"  residual_dot += (__int128)({residual_value}) * ({exported_lower});"
        )
    lines.extend([
        f"  __int128 abs_sum = {abs_sum};",
        f"  __int128 rounding_term = (__int128)({denominator // 2}) * abs_sum;",
        f"  __int128 computed = (__int128)({int(step['input_constant'])}) + bias_dot",
        f"      + floor_div_i128(residual_dot - rounding_term, {denominator});",
        f'  __ESBMC_assert(bias_dot == (__int128)({bias_dot}), "{step["step_id"]} bias dot");',
        f'  __ESBMC_assert(residual_dot == (__int128)({expected_partial_total}), "{step["step_id"]} residual dot");',
        f'  __ESBMC_assert(rounding_term == (__int128)({int(step["rounding_term"])}), "{step["step_id"]} rounding term");',
        f'  __ESBMC_assert(computed == (__int128)({int(step["output_constant"])}), "{step["step_id"]} output constant");',
        "  return 0;",
        "}",
        "",
    ])
    rendered.append((f"{_sanitize(step['step_id'])}_affine_close.c", "\n".join(lines), {
        "chain_id": chain["certificate_id"],
        "step_id": step["step_id"],
        "kind": "affine_close",
        "multiply_terms": len(input_coeff) + len(residual),
        "expected_partial": expected_partial_total,
    }))
    return rendered


def render_relu_step(
    certificate: dict[str, Any],
    chain: dict[str, Any],
    step: dict[str, Any],
    *,
    max_coordinates: int = 512,
) -> list[tuple[str, str, dict[str, Any]]]:
    if max_coordinates <= 0:
        raise ValueError("max_coordinates must be positive")
    layer_index = int(step["layer_index"])
    size = certificate["layers_by_index"][layer_index]["rows"]
    input_coeff = _sparse(step["input_coefficients"], "input_coefficients", size)
    output_coeff = _sparse(step["output_coefficients"], "output_coefficients", size)
    constants = _sparse(step["relu_constants"], "relu_constants", size)
    if (
        any(index not in input_coeff for index in output_coeff)
        or any(index not in input_coeff for index in constants)
    ):
        raise ValueError(f"{step['step_id']} output support exceeds input support")
    bound_ids = step.get("bound_ids", [])
    if len(bound_ids) != len(input_coeff):
        raise ValueError(f"{step['step_id']} bound_ids do not cover input support")
    q_low = -(1 << (certificate["qif"]["total_bits"] - 1))
    q_high = (1 << (certificate["qif"]["total_bits"] - 1)) - 1
    coordinates = list(zip(input_coeff, bound_ids, strict=True))
    rendered = []

    for chunk_index, start in enumerate(range(0, len(coordinates), max_coordinates)):
        chunk = coordinates[start:start + max_coordinates]
        lines = _header()
        lines.append("int main(void) {")
        point_evaluations = 0
        bound_free_coordinates = 0
        symbolic_coordinates = 0
        max_abs_coefficient = 0
        for ordinal, (index, bound_id) in enumerate(chunk):
            a = input_coeff[index]
            a_out = output_coeff.get(index, 0)
            exported = constants.get(index, 0)
            bound_free = a > 0 and a_out in {0, a}
            max_abs_coefficient = max(max_abs_coefficient, abs(a), abs(a_out))
            if bound_free:
                bound_free_coordinates += 1
                lines.extend([
                    f'  __ESBMC_assert(({a}) > 0, "{step["step_id"]} bound-free positive coefficient {index}");',
                    f'  __ESBMC_assert(({a_out}) == 0 || ({a_out}) == ({a}), "{step["step_id"]} bound-free slope {index}");',
                    f'  __ESBMC_assert(({exported}) == 0, "{step["step_id"]} bound-free constant {index}");',
                ])
                continue
            record = certificate["bounds_by_id"].get(bound_id)
            if (
                record is None
                or record["layer_index"] != layer_index
                or record["neuron_index"] != index
            ):
                raise ValueError(f"{step['step_id']} has mismatched {bound_id}")
            lines.extend(_validate_bound_record(record, q_low, q_high))
            lower, upper = int(record["lower"]), int(record["upper"])
            f_lower = a * max(lower, 0) - a_out * lower
            f_upper = a * max(upper, 0) - a_out * upper
            expected = min(
                f_lower,
                f_upper,
                0 if lower < 0 < upper else f_lower,
            )
            point_evaluations += 3 if lower < 0 < upper else 2
            symbolic_coordinates += 1
            lines.extend([
                f"  int64_t d_{ordinal} = {f_lower};",
                f"  if ((int64_t){f_upper} < d_{ordinal}) d_{ordinal} = {f_upper};",
            ])
            if lower < 0 < upper:
                lines.append(f"  if (0 < d_{ordinal}) d_{ordinal} = 0;")
            lines.extend([
                f'  __ESBMC_assert(d_{ordinal} == {exported} && d_{ordinal} == {expected}, "{step["step_id"]} ReLU constant {index}");',
                f"  int32_t z_{ordinal} = (int32_t)nondet_int();",
                f"  __ESBMC_assume(z_{ordinal} >= {lower} && z_{ordinal} <= {upper});",
                f"  int64_t value_{ordinal} = (int64_t)({a})",
                f"      * (z_{ordinal} > 0 ? z_{ordinal} : 0)",
                f"      - (int64_t)({a_out}) * z_{ordinal};",
                f'  __ESBMC_assert(value_{ordinal} >= d_{ordinal}, "{step["step_id"]} ReLU envelope {index}");',
            ])
        q_abs = max(abs(q_low), abs(q_high))
        if max_abs_coefficient * q_abs * 2 > (1 << 63) - 1:
            raise ValueError(f"{step['step_id']} ReLU chunk does not fit int64")
        lines.append(
            f'  __ESBMC_assert((__int128){max_abs_coefficient} * {q_abs} * 2 '
            f'< (__int128)INT64_MAX, "{step["step_id"]} int64 product envelope");'
        )
        lines.extend(["  return 0;", "}", ""])
        rendered.append((
            f"{_sanitize(step['step_id'])}_relu_chunk_{chunk_index}.c",
            "\n".join(lines),
            {
                "chain_id": chain["certificate_id"],
                "step_id": step["step_id"],
                "kind": "relu_chunk",
                "chunk_index": chunk_index,
                "coordinates": len(chunk),
                "bound_free_coordinates": bound_free_coordinates,
                "symbolic_coordinates": symbolic_coordinates,
                "point_evaluations": point_evaluations,
                "arithmetic": "int32_domain_int64_product_checked",
                "max_abs_coefficient": max_abs_coefficient,
            },
        ))

    expected_constant = int(step["input_constant"]) + sum(constants.values())
    lines = _header()
    lines.extend([
        "int main(void) {",
        f'  __ESBMC_assert((__int128){int(step["output_constant"])} == (__int128){expected_constant}, "{step["step_id"]} output constant");',
        "  return 0;",
        "}",
        "",
    ])
    rendered.append((
        f"{_sanitize(step['step_id'])}_relu_close.c",
        "\n".join(lines),
        {
            "chain_id": chain["certificate_id"],
            "step_id": step["step_id"],
            "kind": "relu_close",
            "coordinates": len(input_coeff),
        },
    ))
    return rendered

def render_concretize_step(
    certificate: dict[str, Any],
    chain: dict[str, Any],
    step: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    coefficients = _sparse(
        step["input_coefficients"], "input_coefficients",
        len(certificate["input_box"]["low"]),
    )
    scaled = int(step["input_constant"])
    for index, coefficient in coefficients.items():
        endpoint = (
            certificate["input_box"]["low"][index]
            if coefficient >= 0 else certificate["input_box"]["high"][index]
        )
        scaled += coefficient * endpoint
    lines = _header()
    lines.extend([
        "int main(void) {",
        f'  __ESBMC_assert((__int128){int(step["scaled_bound"])} == (__int128){scaled}, "{step["step_id"]} concretize");',
        "  return 0;",
        "}",
        "",
    ])
    return (
        f"{_sanitize(step['step_id'])}_concretize.c",
        "\n".join(lines),
        {
            "chain_id": chain["certificate_id"],
            "step_id": step["step_id"],
            "kind": "concretize",
            "multiply_terms": len(coefficients),
        },
    )


def _property_lambda(
    certificate: dict[str, Any],
    chain: dict[str, Any],
    size: int,
) -> dict[int, int]:
    scale = 1 << certificate["scale_bits"]
    identifier = chain["certificate_id"]
    hidden = re.fullmatch(r"chain/L(\d+)/n(\d+)/(lower|upper)", identifier)
    logit = re.fullmatch(r"chain/logit/(\d+)/(lower|upper)", identifier)
    margin = re.fullmatch(r"chain/margin/(\d+)-(\d+)", identifier)
    if hidden:
        layer, neuron, direction = hidden.groups()
        expected_var = f"z{layer}"
        expected_claim = direction
        result = {int(neuron): scale if direction == "lower" else -scale}
    elif logit:
        neuron, direction = logit.groups()
        expected_var = f"z{len(certificate['layers_by_index']) - 1}"
        expected_claim = direction
        result = {int(neuron): scale if direction == "lower" else -scale}
    elif margin:
        target, competitor = map(int, margin.groups())
        expected_var = f"z{len(certificate['layers_by_index']) - 1}"
        expected_claim = "margin"
        result = {target: scale, competitor: -scale}
    else:
        raise ValueError(f"Unsupported chain property identity {identifier!r}")
    if chain.get("lambda_var") != expected_var:
        raise ValueError(f"{identifier} has lambda_var {chain.get('lambda_var')!r}")
    if chain.get("claim") != expected_claim:
        raise ValueError(f"{identifier} has claim {chain.get('claim')!r}")
    if any(index < 0 or index >= size for index in result):
        raise ValueError(f"{identifier} property index is out of range")
    return result


def render_chain_closure(
    certificate: dict[str, Any],
    chain: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    steps = chain["steps"]
    if not steps:
        raise ValueError(f"{chain['certificate_id']} has no steps")
    variable = re.fullmatch(r"z(\d+)", str(chain.get("lambda_var")))
    if variable is None or int(variable.group(1)) not in certificate["layers_by_index"]:
        raise ValueError(f"{chain['certificate_id']} has an invalid lambda_var")
    size = certificate["layers_by_index"][int(variable.group(1))]["rows"]
    first = _sparse(
        steps[0]["input_coefficients"],
        "first input coefficients",
        size,
    )
    declared_lambda = _sparse(chain["lambda"], "lambda", size)
    property_lambda = _property_lambda(certificate, chain, size)
    lines = _header()
    lines.append("int main(void) {")
    for index in sorted(set(first) | set(declared_lambda) | set(property_lambda)):
        lines.extend([
            f'  __ESBMC_assert((__int128){first.get(index, 0)} == (__int128){declared_lambda.get(index, 0)}, "{chain["certificate_id"]} lambda link {index}");',
            f'  __ESBMC_assert((__int128){declared_lambda.get(index, 0)} == (__int128){property_lambda.get(index, 0)}, "{chain["certificate_id"]} property lambda {index}");',
        ])
    lines.append(
        f'  __ESBMC_assert((__int128){int(steps[0]["input_constant"])} == 0, "{chain["certificate_id"]} initial constant");'
    )
    for left, right in zip(steps, steps[1:]):
        if left["kind"] == "concretize":
            raise ValueError("Concretize must be the final step")
        size = (
            certificate["layers_by_index"][int(left["layer_index"])]["cols"]
            if left["kind"] == "affine"
            else certificate["layers_by_index"][int(left["layer_index"])]["rows"]
        )
        output = _sparse(left["output_coefficients"], "step output", size)
        incoming = _sparse(right["input_coefficients"], "next step input", size)
        for index in sorted(set(output) | set(incoming)):
            lines.append(
                f'  __ESBMC_assert((__int128){output.get(index, 0)} == (__int128){incoming.get(index, 0)}, "{chain["certificate_id"]} link {left["step_id"]} {index}");'
            )
        lines.append(
            f'  __ESBMC_assert((__int128){int(left["output_constant"])} == (__int128){int(right["input_constant"])}, "{chain["certificate_id"]} constant link {left["step_id"]}");'
        )
    final = steps[-1]
    if final["kind"] != "concretize":
        raise ValueError(f"{chain['certificate_id']} does not end in concretize")
    scaled = int(final["scaled_bound"])
    lines.append(
        f'  __ESBMC_assert((__int128){int(chain["scaled_bound"])} == (__int128){scaled}, "{chain["certificate_id"]} scaled bound");'
    )
    scale = 1 << certificate["scale_bits"]
    if chain["claim"] == "upper":
        expected_bound = (-scaled) // scale
    else:
        expected_bound = -((-scaled) // scale)
    lines.extend([
        f'  __ESBMC_assert((__int128){int(chain["bound"])} == (__int128){expected_bound}, "{chain["certificate_id"]} rounded bound");',
        "  return 0;",
        "}",
        "",
    ])
    return (
        f"{_sanitize(chain['certificate_id'])}_chain_close.c",
        "\n".join(lines),
        {"chain_id": chain["certificate_id"], "step_id": "", "kind": "chain_close"},
    )


def render_weighted_rounding_lemma(fractional_bits: int) -> str:
    denominator = 1 << fractional_bits
    half = denominator // 2
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
static __int128 round_half_away(__int128 n, __int128 d) {{
  if (n >= 0) return (n + d / 2) / d;
  return -((-n + d / 2) / d);
}}
int main(void) {{
  int64_t acc64 = nondet_long_long();
  __int128 acc = acc64;
  __int128 error = {denominator} * round_half_away(acc, {denominator}) - acc;
  __ESBMC_assert(error >= -(__int128){half},
                 "round-half-away lower error");
  __ESBMC_assert(error <= (__int128){half},
                 "round-half-away upper error");
  return 0;
}}
"""


def render_relu_breakpoint_lemma(total_bits: int) -> str:
    q_low = -(1 << (total_bits - 1))
    q_high = (1 << (total_bits - 1)) - 1
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
int main(void) {{
  int64_t a64 = nondet_long_long(), ap64 = nondet_long_long();
  int64_t l = nondet_long_long(), u = nondet_long_long(), z = nondet_long_long();
  __ESBMC_assume(l >= {q_low} && u <= {q_high} && l <= u && z >= l && z <= u);
  __int128 a = a64, ap = ap64;
  __int128 fl = a * (l > 0 ? l : 0) - ap * l;
  __int128 fu = a * (u > 0 ? u : 0) - ap * u;
  __int128 d = fl < fu ? fl : fu;
  if (l < 0 && u > 0 && d > 0) d = 0;
  __int128 value = a * (z > 0 ? z : 0) - ap * z;
  __ESBMC_assert(value >= d, "ReLU affine minimum occurs at a breakpoint");
  return 0;
}}
"""



def render_relu_order_lemma(total_bits: int) -> str:
    """Prove ReLU(z) >= 0 and ReLU(z) >= z over the Q domain."""

    q_low = -(1 << (total_bits - 1))
    q_high = (1 << (total_bits - 1)) - 1
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
int nondet_int(void);
int main(void) {{
  int32_t z = (int32_t)nondet_int();
  __ESBMC_assume(z >= {q_low} && z <= {q_high});
  int32_t relu_z = z > 0 ? z : 0;
  __ESBMC_assert(relu_z >= 0, "ReLU is nonnegative");
  __ESBMC_assert(relu_z >= z, "ReLU dominates identity");
  return 0;
}}
"""


def render_positive_product_lemma() -> str:
    """Prove that the checked int64 product of nonnegative factors is nonnegative."""

    return """#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
int nondet_int(void);
int main(void) {
  int32_t coefficient = (int32_t)nondet_int();
  int32_t difference = (int32_t)nondet_int();
  __ESBMC_assume(coefficient > 0);
  __ESBMC_assume(difference >= 0 && difference <= 65535);
  int64_t product = (int64_t)coefficient * difference;
  __ESBMC_assert(product >= 0, "positive int32 times nonnegative Q16 difference");
  return 0;
}
"""


def render_int64_distributivity_lemma(total_bits: int) -> str:
    """Prove the int64 rewrite used by the identity-slope bound-free case."""

    q_low = -(1 << (total_bits - 1))
    q_high = (1 << (total_bits - 1)) - 1
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
int nondet_int(void);
int main(void) {{
  int32_t coefficient = (int32_t)nondet_int();
  int32_t z = (int32_t)nondet_int();
  int32_t relu_z = (int32_t)nondet_int();
  __ESBMC_assume(coefficient > 0);
  __ESBMC_assume(z >= {q_low} && z <= {q_high});
  __ESBMC_assume(relu_z >= 0 && relu_z <= {q_high});
  int64_t left = (int64_t)coefficient * relu_z
                 - (int64_t)coefficient * z;
  int64_t right = (int64_t)coefficient * ((int64_t)relu_z - z);
  __ESBMC_assert(left == right, "int64 distributivity");
  return 0;
}}
"""


def render_bound_free_relu_lemma(total_bits: int) -> str:
    """Prove the two bound-free ReLU cases over the complete integer domain."""

    q_low = -(1 << (total_bits - 1))
    q_high = (1 << (total_bits - 1)) - 1
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
int nondet_int(void);
int main(void) {{
  int32_t a = (int32_t)nondet_int();
  int32_t z = (int32_t)nondet_int();
  __ESBMC_assume(a > 0);
  __ESBMC_assume(z >= {q_low} && z <= {q_high});
  int64_t relu_z = z > 0 ? z : 0;
  int64_t case_zero = (int64_t)a * relu_z;
  int64_t case_identity = (int64_t)a * (relu_z - z);
  __ESBMC_assert(case_zero >= 0,
                 "positive coefficient times ReLU is nonnegative");
  __ESBMC_assert(case_identity >= 0,
                 "positive coefficient times ReLU dominates identity");
  return 0;
}}
"""


def render_generic_relu_envelope_lemma_16bit(total_bits: int) -> str:
    """Prove endpoint minimization for signed-16-bit ReLU coefficients."""

    q_low = -(1 << (total_bits - 1))
    q_high = (1 << (total_bits - 1)) - 1
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
int nondet_int(void);
int main(void) {{
  int16_t a = (int16_t)nondet_int();
  int16_t a_out = (int16_t)nondet_int();
  int32_t lower = (int32_t)nondet_int();
  int32_t upper = (int32_t)nondet_int();
  int32_t z = (int32_t)nondet_int();
  __ESBMC_assume(lower >= {q_low} && upper <= {q_high} && lower <= upper);
  __ESBMC_assume(z >= lower && z <= upper);
  int64_t f_lower = (int64_t)a * (lower > 0 ? lower : 0)
                    - (int64_t)a_out * lower;
  int64_t f_upper = (int64_t)a * (upper > 0 ? upper : 0)
                    - (int64_t)a_out * upper;
  int64_t d = f_lower < f_upper ? f_lower : f_upper;
  if (lower < 0 && upper > 0 && d > 0) d = 0;
  int64_t value = (int64_t)a * (z > 0 ? z : 0)
                  - (int64_t)a_out * z;
  __ESBMC_assert(value >= d,
                 "ReLU affine residual minimum occurs at an endpoint or zero");
  return 0;
}}
"""

def render_margin_closure(certificate: dict[str, Any]) -> str:
    closure = certificate["margin_closure"]
    total_bits = certificate["qif"]["total_bits"]
    q_low = -(1 << (total_bits - 1))
    q_high = (1 << (total_bits - 1)) - 1
    margin = int(closure["raw_margin_lower_bound"])
    target_low = int(closure["raw_target_lower"])
    competitor_high = int(closure["raw_competitor_upper"])
    target = int(closure["target"])
    competitor = int(closure["competitor"])
    chains_by_id = {
        chain["certificate_id"]: chain for chain in certificate["chains"]
    }
    margin_id = f"chain/margin/{target}-{competitor}"
    target_id = f"chain/logit/{target}/lower"
    if margin_id not in chains_by_id or target_id not in chains_by_id:
        raise ValueError("Margin closure is missing its source chains")
    margin_chain_bound = int(chains_by_id[margin_id]["bound"])
    target_chain_bound = int(chains_by_id[target_id]["bound"])
    logit_box = closure.get("logit_post_clamp_box", {})
    competitor_box = logit_box.get(str(competitor))
    if not isinstance(competitor_box, list) or len(competitor_box) != 2:
        raise ValueError("Margin closure is missing the competitor logit box")
    competitor_box_high = int(competitor_box[1])

    top = competitor_high <= q_high - 1
    bottom = target_low >= q_low + 1
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
__int128 nondet_i128(void);
static __int128 clamp_q(__int128 v) {{
  if (v < {q_low}) return {q_low};
  if (v > {q_high}) return {q_high};
  return v;
}}
int main(void) {{
  __int128 raw_target = nondet_i128();
  __int128 raw_competitor = nondet_i128();
  __ESBMC_assert({margin} == {margin_chain_bound}, "margin source chain");
  __ESBMC_assert({target_low} == {target_chain_bound}, "target source chain");
  __ESBMC_assert({competitor_high} == {competitor_box_high}, "competitor source box");
  __ESBMC_assume(raw_target - raw_competitor >= {margin});
  __ESBMC_assume(raw_target >= {target_low});
  __ESBMC_assume(raw_competitor <= {competitor_high});
  __ESBMC_assert({int(top)}, "two-sided top clamp condition");
  __ESBMC_assert({int(bottom)}, "two-sided bottom clamp condition");
  __ESBMC_assert(clamp_q(raw_target) > clamp_q(raw_competitor),
                 "strict deployed margin after two-sided clamp");
  return 0;
}}
"""


def render_all_harnesses(
    certificate: dict[str, Any],
    output: Path,
    *,
    max_affine_terms: int = 25_000,
    max_relu_coordinates: int = 512,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []

    def write(name: str, source: str, metadata: dict[str, Any]) -> None:
        path = output / name
        path.write_text(source, encoding="utf-8")
        records.append({**metadata, "harness": str(path), "source_bytes": path.stat().st_size})

    write(
        "generic_weighted_rounding.c",
        render_weighted_rounding_lemma(certificate["qif"]["fractional_bits"]),
        {"chain_id": "__generic__", "step_id": "", "kind": "weighted_rounding_lemma"},
    )
    for name, source, kind in (
        (
            "generic_relu_order.c",
            render_relu_order_lemma(certificate["qif"]["total_bits"]),
            "relu_order_lemma",
        ),
        (
            "generic_positive_product.c",
            render_positive_product_lemma(),
            "positive_product_lemma",
        ),
        (
            "generic_int64_distributivity.c",
            render_int64_distributivity_lemma(certificate["qif"]["total_bits"]),
            "int64_distributivity_lemma",
        ),
    ):
        write(
            name,
            source,
            {"chain_id": "__generic__", "step_id": "", "kind": kind},
        )
    for chain in certificate["chains"]:
        for step in chain["steps"]:
            if step["kind"] == "affine":
                harnesses = render_affine_step(
                    certificate, chain, step, max_terms=max_affine_terms
                )
            elif step["kind"] == "relu":
                harnesses = render_relu_step(
                    certificate, chain, step,
                    max_coordinates=max_relu_coordinates,
                )
            elif step["kind"] == "concretize":
                harnesses = [render_concretize_step(certificate, chain, step)]
            else:
                raise ValueError(f"Unsupported step kind {step['kind']!r}")
            for name, source, metadata in harnesses:
                write(name, source, metadata)
        name, source, metadata = render_chain_closure(certificate, chain)
        write(name, source, metadata)
    write(
        "margin_two_sided_closure.c",
        render_margin_closure(certificate),
        {"chain_id": "__margin_closure__", "step_id": "", "kind": "margin_closure"},
    )
    manifest = {
        "schema": "crown_chain_harness_manifest_v1",
        "certificate_schema": certificate["schema"],
        "certificate_claim": certificate["claim"],
        "max_affine_terms": max_affine_terms,
        "max_relu_coordinates": max_relu_coordinates,
        "dependency_records": len(certificate["bound_records"]),
        "dependencies_exported": sum(
            record.get("status") != "dependency_not_exported_in_pilot"
            for record in certificate["bound_records"]
        ),
        "conditional": any(
            record.get("status") == "dependency_not_exported_in_pilot"
            for record in certificate["bound_records"]
        ),
        "records": records,
    }
    (output / "harness_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
