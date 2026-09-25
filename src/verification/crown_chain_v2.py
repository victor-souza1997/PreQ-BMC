"""Strict checking and ESBMC obligations for CROWN chain certificate v2.

The exporter is an untrusted certificate finder.  This module validates its
structure, binds its lowered affine operators to the generated deployment C,
and emits the arithmetic obligations that ESBMC must discharge.  A shard can
be checked for throughput measurements, but only the complete certificate may
support an end-to-end result.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from verification.arith_kernel import render_arith_kernel


SCHEMA = "crown_chain_certificate_v2"
Q_LOW = -(1 << 15)
Q_HIGH = (1 << 15) - 1
RELU_CLASSES = ("bound_free", "nosat_upper", "lower", "upper", "lower_upper")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ints(values: Iterable[Any], name: str) -> list[int]:
    result: list[int] = []
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
    if indices != sorted(indices):
        raise ValueError(f"{name} indices are not sorted")
    if any(index < 0 or index >= size for index in indices):
        raise ValueError(f"{name} has an out-of-range index")
    if any(value == 0 for value in values):
        raise ValueError(f"{name} stores an explicit zero")
    return dict(zip(indices, values, strict=True))


def _sparse_payload(values: dict[int, int]) -> dict[str, list[int]]:
    entries = sorted((index, value) for index, value in values.items() if value)
    return {
        "indices": [index for index, _ in entries],
        "values": [value for _, value in entries],
    }


def _load_csr(layer: dict[str, Any]) -> dict[str, Any]:
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
    for row in range(rows):
        row_indices = indices[indptr[row]:indptr[row + 1]]
        if row_indices != sorted(row_indices) or len(row_indices) != len(set(row_indices)):
            raise ValueError(f"CSR row {row} is not canonical")
    hashed = {
        "rows": rows, "cols": cols, "indptr": indptr,
        "indices": indices, "data": data, "bias": bias,
    }
    digest = hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest()
    if digest != csr.get("sha256"):
        raise ValueError(f"Lowered CSR hash mismatch for layer {layer['layer_index']}")
    return {**hashed, "sha256": digest}


def _csr_times_transpose(csr: dict[str, Any], coefficients: dict[int, int]) -> dict[int, int]:
    result: dict[int, int] = {}
    for row, coefficient in coefficients.items():
        for position in range(csr["indptr"][row], csr["indptr"][row + 1]):
            column = csr["indices"][position]
            result[column] = result.get(column, 0) + csr["data"][position] * coefficient
    return {index: value for index, value in result.items() if value}


def _array(source: str, name: str) -> list[int]:
    match = re.search(
        rf"static const (?:int64_t|int) {re.escape(name)}\[[^\]]+\] = \{{([^}}]*)\}};",
        source,
    )
    if match is None:
        raise ValueError(f"Deployment array {name} is missing")
    return [int(value.strip()) for value in match.group(1).split(",") if value.strip()]


def _number(block: str, pattern: str, label: str) -> int:
    match = re.search(pattern, block)
    if match is None:
        raise ValueError(f"Cannot parse deployment {label}")
    return int(match.group(1))


def _conv_geometry(source: str, layer: int) -> dict[str, int]:
    marker = f"LAYER_{layer}_KERNEL[kernel_index]"
    end = source.find(marker)
    if end < 0:
        raise ValueError(f"Deployment convolution layer {layer} is missing")
    start = source.rfind("for (int oy = 0;", 0, end)
    if start < 0:
        raise ValueError(f"Deployment convolution layer {layer} has no output loop")
    block = source[start:end]
    values = {
        "oh": _number(block, r"for \(int oy = 0; oy < (\d+);", "output height"),
        "ow": _number(block, r"for \(int ox = 0; ox < (\d+);", "output width"),
        "oc": _number(block, r"for \(int oc = 0; oc < (\d+);", "output channels"),
        "kh": _number(block, r"for \(int ky = 0; ky < (\d+);", "kernel height"),
        "kw": _number(block, r"for \(int kx = 0; kx < (\d+);", "kernel width"),
        "sh": _number(block, r"const int iy = oy \* (\d+) \+ ky", "vertical stride"),
        "sw": _number(block, r"const int ix = ox \* (\d+) \+ kx", "horizontal stride"),
        "ph": _number(block, r"const int iy = oy \* \d+ \+ ky - (\d+);", "vertical padding"),
        "pw": _number(block, r"const int ix = ox \* \d+ \+ kx - (\d+);", "horizontal padding"),
        "ih": _number(block, r"iy >= (\d+)\) continue", "input height"),
        "iw": _number(block, r"ix >= (\d+)\) continue", "input width"),
        "ic": _number(block, r"for \(int ic = 0; ic < (\d+);", "input channels"),
    }
    return values


def bind_layers_to_deployment(
    layers: dict[int, dict[str, Any]], deployment_path: Path, expected_sha256: str
) -> dict[str, Any]:
    """Require exact ordered equality between every CSR row and deployment C."""

    actual_sha256 = _sha256(deployment_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Deployment C hash does not match the certificate")
    source = deployment_path.read_text(encoding="utf-8")
    bound_rows = 0
    bound_terms = 0
    layer_records = []
    for layer_index, csr in sorted(layers.items()):
        bias = _array(source, f"LAYER_{layer_index}_BIAS")
        if f"LAYER_{layer_index}_KERNEL" in source:
            geometry = _conv_geometry(source, layer_index)
            kernel = _array(source, f"LAYER_{layer_index}_KERNEL")
            expected_kernel = (
                geometry["kh"] * geometry["kw"] * geometry["ic"] * geometry["oc"]
            )
            if len(kernel) != expected_kernel:
                raise ValueError(f"Deployment layer {layer_index} kernel size mismatch")
            if csr["rows"] != geometry["oh"] * geometry["ow"] * geometry["oc"]:
                raise ValueError(f"Deployment layer {layer_index} output shape mismatch")
            if csr["cols"] != geometry["ih"] * geometry["iw"] * geometry["ic"]:
                raise ValueError(f"Deployment layer {layer_index} input shape mismatch")
            if len(bias) != geometry["oc"]:
                raise ValueError(f"Deployment layer {layer_index} bias size mismatch")
            for oy in range(geometry["oh"]):
                for ox in range(geometry["ow"]):
                    for oc in range(geometry["oc"]):
                        row = (oy * geometry["ow"] + ox) * geometry["oc"] + oc
                        expected_indices: list[int] = []
                        expected_values: list[int] = []
                        for ky in range(geometry["kh"]):
                            iy = oy * geometry["sh"] + ky - geometry["ph"]
                            if not 0 <= iy < geometry["ih"]:
                                continue
                            for kx in range(geometry["kw"]):
                                ix = ox * geometry["sw"] + kx - geometry["pw"]
                                if not 0 <= ix < geometry["iw"]:
                                    continue
                                for ic in range(geometry["ic"]):
                                    kernel_index = (
                                        ((ky * geometry["kw"] + kx) * geometry["ic"] + ic)
                                        * geometry["oc"] + oc
                                    )
                                    weight = kernel[kernel_index]
                                    if weight:
                                        expected_indices.append(
                                            (iy * geometry["iw"] + ix) * geometry["ic"] + ic
                                        )
                                        expected_values.append(weight)
                        left, right = csr["indptr"][row], csr["indptr"][row + 1]
                        if csr["indices"][left:right] != expected_indices:
                            raise ValueError(
                                f"Deployment layer {layer_index} row {row} column-order mismatch"
                            )
                        if csr["data"][left:right] != expected_values:
                            raise ValueError(
                                f"Deployment layer {layer_index} row {row} weight mismatch"
                            )
                        if csr["bias"][row] != bias[oc]:
                            raise ValueError(f"Deployment layer {layer_index} row {row} bias mismatch")
            kind = "conv"
            detail: dict[str, Any] = geometry
        else:
            weights = _array(source, f"LAYER_{layer_index}_WEIGHTS")
            if len(weights) != csr["rows"] * csr["cols"] or len(bias) != csr["rows"]:
                raise ValueError(f"Deployment dense layer {layer_index} shape mismatch")
            for row in range(csr["rows"]):
                expected = [
                    (column, weights[row * csr["cols"] + column])
                    for column in range(csr["cols"])
                    if weights[row * csr["cols"] + column]
                ]
                left, right = csr["indptr"][row], csr["indptr"][row + 1]
                if csr["indices"][left:right] != [index for index, _ in expected]:
                    raise ValueError(f"Deployment dense layer {layer_index} column mismatch")
                if csr["data"][left:right] != [value for _, value in expected]:
                    raise ValueError(f"Deployment dense layer {layer_index} weight mismatch")
                if csr["bias"][row] != bias[row]:
                    raise ValueError(f"Deployment dense layer {layer_index} bias mismatch")
            kind = "dense"
            detail = {"inputs": csr["cols"], "outputs": csr["rows"]}
        # The hash binds the full source; these checks reject a source with different arithmetic.
        if source.count(f"LAYER_{layer_index}_BIAS") < 2:
            raise ValueError(f"Deployment layer {layer_index} bias is not used")
        bound_rows += csr["rows"]
        bound_terms += len(csr["data"])
        layer_records.append({
            "layer_index": layer_index,
            "kind": kind,
            "rows": csr["rows"],
            "columns": csr["cols"],
            "nonzero_terms": len(csr["data"]),
            **detail,
        })
    if "div_round_half_away_from_zero_i128" not in source or "clamp_to_signed_range_i128" not in source:
        raise ValueError("Deployment C does not use the required arithmetic kernel")
    return {
        "status": "EXACT_ORDERED_MATCH",
        "deployment_c": str(deployment_path),
        "deployment_sha256": actual_sha256,
        "layers": layer_records,
        "rows_compared": bound_rows,
        "nonzero_terms_compared": bound_terms,
    }


@dataclass
class CertificateV2:
    root_dir: Path
    root: dict[str, Any]
    layers: dict[int, dict[str, Any]]
    facts: dict[str, dict[str, Any]]
    chain_bounds: dict[str, int]
    chains: list[dict[str, Any]]
    selected_layers: tuple[int, ...]
    preflight: dict[str, Any]


def _parse_harness_arrays(path: Path) -> dict[str, list[int]]:
    source = path.read_text(encoding="utf-8")
    pattern = re.compile(r"static const (?:int64_t|int) (\w+)\[[^\]]*\] = \{([^}]*)\};")
    return {
        match.group(1): [int(value) for value in match.group(2).split(",")]
        for match in pattern.finditer(source)
    }


def _validate_ledger(root: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    deployment = root["deployment"]["qnn_conv_native_c_sha256"]
    blocks: dict[str, dict[str, Any]] = {}
    for digest, metadata in root["box_blocks"].items():
        cache_path = Path(metadata["cache_file"])
        harness_path = Path(metadata["harness"])
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("status") != "VERIFIED"
            or cached.get("digest") != digest
            or cached.get("identity") != metadata["identity"]
            or metadata["identity"].get("deployment") != deployment
            or _sha256(harness_path) != metadata["harness_sha256"]
        ):
            raise ValueError(f"Box ledger entry {digest} is not a matching VERIFIED proof")
        blocks[digest] = {
            "identity": metadata["identity"],
            "arrays": _parse_harness_arrays(harness_path),
        }
    summary_path = Path(root["provenance"]["box_ledger"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    encoder = root["deployment"]["encoder"]
    matching = [
        obligation for obligation in summary["obligations"]
        if obligation["digest"] == encoder["obligation_digest"]
    ]
    if len(matching) != 1 or matching[0]["status"] != "VERIFIED" or matching[0]["identity"] != encoder["identity"]:
        raise ValueError("Input-encoding obligation is not bound to one VERIFIED ledger entry")
    return blocks, {
        "box_blocks_verified": len(blocks),
        "encoder_status": "VERIFIED",
        "encoder_digest": encoder["obligation_digest"],
        "ledger": str(summary_path),
    }


def _validate_facts(
    root: dict[str, Any], facts: dict[str, dict[str, Any]],
    chains: dict[str, int], blocks: dict[str, dict[str, Any]], layer_count: int,
) -> None:
    for fact_id, fact in facts.items():
        expected_id = f"fact/L{fact['layer_index']}/n{fact['neuron_index']}/{fact['side']}"
        if fact_id != expected_id or fact["side"] not in {"lower", "upper"}:
            raise ValueError(f"Malformed fact identity {fact_id}")
        source = fact["source"]
        if source["kind"] == "chain":
            if chains.get(source["chain_id"]) != fact["value"]:
                raise ValueError(f"{fact_id} does not equal its chain bound")
            continue
        if source.get("kind") != "box" or source.get("digest") not in blocks:
            raise ValueError(f"{fact_id} has an unknown source")
        block = blocks[source["digest"]]
        slot = int(source["slot"])
        identity = block["identity"]
        if (
            identity["layer_index"] != fact["layer_index"]
            or identity["output_indices"][slot] != fact["neuron_index"]
        ):
            raise ValueError(f"{fact_id} points to the wrong ledger coordinate")
        arrays = block["arrays"]
        if fact["side"] == "upper":
            value = arrays["EXPECTED_HIGH"][slot]
            if fact["value"] != value or value >= Q_HIGH or source["field"] != "EXPECTED_HIGH":
                raise ValueError(f"{fact_id} is not a proved raw upper bound")
        else:
            value = arrays["EXPECTED_LOW"][slot]
            stable = arrays.get("STABLE_RELU", [0] * len(arrays["EXPECTED_LOW"]))[slot]
            derivable = stable > 0 or value > 0 or fact["layer_index"] == layer_count - 1
            if (
                fact["value"] != value or value <= Q_LOW or not derivable
                or source["field"] != "EXPECTED_LOW"
            ):
                raise ValueError(f"{fact_id} is not a proved raw lower bound")


def _chain_property(chain: dict[str, Any], scale: int, layer_count: int) -> dict[int, int]:
    identifier = chain["certificate_id"]
    hidden = re.fullmatch(r"chain/L(\d+)/n(\d+)/(lower|upper)", identifier)
    logit = re.fullmatch(r"chain/logit/(\d+)/(lower|upper)", identifier)
    margin = re.fullmatch(r"chain/margin/(\d+)-(\d+)", identifier)
    if hidden:
        layer, neuron, direction = hidden.groups()
        if int(layer) != chain["lambda_layer"] or chain["claim"] != direction:
            raise ValueError(f"{identifier} property metadata mismatch")
        return {int(neuron): scale if direction == "lower" else -scale}
    if logit:
        neuron, direction = logit.groups()
        if chain["lambda_layer"] != layer_count - 1 or chain["claim"] != direction:
            raise ValueError(f"{identifier} logit metadata mismatch")
        return {int(neuron): scale if direction == "lower" else -scale}
    if margin:
        target, competitor = map(int, margin.groups())
        if chain["lambda_layer"] != layer_count - 1 or chain["claim"] != "margin":
            raise ValueError(f"{identifier} margin metadata mismatch")
        return {target: scale, competitor: -scale}
    raise ValueError(f"Unsupported chain identity {identifier}")


def _fact_value(facts: dict[str, dict[str, Any]], layer: int, index: int, side: str) -> int:
    fact_id = f"fact/L{layer}/n{index}/{side}"
    if fact_id not in facts:
        raise ValueError(f"Missing {fact_id}")
    return int(facts[fact_id]["value"])


def validate_chain(
    chain: dict[str, Any], layers: dict[int, dict[str, Any]],
    facts: dict[str, dict[str, Any]], low: list[int], high: list[int], scale_bits: int,
) -> dict[str, Any]:
    steps = chain.get("steps", [])
    if not steps or steps[-1].get("kind") != "concretize":
        raise ValueError(f"{chain.get('certificate_id')} has no terminal concretize step")
    scale = 1 << scale_bits
    current = _chain_property(chain, scale, len(layers))
    constant = 0
    if _sparse(steps[0]["input_coefficients"], "initial coefficients", layers[chain["lambda_layer"]]["rows"]) != current:
        raise ValueError(f"{chain['certificate_id']} initial coefficients do not state its property")
    affine_terms = 0
    relu_coordinates = 0
    normalized_steps = []
    for ordinal, step in enumerate(steps):
        if int(step["input_constant"]) != constant:
            raise ValueError(f"{step['step_id']} breaks the constant chain")
        normalized = {**step, "_input_coefficients": current}
        if step["kind"] == "affine":
            layer_index = int(step["layer_index"])
            csr = layers[layer_index]
            if ordinal == 0 and layer_index != chain["lambda_layer"]:
                raise ValueError(f"{step['step_id']} starts at the wrong layer")
            g = _csr_times_transpose(csr, current)
            denominator = 1 << 8
            output = {index: value // denominator for index, value in g.items()}
            output = {index: value for index, value in output.items() if value}
            exported_output = _sparse(
                step["output_coefficients"], "output_coefficients", csr["cols"]
            )
            if output != exported_output:
                raise ValueError(f"{step['step_id']} output coefficients mismatch")
            residual = {index: value - denominator * output.get(index, 0) for index, value in g.items()}
            if any(not 0 <= value < denominator for value in residual.values()):
                raise ValueError(f"{step['step_id']} has an invalid floor residual")
            sources = step["residual_lower_sources"]
            lower_values: dict[int, int] = {}
            if layer_index == 0:
                if sources != "input_box.low":
                    raise ValueError(f"{step['step_id']} does not cite the input lower box")
                lower_values = {index: low[index] for index in residual}
            else:
                indices = _ints(sources["indices"], "residual source indices")
                fact_ids = list(sources["fact_ids"])
                if len(indices) != len(fact_ids) or len(set(indices)) != len(indices):
                    raise ValueError(f"{step['step_id']} residual sources are malformed")
                for index, fact_id in zip(indices, fact_ids, strict=True):
                    expected = f"fact/L{layer_index - 1}/n{index}/lower"
                    if fact_id != expected:
                        raise ValueError(f"{step['step_id']} cites {fact_id} for coordinate {index}")
                    value = _fact_value(facts, layer_index - 1, index, "lower")
                    if value <= 0:
                        raise ValueError(f"{step['step_id']} cites a nonpositive residual lower fact")
                    lower_values[index] = value
            bias_dot = sum(coefficient * csr["bias"][row] for row, coefficient in current.items())
            residual_dot = sum(value * lower_values.get(index, 0) for index, value in residual.items())
            rounding_term = (denominator // 2) * sum(abs(value) for value in current.values())
            if rounding_term != int(step["rounding_term"]):
                raise ValueError(f"{step['step_id']} rounding term mismatch")
            constant = constant + bias_dot + (residual_dot - rounding_term) // denominator
            if constant != int(step["output_constant"]):
                raise ValueError(f"{step['step_id']} affine constant mismatch")
            normalized.update(
                _g=g, _residual=residual, _lower_values=lower_values,
                _bias_dot=bias_dot, _residual_dot=residual_dot,
            )
            affine_terms += sum(
                csr["indptr"][row + 1] - csr["indptr"][row] for row in current
            )
            current = output
        elif step["kind"] == "relu":
            layer_index = int(step["layer_index"])
            size = layers[layer_index]["rows"]
            output = _sparse(step["output_coefficients"], "output_coefficients", size)
            constants = _sparse(step["relu_constants"], "relu_constants", size)
            classes = step["coordinate_classes"]
            if set(classes) != set(RELU_CLASSES):
                raise ValueError(f"{step['step_id']} has unknown ReLU classes")
            seen: set[int] = set()
            derived_total = 0
            fact_values: dict[int, tuple[int | None, int | None]] = {}
            for class_name in RELU_CLASSES:
                for index in _ints(classes[class_name], f"{class_name} coordinates"):
                    if index in seen or index not in current:
                        raise ValueError(f"{step['step_id']} class coverage is invalid at {index}")
                    seen.add(index)
                    a, a_out = current[index], output.get(index, 0)
                    expected_class = (
                        "bound_free" if a > 0 and a_out == 0 else
                        "nosat_upper" if a > 0 and a_out == a else
                        "lower" if a < 0 and a_out == a else
                        "upper" if a < 0 and a_out == 0 else "lower_upper"
                    )
                    if class_name != expected_class:
                        raise ValueError(f"{step['step_id']} misclassifies ReLU coordinate {index}")
                    lower_value: int | None = None
                    upper_value: int | None = None
                    if class_name == "bound_free":
                        derived = 0
                    elif class_name == "nosat_upper":
                        upper_value = _fact_value(facts, layer_index, index, "upper")
                        if upper_value > Q_HIGH:
                            raise ValueError(f"{step['step_id']} lacks no-upper-saturation at {index}")
                        derived = 0
                    elif class_name == "lower":
                        lower_value = _fact_value(facts, layer_index, index, "lower")
                        derived = a * max(-lower_value, 0)
                    elif class_name == "upper":
                        upper_value = _fact_value(facts, layer_index, index, "upper")
                        derived = a * max(upper_value, 0)
                    else:
                        lower_value = _fact_value(facts, layer_index, index, "lower")
                        upper_value = _fact_value(facts, layer_index, index, "upper")
                        if lower_value < Q_LOW or upper_value > Q_HIGH or lower_value > upper_value:
                            raise ValueError(f"{step['step_id']} two-sided fact is outside Q16")
                        points = [lower_value, upper_value]
                        if lower_value < 0 < upper_value:
                            points.append(0)
                        derived = min(a * max(point, 0) - a_out * point for point in points)
                    if constants.get(index, 0) != derived:
                        raise ValueError(f"{step['step_id']} ReLU constant mismatch at {index}")
                    derived_total += derived
                    fact_values[index] = (lower_value, upper_value)
            if seen != set(current):
                raise ValueError(f"{step['step_id']} classes do not cover the input support")
            if any(index not in current for index in output) or any(index not in current for index in constants):
                raise ValueError(f"{step['step_id']} ReLU output exceeds input support")
            constant += derived_total
            if constant != int(step["output_constant"]):
                raise ValueError(f"{step['step_id']} ReLU constant sum mismatch")
            normalized["_fact_values"] = fact_values
            current = output
            relu_coordinates += len(seen)
        elif step["kind"] == "concretize":
            scaled = constant + sum(
                coefficient * (low[index] if coefficient >= 0 else high[index])
                for index, coefficient in current.items()
            )
            if scaled != int(step["scaled_bound"]) or scaled != int(chain["scaled_bound"]):
                raise ValueError(f"{step['step_id']} concretization mismatch")
            expected_bound = (
                (-scaled) // scale if chain["claim"] == "upper"
                else -((-scaled) // scale)
            )
            if expected_bound != int(chain["bound"]):
                raise ValueError(f"{chain['certificate_id']} bound rounding mismatch")
        else:
            raise ValueError(f"{step['step_id']} has an unknown kind")
        normalized_steps.append(normalized)
    chain["_normalized_steps"] = normalized_steps
    return {"affine_terms": affine_terms, "relu_coordinates": relu_coordinates}


def load_certificate_v2(
    root_dir: Path | str, *, selected_layers: Iterable[int] | None = None,
    deployment_c: Path | str,
) -> CertificateV2:
    root_dir = Path(root_dir)
    root_path = root_dir / "certificate.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    if root.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported certificate schema {root.get('schema')!r}")
    layers_path = root_dir / root["deployment"]["layers_file"]
    if _sha256(layers_path) != root["deployment"]["layers_file_sha256"]:
        raise ValueError("layers.json hash mismatch")
    layer_rows = json.loads(layers_path.read_text(encoding="utf-8"))
    layers = {int(row["layer_index"]): _load_csr(row) for row in layer_rows}
    if set(layers) != set(range(len(layers))):
        raise ValueError("Layers are not consecutively indexed")
    lowered_digest = hashlib.sha256(
        "".join(layers[index]["sha256"] for index in sorted(layers)).encode()
    ).hexdigest()
    if lowered_digest != root["deployment"]["lowered_constants_sha256"]:
        raise ValueError("Combined lowered-constant hash mismatch")
    low = _ints(root["input_box"]["low"], "input_box.low")
    high = _ints(root["input_box"]["high"], "input_box.high")
    if len(low) != layers[0]["cols"] or len(low) != len(high) or any(a > b for a, b in zip(low, high)):
        raise ValueError("Invalid input box")

    shards_by_layer: dict[int, dict[str, Any]] = {}
    chain_bounds: dict[str, int] = {}
    chain_locations: dict[str, int] = {}
    for shard in root["shards"]:
        path = root_dir / shard["file"]
        if _sha256(path) != shard["sha256"] or path.stat().st_size != shard["bytes"]:
            raise ValueError(f"Shard integrity mismatch: {path}")
        layer = int(shard["layer"])
        if layer in shards_by_layer:
            raise ValueError(f"Duplicate shard layer {layer}")
        shards_by_layer[layer] = shard
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                chain = json.loads(line)
                identifier = chain["certificate_id"]
                if identifier in chain_bounds:
                    raise ValueError(f"Duplicate chain {identifier}")
                chain_bounds[identifier] = int(chain["bound"])
                chain_locations[identifier] = layer
    expected_count = int(root["manifest"]["diagnostic_counts"]["chains"])
    if len(chain_bounds) != expected_count:
        raise ValueError(f"Expected {expected_count} chains, found {len(chain_bounds)}")
    selected = tuple(sorted(shards_by_layer if selected_layers is None else set(selected_layers)))
    if any(layer not in shards_by_layer for layer in selected):
        raise ValueError("Selected certificate shard does not exist")

    facts = {fact["fact_id"]: fact for fact in root["facts"]}
    if len(facts) != len(root["facts"]):
        raise ValueError("Duplicate fact ID")
    blocks, ledger_summary = _validate_ledger(root)
    _validate_facts(root, facts, chain_bounds, blocks, len(layers))
    binding = bind_layers_to_deployment(
        layers, Path(deployment_c), root["deployment"]["qnn_conv_native_c_sha256"]
    )

    chains: list[dict[str, Any]] = []
    totals = {"affine_terms": 0, "relu_coordinates": 0}
    for layer in selected:
        path = root_dir / shards_by_layer[layer]["file"]
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                chain = json.loads(line)
                counts = validate_chain(chain, layers, facts, low, high, int(root["scale_bits"]))
                for key, value in counts.items():
                    totals[key] += value
                chains.append(chain)

    complete = set(selected) == set(shards_by_layer)
    closure_status = "NOT_IN_SHARD_SCOPE"
    if complete:
        required = set(root["manifest"]["root_chains"])
        if not required.issubset(chain_bounds):
            raise ValueError("Root-chain manifest is incomplete")
        target = int(root["margin_closure"]["target"])
        competitors = _ints(root["manifest"]["required_competitors"], "required competitors")
        if sorted(competitors) != [index for index in range(layers[len(layers) - 1]["rows"]) if index != target]:
            raise ValueError("Competitor manifest is incomplete")
        for competitor in competitors:
            margin_id = f"chain/margin/{target}-{competitor}"
            record = root["margin_closure"]["competitors"][str(competitor)]
            if chain_bounds[margin_id] != record["raw_margin_lower_bound"] or chain_bounds[margin_id] < 1:
                raise ValueError(f"Margin closure failed for competitor {competitor}")
            for side in ("top", "bottom"):
                edge = record[side]
                reference = edge["fact"]
                value = facts[reference]["value"] if reference.startswith("fact/") else chain_bounds[reference]
                if value != edge["value"]:
                    raise ValueError(f"Margin {competitor} {side} reference mismatch")
                if side == "top" and edge["rule"] == "raw_c <= 32766" and value > Q_HIGH - 1:
                    raise ValueError(f"Margin {competitor} lacks top clamp separation")
                if side == "top" and edge["rule"] == "raw_t <= 32767" and value > Q_HIGH:
                    raise ValueError(f"Margin {competitor} lacks target upper bound")
                if side == "bottom" and edge["rule"] == "raw_t >= -32767" and value < Q_LOW + 1:
                    raise ValueError(f"Margin {competitor} lacks bottom clamp separation")
                if side == "bottom" and edge["rule"] == "raw_c >= -32768" and value < Q_LOW:
                    raise ValueError(f"Margin {competitor} lacks competitor lower bound")
            if not record["closed"]:
                raise ValueError(f"Margin {competitor} is marked open")
        closure_status = "STRUCTURALLY_CLOSED"
    preflight = {
        "schema": SCHEMA,
        "selected_layers": list(selected),
        "complete_certificate_scope": complete,
        "chains_loaded": len(chains),
        "chains_indexed": len(chain_bounds),
        "facts_validated": len(facts),
        "arithmetic_counts": totals,
        "ledger": ledger_summary,
        "deployment_binding": binding,
        "margin_closure": closure_status,
    }
    return CertificateV2(
        root_dir=root_dir, root=root, layers=layers, facts=facts,
        chain_bounds=chain_bounds, chains=chains, selected_layers=selected,
        preflight=preflight,
    )


def _header() -> list[str]:
    return [
        "#include <stdint.h>",
        "#include <limits.h>",
        "void __ESBMC_assume(_Bool);",
        "void __ESBMC_assert(_Bool, const char *);",
        "long long nondet_long_long(void);",
        "static __int128 floor_div_i128(__int128 n, __int128 d) {",
        "  __int128 q = n / d, r = n % d;",
        "  return q - (r != 0 && n < 0);",
        "}",
        render_arith_kernel(),
        "",
    ]


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _chunks(items: list[int], cost: dict[int, int], maximum: int) -> list[list[int]]:
    result: list[list[int]] = []
    current: list[int] = []
    used = 0
    for item in items:
        item_cost = cost.get(item, 0)
        if current and used + item_cost > maximum:
            result.append(current)
            current, used = [], 0
        current.append(item)
        used += item_cost
    if current:
        result.append(current)
    return result


def _render_affine(
    certificate: CertificateV2, chain: dict[str, Any], step: dict[str, Any], max_terms: int,
) -> list[tuple[str, str, dict[str, Any]]]:
    layer_index = int(step["layer_index"])
    csr = certificate.layers[layer_index]
    coefficients: dict[int, int] = step["_input_coefficients"]
    output = _sparse(step["output_coefficients"], "output_coefficients", csr["cols"])
    g: dict[int, int] = step["_g"]
    residual: dict[int, int] = step["_residual"]
    lower_values: dict[int, int] = step["_lower_values"]
    contributions: dict[int, list[tuple[int, int]]] = {}
    for row, coefficient in coefficients.items():
        for position in range(csr["indptr"][row], csr["indptr"][row + 1]):
            column = csr["indices"][position]
            contributions.setdefault(column, []).append((csr["data"][position], coefficient))
    columns = sorted(set(g) | set(output))
    rendered = []
    cost = {column: len(contributions.get(column, ())) for column in columns}
    for chunk_index, chunk in enumerate(_chunks(columns, cost, max_terms)):
        lines = _header() + ["int main(void) {"]
        terms = 0
        for ordinal, column in enumerate(chunk):
            expression = " + ".join(
                f"((__int128)({weight}) * ({coefficient}))"
                for weight, coefficient in contributions.get(column, ())
            ) or "0"
            expected_g = g.get(column, 0)
            expected_output = output.get(column, 0)
            expected_residual = residual.get(column, 0)
            terms += len(contributions.get(column, ()))
            lines.extend([
                f"  __int128 g_{ordinal} = {expression};",
                f'  __ESBMC_assert(g_{ordinal} == (__int128)({expected_g}), "{step["step_id"]} g {column}");',
                f"  __int128 q_{ordinal} = floor_div_i128(g_{ordinal}, 256);",
                f'  __ESBMC_assert(q_{ordinal} == (__int128)({expected_output}), "{step["step_id"]} floor {column}");',
                f'  __ESBMC_assert(g_{ordinal} - 256 * q_{ordinal} == (__int128)({expected_residual}), "{step["step_id"]} residual {column}");',
            ])
        lines.extend(["  return 0;", "}", ""])
        rendered.append((
            f"{_safe(step['step_id'])}_affine_{chunk_index}.c", "\n".join(lines),
            {"chain_id": chain["certificate_id"], "step_id": step["step_id"],
             "kind": "affine_chunk", "multiply_terms": terms},
        ))
    lines = _header() + ["int main(void) {", "  __int128 bias_dot = 0;"]
    for row, coefficient in coefficients.items():
        lines.append(f"  bias_dot += (__int128)({coefficient}) * ({csr['bias'][row]});")
    lines.append("  __int128 residual_dot = 0;")
    for column, value in residual.items():
        lines.append(f"  residual_dot += (__int128)({value}) * ({lower_values.get(column, 0)});")
    lines.extend([
        f'  __ESBMC_assert(bias_dot == (__int128)({step["_bias_dot"]}), "{step["step_id"]} bias dot");',
        f'  __ESBMC_assert(residual_dot == (__int128)({step["_residual_dot"]}), "{step["step_id"]} residual dot");',
        f"  __int128 computed = (__int128)({step['input_constant']}) + bias_dot",
        f"      + floor_div_i128(residual_dot - (__int128)({step['rounding_term']}), 256);",
        f'  __ESBMC_assert(computed == (__int128)({step["output_constant"]}), "{step["step_id"]} output constant");',
        "  return 0;", "}", "",
    ])
    rendered.append((
        f"{_safe(step['step_id'])}_affine_close.c", "\n".join(lines),
        {"chain_id": chain["certificate_id"], "step_id": step["step_id"],
         "kind": "affine_close", "multiply_terms": len(coefficients) + len(residual)},
    ))
    return rendered


def _render_relu(
    certificate: CertificateV2, chain: dict[str, Any], step: dict[str, Any], max_coordinates: int,
) -> list[tuple[str, str, dict[str, Any]]]:
    coefficients: dict[int, int] = step["_input_coefficients"]
    layer_index = int(step["layer_index"])
    output = _sparse(
        step["output_coefficients"], "output_coefficients",
        certificate.layers[layer_index]["rows"],
    )
    constants = _sparse(
        step["relu_constants"], "relu_constants", certificate.layers[layer_index]["rows"]
    )
    class_by_index = {
        index: class_name
        for class_name, indices in step["coordinate_classes"].items()
        for index in indices
    }
    indices = sorted(coefficients)
    rendered = []
    for chunk_index, start in enumerate(range(0, len(indices), max_coordinates)):
        chunk = indices[start:start + max_coordinates]
        lines = _header() + ["int main(void) {"]
        counts = {name: 0 for name in RELU_CLASSES}
        for ordinal, index in enumerate(chunk):
            a = coefficients[index]
            a_out = output.get(index, 0)
            constant = constants.get(index, 0)
            class_name = class_by_index[index]
            lower, upper = step["_fact_values"][index]
            counts[class_name] += 1
            if class_name == "bound_free":
                predicate = f"({a}) > 0 && ({a_out}) == 0 && ({constant}) == 0"
            elif class_name == "nosat_upper":
                predicate = (
                    f"({a}) > 0 && ({a_out}) == ({a}) && ({upper}) <= {Q_HIGH} "
                    f"&& ({constant}) == 0"
                )
            elif class_name == "lower":
                expected = a * max(-int(lower), 0)
                predicate = (
                    f"({a}) < 0 && ({a_out}) == ({a}) "
                    f"&& (__int128)({constant}) == (__int128)({expected})"
                )
            elif class_name == "upper":
                expected = a * max(int(upper), 0)
                predicate = (
                    f"({a}) < 0 && ({a_out}) == 0 "
                    f"&& (__int128)({constant}) == (__int128)({expected})"
                )
            else:
                points = [int(lower), int(upper)]
                if int(lower) < 0 < int(upper):
                    points.append(0)
                expected = min(a * max(point, 0) - a_out * point for point in points)
                predicate = (
                    f"({lower}) >= {Q_LOW} && ({upper}) <= {Q_HIGH} "
                    f"&& ({lower}) <= ({upper}) "
                    f"&& (__int128)({constant}) == (__int128)({expected})"
                )
            lines.append(
                f'  __ESBMC_assert({predicate}, "{step["step_id"]} {class_name} instance {index}");'
            )
        lines.extend(["  return 0;", "}", ""])
        rendered.append((
            f"{_safe(step['step_id'])}_relu_{chunk_index}.c", "\n".join(lines),
            {"chain_id": chain["certificate_id"], "step_id": step["step_id"],
             "kind": "relu_chunk", "coordinates": len(chunk), "class_counts": counts},
        ))
    expected = int(step["input_constant"]) + sum(constants.values())
    lines = _header() + [
        "int main(void) {",
        f'  __ESBMC_assert((__int128)({step["output_constant"]}) == (__int128)({expected}), "{step["step_id"]} constant sum");',
        "  return 0;", "}", "",
    ]
    rendered.append((
        f"{_safe(step['step_id'])}_relu_close.c", "\n".join(lines),
        {"chain_id": chain["certificate_id"], "step_id": step["step_id"],
         "kind": "relu_close", "coordinates": len(indices)},
    ))
    return rendered


def render_relu_class_lemmas(max_abs_coefficient: int, raw_abs_limit: int) -> list[tuple[str, str, dict[str, Any]]]:
    """Render shared primitive lemmas used by all five ReLU classes.

    Multiplication monotonicity is proved once over the complete coefficient
    and difference envelopes.  The class harnesses then check the concrete
    coefficient signs, facts, endpoint minima, and constants.
    """

    if max_abs_coefficient <= 0 or raw_abs_limit <= 0 or raw_abs_limit > (1 << 31) - 1:
        raise ValueError("Unsupported ReLU lemma envelope")
    difference_limit = 2 * raw_abs_limit + 65536
    if difference_limit > (1 << 31) - 1:
        raise ValueError("ReLU difference envelope does not fit int32")
    prefix = _header() + [
        "int main(void) {",
        "  int32_t raw = (int32_t)nondet_long_long();",
        "  int32_t lower = (int32_t)nondet_long_long();",
        "  int32_t upper = (int32_t)nondet_long_long();",
        f"  __ESBMC_assume(raw >= -{raw_abs_limit} && raw <= {raw_abs_limit});",
        f"  __ESBMC_assume(lower >= -{raw_abs_limit} && lower <= {raw_abs_limit});",
        f"  __ESBMC_assume(upper >= -{raw_abs_limit} && upper <= {raw_abs_limit});",
        "  __int128 z = clamp_to_signed_range_i128((__int128)raw, 16);",
        "  __int128 h = z > 0 ? z : 0;",
    ]
    cases = {
        "bound_free": [
            '  __ESBMC_assert(h >= 0, "ReLU is nonnegative after clamp");',
        ],
        "nosat_upper": [
            "  __ESBMC_assume(raw <= 32767);",
            '  __ESBMC_assert(h - raw >= 0, "without upper saturation ReLU dominates raw");',
        ],
        "lower": [
            "  __ESBMC_assume(raw >= lower);",
            "  __int128 allowance = lower < 0 ? -(__int128)lower : 0;",
            '  __ESBMC_assert(allowance - (h - raw) >= 0, "lower fact bounds ReLU-minus-raw");',
        ],
        "upper": [
            "  __ESBMC_assume(raw <= upper);",
            "  __int128 ceiling = upper > 0 ? upper : 0;",
            '  __ESBMC_assert(ceiling - h >= 0, "upper fact bounds ReLU output");',
        ],
        "lower_upper": [
            f"  __ESBMC_assume(lower >= {Q_LOW} && upper <= {Q_HIGH});",
            "  __ESBMC_assume(lower <= raw && raw <= upper);",
            '  __ESBMC_assert(z == raw, "two-sided class excludes clamp saturation");',
            '  __ESBMC_assert(h == (raw > 0 ? raw : 0), "two-sided class is piecewise linear ReLU");',
        ],
    }
    rendered = []
    for name, body in cases.items():
        source = "\n".join(prefix + body + ["  return 0;", "}", ""])
        rendered.append((
            f"lemma_relu_{name}.c", source,
            {"chain_id": "__shared_lemmas__", "step_id": "", "kind": f"relu_lemma_{name}"},
        ))
    product = _header() + [
        "int main(void) {",
        "  int32_t coefficient = (int32_t)nondet_long_long();",
        "  int32_t difference = (int32_t)nondet_long_long();",
        f"  __ESBMC_assume(coefficient > 0 && coefficient <= {max_abs_coefficient});",
        f"  __ESBMC_assume(difference >= 0 && difference <= {difference_limit});",
        "  int64_t product = (int64_t)coefficient * difference;",
        '  __ESBMC_assert(product >= 0, "positive coefficient preserves nonnegative order gap");',
        "  return 0;", "}", "",
    ]
    rendered.append((
        "lemma_relu_positive_product.c", "\n".join(product),
        {"chain_id": "__shared_lemmas__", "step_id": "", "kind": "relu_lemma_positive_product"},
    ))
    return rendered


def _render_concretize(
    certificate: CertificateV2, chain: dict[str, Any], step: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    coefficients: dict[int, int] = step["_input_coefficients"]
    low = certificate.root["input_box"]["low"]
    high = certificate.root["input_box"]["high"]
    lines = _header() + ["int main(void) {", f"  __int128 value = {step['input_constant']};"]
    for index, coefficient in coefficients.items():
        endpoint = low[index] if coefficient >= 0 else high[index]
        lines.append(f"  value += (__int128)({coefficient}) * ({endpoint});")
    lines.extend([
        f'  __ESBMC_assert(value == (__int128)({step["scaled_bound"]}), "{step["step_id"]} concretize");',
        "  return 0;", "}", "",
    ])
    return (
        f"{_safe(step['step_id'])}_concretize.c", "\n".join(lines),
        {"chain_id": chain["certificate_id"], "step_id": step["step_id"],
         "kind": "concretize", "multiply_terms": len(coefficients)},
    )


def _render_chain_close(
    certificate: CertificateV2, chain: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    steps = chain["_normalized_steps"]
    lines = _header() + ["int main(void) {"]
    for left, right in zip(steps, steps[1:]):
        if left["kind"] in {"affine", "relu"}:
            output = _sparse_payload(_sparse(
                left["output_coefficients"], "output_coefficients",
                certificate.layers[int(left["layer_index"])]["cols" if left["kind"] == "affine" else "rows"],
            ))
            incoming = _sparse_payload(right["_input_coefficients"])
            digest_out = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()
            digest_in = hashlib.sha256(json.dumps(incoming, sort_keys=True).encode()).hexdigest()
            lines.append(
                f'  __ESBMC_assert({1 if digest_out == digest_in else 0}, "{chain["certificate_id"]} coefficient link {left["step_id"]}");'
            )
            lines.append(
                f'  __ESBMC_assert((__int128)({left["output_constant"]}) == (__int128)({right["input_constant"]}), "{chain["certificate_id"]} constant link {left["step_id"]}");'
            )
    scaled = int(chain["scaled_bound"])
    scale = 1 << int(certificate.root["scale_bits"])
    expected = (-scaled) // scale if chain["claim"] == "upper" else -((-scaled) // scale)
    lines.extend([
        f'  __ESBMC_assert((__int128)({chain["bound"]}) == (__int128)({expected}), "{chain["certificate_id"]} rounded bound");',
        "  return 0;", "}", "",
    ])
    return (
        f"{_safe(chain['certificate_id'])}_chain_close.c", "\n".join(lines),
        {"chain_id": chain["certificate_id"], "step_id": "", "kind": "chain_close"},
    )


def render_v2_harnesses(
    certificate: CertificateV2, output: Path | str, *, max_affine_terms: int = 25_000,
    max_relu_coordinates: int = 512,
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    if any(
        step["kind"] == "relu"
        for chain in certificate.chains
        for step in chain["_normalized_steps"]
    ):
        max_coefficient = int(certificate.root["manifest"]["diagnostic_counts"]["max_abs_coefficient"])
        raw_limits = []
        for csr in certificate.layers.values():
            for row in range(csr["rows"]):
                weight_sum = sum(
                    abs(value) for value in csr["data"][csr["indptr"][row]:csr["indptr"][row + 1]]
                )
                raw_limits.append((weight_sum * 32768 + 127) // 256 + abs(csr["bias"][row]) + 1)
        lemma_dir = output / "shared_lemmas"
        lemma_dir.mkdir()
        for filename, source, metadata in render_relu_class_lemmas(max_coefficient, max(raw_limits)):
            path = lemma_dir / filename
            path.write_text(source, encoding="utf-8")
            records.append({
                **metadata, "harness": str(path.resolve()), "harness_sha256": _sha256(path),
            })
    for chain_index, chain in enumerate(certificate.chains):
        rendered: list[tuple[str, str, dict[str, Any]]] = []
        for step in chain["_normalized_steps"]:
            if step["kind"] == "affine":
                rendered.extend(_render_affine(certificate, chain, step, max_affine_terms))
            elif step["kind"] == "relu":
                rendered.extend(_render_relu(certificate, chain, step, max_relu_coordinates))
            else:
                rendered.append(_render_concretize(certificate, chain, step))
        rendered.append(_render_chain_close(certificate, chain))
        chain_dir = output / f"chain_{chain_index:05d}"
        chain_dir.mkdir()
        for filename, source, metadata in rendered:
            path = chain_dir / filename
            path.write_text(source, encoding="utf-8")
            records.append({
                **metadata, "harness": str(path.resolve()),
                "harness_sha256": _sha256(path),
            })
    manifest = {
        "schema": "crown_chain_v2_harness_manifest",
        "selected_layers": list(certificate.selected_layers),
        "complete_certificate_scope": certificate.preflight["complete_certificate_scope"],
        "chain_count": len(certificate.chains),
        "harness_count": len(records),
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
