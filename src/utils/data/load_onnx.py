from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import os
import numpy as np
import onnx
from onnx import numpy_helper, external_data_helper

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

import tensorflow as tf


def infer_onnx_input_dim(onnx_path: str) -> int:
    """
    Retorna a dimensionalidade da entrada (sem batch) do primeiro input do grafo.
    Para Iris, deve retornar 4.
    """
    model = onnx.load(onnx_path)
    ten = model.graph.input[0].type.tensor_type
    dims = [d.dim_value for d in ten.shape.dim]
    # Remove batch (0 ou None) e multiplica o resto
    if len(dims) == 0:
        raise ValueError("ONNX input shape not found.")
    # se formato [N, D] => usa D; se for algo mais alto, faz o produto
    feat_dims = [d for i, d in enumerate(dims) if i != 0 and d != 0]
    if not feat_dims:
        # pode ser que a informação de dim_value não esteja setada; caia no plano B:
        # tente usar feature size pelo primeiro initializer compatível, se necessário.
        raise ValueError(f"Cannot infer input dimension from ONNX ({dims}).")
    prod = 1
    for d in feat_dims:
        prod *= d
    return prod


def extract_gemm_params(onnx_path: str) -> List[Tuple[np.ndarray, Optional[np.ndarray], int]]:
    """
    Extrai, em ordem, os pesos e bias das camadas totalmente conectadas
    representadas como nós 'Gemm' no grafo ONNX.

    Retorna lista de tuplas: (B, C, transB)
      - B: matriz de pesos como armazenada no ONNX (atenção ao transB)
      - C: vetor de bias (ou None, se ausente)
      - transB: atributo do nó Gemm (0 ou 1)
    """
    model = onnx.load(onnx_path)
    g = model.graph

    # índice rápido para initializers por nome
    init_map = {t.name: numpy_helper.to_array(t) for t in g.initializer}

    gemms = []
    for node in g.node:
        if node.op_type != "Gemm":
            continue

        # inputs: [A, B, (optional) C]
        B_name = node.input[1] if len(node.input) >= 2 else None
        C_name = node.input[2] if len(node.input) >= 3 else None

        if B_name is None or B_name not in init_map:
            raise ValueError(f"Gemm node '{node.name}' without B initializer.")

        B = np.array(init_map[B_name], copy=True)
        C = None
        if C_name and C_name in init_map:
            C = np.array(init_map[C_name], copy=True).reshape(-1)

        # pega transB (padrão 1 em exports de Keras, mas vamos ler de fato)
        transB = 0
        for attr in node.attribute:
            if attr.name == "transB":
                transB = int(attr.i)
                break

        gemms.append((B, C, transB))

    if not gemms:
        raise ValueError("No Gemm nodes found in the ONNX model. "
                         "If your export uses MatMul+Add, we need a different extractor.")
    return gemms


def load_onnx_weights_into_keras(model: tf.keras.Model, onnx_path: str) -> None:
    """
    Mapeia, na ordem, os nós Gemm do ONNX para as camadas Dense do modelo Keras.
    Ajusta transposição conforme 'transB' e injeta [kernel, bias].
    """
    gemms = extract_gemm_params(onnx_path)

    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    if len(dense_layers) != len(gemms):
        raise ValueError(f"Mismatch: found {len(gemms)} Gemm nodes in ONNX, "
                         f"but Keras model has {len(dense_layers)} Dense layers.")

    for idx, (layer, (B, C, transB)) in enumerate(zip(dense_layers, gemms)):
        # Keras Dense espera (in_dim, out_dim) em layer.get_weights()[0].shape
        # ONNX Gemm usa Y = A @ (B^T if transB=1 else B) + C
        W = B.T if transB == 1 else B  # converte para (in, out)

        # sanity check com shapes do Keras
        in_dim = layer.input_shape[-1] if hasattr(layer, "input_shape") and layer.input_shape[-1] else W.shape[0]
        out_dim = layer.units
        if W.shape != (in_dim, out_dim):
            # Tenta resortir ao shape da camada, só para diagnosticar melhor:
            raise ValueError(f"[Layer {idx} '{layer.name}'] weight shape mismatch. "
                             f"Expected {(in_dim, out_dim)}, got {W.shape} (transB={transB}).")

        if C is None:
            C = np.zeros((out_dim,), dtype=W.dtype)

        layer.set_weights([W.astype(np.float32), C.astype(np.float32)])


def _to_array(tensor_proto) -> np.ndarray:
    """Convert TensorProto (or SparseTensorProto) to dense numpy array."""
    # onnx.numpy_helper.to_array supports TensorProto and (in recent onnx) SparseTensorProto
    arr = numpy_helper.to_array(tensor_proto)
    # Defensive copy so the caller is safe to mutate
    return np.array(arr, copy=True)


def _maybe_to_torch(arr: np.ndarray, as_torch: bool):
    if as_torch:
        if not _HAS_TORCH:
            raise RuntimeError("as_torch=True but PyTorch is not available.")
        return torch.from_numpy(arr.copy())
    return arr


def load_onnx_model(model_path: os.PathLike | str) -> onnx.ModelProto:
    """
    Loads an ONNX model, ensuring external data tensors are brought in.
    """
    model_path = str(model_path)
    try:
        model = onnx.load(model_path, load_external_data=True)
    except TypeError:
        # Older onnx versions
        model = onnx.load(model_path)
        external_data_helper.load_external_data_for_model(model, os.path.dirname(model_path))
    return model


def load_all_parameters(
    model_path: os.PathLike | str,
    as_torch: bool = False
) -> Dict[str, Any]:
    """
    Returns a dict {parameter_name -> array/tensor} with:
      - graph.initializer tensors
      - sparse_initializers (densified)
      - Constant node tensors (keyed by the node's output[0] name)
    """
    model = load_onnx_model(model_path)
    g = model.graph
    params: Dict[str, Any] = {}

    # Dense initializers
    for t in g.initializer:
        arr = _to_array(t)
        params[t.name] = _maybe_to_torch(arr, as_torch)

    # Sparse initializers (densify)
    for st in getattr(g, "sparse_initializer", []):
        arr = _to_array(st)  # to_array handles SparseTensorProto
        params[st.name] = _maybe_to_torch(arr, as_torch)

    # Constant nodes (grab the tensor-valued 'value' attribute)
    for node in g.node:
        if node.op_type == "Constant":
            # Prefer the tensor 'value' attribute, ignore scalar attrs (value_int, value_float, ...)
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type != 0:  # data_type==0 means not set
                    out_name = node.output[0] if node.output else f"Constant_{id(node)}"
                    arr = _to_array(attr.t)
                    params[out_name] = _maybe_to_torch(arr, as_torch)
                    break

    return pa
