"""Compatibility wrapper for legacy ESBMC C-template helpers."""

from verification.c_templates import (
    render_hidden_affine_bounds_program as _render_hidden_affine_bounds_program,
    render_output_target_program as _render_output_target_program,
    render_output_valid_set_program as _render_output_valid_set_program,
)


def innerlayer_fixed_int(
    cur_layer_layer_size,
    in_layer_layer_size,
    weights_c_int,
    biases_c_int,
    preimage_low_int,
    preimage_high_int,
    input_bounds_low_int,
    input_bounds_high_int,
    scale_factor,
    total_bits=64,
):
    return _render_hidden_affine_bounds_program(
        output_size=cur_layer_layer_size,
        input_size=in_layer_layer_size,
        weights_c_int=weights_c_int,
        biases_c_int=biases_c_int,
        preimage_low_c_int=preimage_low_int,
        preimage_high_c_int=preimage_high_int,
        input_bounds_low_c_int=input_bounds_low_int,
        input_bounds_high_c_int=input_bounds_high_int,
        scale_factor=scale_factor,
        total_bits=total_bits,
    )


def innerlayer_fixed_int_bounds_only(
    cur_layer_layer_size,
    in_layer_layer_size,
    weights_c_int,
    biases_c_int,
    preimage_low_int,
    preimage_high_int,
    input_bounds_low_int,
    input_bounds_high_int,
    scale_factor,
    total_bits=64,
):
    return _render_hidden_affine_bounds_program(
        output_size=cur_layer_layer_size,
        input_size=in_layer_layer_size,
        weights_c_int=weights_c_int,
        biases_c_int=biases_c_int,
        preimage_low_c_int=preimage_low_int,
        preimage_high_c_int=preimage_high_int,
        input_bounds_low_c_int=input_bounds_low_int,
        input_bounds_high_c_int=input_bounds_high_int,
        scale_factor=scale_factor,
        total_bits=total_bits,
    )


def outerlayer_fixed_int(
    in_layer_layer_size,
    cur_layer_layer_size,
    weights_c_int,
    biases_c_int,
    input_bounds_low_int,
    input_bounds_high_int,
    targetCls,
    scale_factor,
    total_bits=64,
):
    return _render_output_target_program(
        output_size=cur_layer_layer_size,
        input_size=in_layer_layer_size,
        weights_c_int=weights_c_int,
        biases_c_int=biases_c_int,
        input_bounds_low_c_int=input_bounds_low_int,
        input_bounds_high_c_int=input_bounds_high_int,
        target_label=targetCls,
        scale_factor=scale_factor,
        total_bits=total_bits,
    )


def outerlayer_fixed_int_multiclass(
    in_layer_layer_size,
    cur_layer_layer_size,
    weights_c_int,
    biases_c_int,
    input_bounds_low_int,
    input_bounds_high_int,
    valid_classes,
    scale_factor,
    total_bits=64,
):
    return _render_output_valid_set_program(
        output_size=cur_layer_layer_size,
        input_size=in_layer_layer_size,
        weights_c_int=weights_c_int,
        biases_c_int=biases_c_int,
        input_bounds_low_c_int=input_bounds_low_int,
        input_bounds_high_c_int=input_bounds_high_int,
        valid_classes=tuple(valid_classes),
        scale_factor=scale_factor,
        total_bits=total_bits,
    )


__all__ = [
    "innerlayer_fixed_int",
    "innerlayer_fixed_int_bounds_only",
    "outerlayer_fixed_int",
    "outerlayer_fixed_int_multiclass",
]
