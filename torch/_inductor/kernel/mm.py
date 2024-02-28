import logging
from typing import Any, Dict, List, Optional

import torch
from torch._inductor.virtualized import V
from .. import config as inductor_config
from ..codegen.cuda.gemm_template import CUTLASSGemmTemplate
from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)
from ..utils import (
    use_aten_gemm_kernels,
    use_cutlass_template,
    use_max_autotune,
    use_triton_template,
)
from .mm_common import (
    addmm_epilogue,
    int8_mm_configs,
    mm_args,
    mm_configs,
    mm_grid,
    mm_options,
)

log = logging.getLogger(__name__)
aten = torch.ops.aten

mm_template = TritonTemplate(
    name="mm",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)

aten_mm = ExternKernelChoice(torch.mm, "at::mm_out")


aten_addmm = ExternKernelChoice(torch.addmm, "at::addmm_out")

aten__int_mm = ExternKernelChoice(torch._int_mm, "at::_int_mm")


def _is_int8_mat(mat):
    return mat.get_dtype() in (torch.int8, torch.uint8)


def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    """
    Giving torch.addmm a 1D tensor calls a different (faster) cublasLt
    kernel under the hood.  There are a few shapes where this is slower,
    but they are rare.
    """
    if inp.stride(0) == 0 or inp.size(0) == 1:
        return torch.addmm(inp[0], mat1, mat2, out=out, alpha=alpha, beta=beta)
    return torch.addmm(inp, mat1, mat2, out=out, alpha=alpha, beta=beta)


aten_bias_addmm = ExternKernelChoice(bias_addmm, None)


@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)

    # options to tune from
    choices = [aten_mm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []

    if m * n != 0 and use_triton_template(layout):
        for config in mm_configs(m, n, k):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, k, layout),
            )

    if m * n != 0 and use_cutlass_template(layout):
        CUTLASSGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2], fuseable=True, non_fuseable=True
        )

    from torch._inductor.ir import FixedLayout, FlexibleLayout

    if (
        len(choices) == 1
        and use_aten_gemm_kernels()
        and isinstance(layout, FixedLayout)
    ):
        # If we are not autotuning, we can swap to a FlexibleLayout
        # in order to get fusion optimizations to kick in, e.g. ConcatFusion
        layout = FlexibleLayout(
            device=layout.device, dtype=layout.dtype, size=layout.size
        )
        choices = [aten_mm.bind((mat1, mat2), layout)]

    return autotune_select_algorithm("mm", choices, [mat1, mat2], layout)


@register_lowering(aten._int_mm, type_promotion_kind=None)
def tuned_int_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=torch.int32
    )
    choices = (
        [aten__int_mm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    )
    if m * n != 0 and use_triton_template(layout, enable_int32=True):
        # TODO: Re-enable eager mode implementation once cuBLAS is fixed
        choices = []
        for config in int8_mm_configs(m, n, k):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, k, layout),
            )
    return autotune_select_algorithm("int_mm", choices, [mat1, mat2], layout)


@register_lowering(aten.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    ordered_kwargs_for_cpp_kernel = ("beta", "alpha")

    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    if m * n == 0 or not use_max_autotune():
        choices = (
            [
                aten_addmm.bind(
                    (inp, mat1, mat2),
                    layout,
                    ordered_kwargs_for_cpp_kernel,
                    alpha=alpha,
                    beta=beta,
                )
            ]
            if use_aten_gemm_kernels()
            else []
        )
        return autotune_select_algorithm("addmm", choices, [inp, mat1, mat2], layout)

    choices = (
        [
            aten_addmm.bind(
                (inp_expanded, mat1, mat2),
                layout,
                ordered_kwargs_for_cpp_kernel,
                alpha=alpha,
                beta=beta,
            )
        ]
        if use_aten_gemm_kernels()
        else []
    )

    if (
        use_aten_gemm_kernels()
        and inp_expanded.get_stride()[0] == 0
        and inp_expanded.get_device().type == "cuda"
        and inductor_config.triton.autotune_cublasLt
    ):
        # unexpand inp to make sure fused addmm from cublasLt is used
        choices.insert(
            0,
            aten_bias_addmm.bind(
                (inp_expanded, mat1, mat2), layout, alpha=alpha, beta=beta
            ),
        )

    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(inp_expanded, mat1, mat2),
                layout=layout,
                **mm_options(config, k, layout),
                prefix_args=1,
                epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta),
            )

    if use_cutlass_template(layout):
        CUTLASSGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            [mat1, mat2, inp_expanded],
            alpha=alpha,
            beta=beta,
            input_reorder=[2, 0, 1],
            fuseable=False,
        )

    return autotune_select_algorithm(
        "addmm", choices, [inp_expanded, mat1, mat2], layout
    )


def fallback_mixed_mm(mat1, mat2, *, out):
    return torch.mm(mat1, mat2.to(mat1.dtype), out=out)


aten_fallback_mixed_mm = ExternKernelChoice(fallback_mixed_mm, None)


@functools.lru_cache(None)
def _is_sm7x_or_older_gpu(index: Optional[int]) -> bool:
    props = torch.cuda.get_device_properties(index or 0)
    return props.major <= 7


def tuned_mixed_mm(mat1, mat2, mat2_dtype):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=None)
    choices = [aten_fallback_mixed_mm.bind((mat1, mat2), layout)]
    if (
        mat1.layout.dtype != torch.float32 and not mat2.layout.is_contiguous()
    ) or _is_sm7x_or_older_gpu(layout.device.index):
        # can't use triton kernel unless one of these is true
        return autotune_select_algorithm("mixed_mm", choices, [mat1, mat2], layout)
    if inductor_config.force_mixed_mm:
        choices = []
    b_prologue_cast_type = f"tl.{mat2_dtype}".replace("torch.", "")
    has_int8_tensor = _is_int8_mat(mat1) or _is_int8_mat(mat2)
    for config in mm_configs(m, n, k, has_int8_tensor=has_int8_tensor):
        mm_template.maybe_append_choice(
            choices,
            input_nodes=(mat1, mat2),
            layout=layout,
            **mm_options(config, k, layout, b_prologue_cast_type),
        )
    return autotune_select_algorithm("mixed_mm", choices, [mat1, mat2], layout)


# This op is a special case of the int_mm op which we use based on the pattern
# _int_mm -> mul (defined in ../fx_passes/post_grad.py) in order to prevent
# realization of the int32 _int_mm output by forcing fusion with the mul op.
# This is only used when config.force_fuse_int_mm_with_mul = True
def tuned_fused_int_mm_mul(mat1, mat2, mat3, out_dtype, *, layout=None):
    out_dtype = (
        torch.promote_types(mat3.get_dtype(), torch.int32)
        if out_dtype is None
        else out_dtype
    )
    m, n, k, layout, mat1, mat2, mat3 = mm_args(
        mat1, mat2, mat3, layout=layout, out_dtype=out_dtype
    )
    choices: List[Dict[Any, Any]] = []
    for config in int8_mm_configs(m, n, k):
        mm_template.maybe_append_choice(
            choices,
            input_nodes=(mat1, mat2, mat3),
            layout=layout,
            **dict(mm_options(config, k, layout), ACC_TYPE="tl.int32"),
            suffix_args=1,
            epilogue_fn=V.ops.mul,
        )
    return autotune_select_algorithm("int_mm", choices, [mat1, mat2, mat3], layout)
