# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import importlib.metadata

import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except (ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ImportError, OSError):
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'require_flash_attention_2',
]

EXPECTED_FLASH_ATTN_2_VERSION = "2.8.3.post1"


def require_flash_attention_2(
    expected_version: str = EXPECTED_FLASH_ATTN_2_VERSION,
) -> str:
    """Require the external FA2 backend used by the reported ViRDM runs."""
    if not FLASH_ATTN_2_AVAILABLE:
        raise RuntimeError(
            "ViRDM requires flash-attn=="
            f"{expected_version}; install the pinned requirements"
        )
    try:
        installed_version = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "FlashAttention imported but its package metadata is unavailable"
        ) from error
    if installed_version != expected_version:
        raise RuntimeError(
            f"ViRDM requires flash-attn=={expected_version}, "
            f"found {installed_version}"
        )
    return f"flash_attn_2=={installed_version}"


def _scaled_dot_product_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    """PyTorch SDPA fallback with the same public tensor layout as FA2."""
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    out_dtype = q.dtype
    q = q if q.dtype in half_dtypes else q.to(dtype)
    k = k if k.dtype in half_dtypes else k.to(dtype)
    v = v if v.dtype in half_dtypes else v.to(dtype)
    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale

    batch, lq, _, _ = q.shape
    lk = k.shape[1]
    attn_mask = None
    if q_lens is not None or k_lens is not None or window_size != (-1, -1):
        q_lens = (
            torch.full((batch,), lq, dtype=torch.long, device=q.device)
            if q_lens is None
            else torch.as_tensor(q_lens, device=q.device, dtype=torch.long)
        )
        k_lens = (
            torch.full((batch,), lk, dtype=torch.long, device=q.device)
            if k_lens is None
            else torch.as_tensor(k_lens, device=q.device, dtype=torch.long)
        )
        q_index = torch.arange(lq, device=q.device).view(1, lq, 1)
        k_index = torch.arange(lk, device=q.device).view(1, 1, lk)
        attn_mask = (q_index < q_lens[:, None, None]) & (
            k_index < k_lens[:, None, None]
        )
        aligned_q_index = q_index + (k_lens - q_lens)[:, None, None]
        if causal:
            attn_mask &= k_index <= aligned_q_index
        left, right = window_size
        if left >= 0:
            attn_mask &= k_index >= aligned_q_index - left
        if right >= 0:
            attn_mask &= k_index <= aligned_q_index + right
        attn_mask = attn_mask.unsqueeze(1)

    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal and attn_mask is None,
        scale=softmax_scale,
    )
    if q_lens is not None:
        valid_queries = (
            torch.arange(lq, device=q.device)[None, :] < q_lens[:, None]
        )
        out = out * valid_queries[:, None, :, None]
    return out.transpose(1, 2).contiguous().to(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=2,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return _scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
        )

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=2,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        return _scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            dtype=dtype,
        )
