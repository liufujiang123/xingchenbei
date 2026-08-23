"""Independent FP32 reference and executable-contract validation for MhcSinkhorn."""

from __future__ import annotations

from typing import Optional

import numpy as np


SUPPORTED_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))
SUPPORTED_N = (4, 6, 8)


def validate_contract(
    logits: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    iterations: int = 20,
    eps: float = 1.0e-6,
) -> int:
    """Validate the frozen executable domain and return MASK_MODE."""
    if not isinstance(logits, np.ndarray):
        raise TypeError("logits must be a numpy array")
    if logits.dtype not in SUPPORTED_DTYPES:
        raise TypeError("logits dtype must be float16 or float32")
    if logits.ndim < 2:
        raise ValueError("logits rank must be at least 2")
    if any(dimension < 0 for dimension in logits.shape):
        raise ValueError("logits dimensions must be nonnegative")
    if logits.shape[-2] != logits.shape[-1] or logits.shape[-1] not in SUPPORTED_N:
        raise ValueError("logits must have trailing shape N x N for N in {4,6,8}")
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(
        iterations, (int, np.integer)
    ):
        raise TypeError("iterations must be an integer")
    if not 1 <= int(iterations) <= 100:
        raise ValueError("iterations must be in [1,100]")
    if mask is None:
        return 0
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy array when present")
    if mask.dtype != logits.dtype:
        raise TypeError("mask dtype must equal logits dtype")
    if mask.size == 1:
        return 1
    if mask.size == logits.size:
        return 2
    raise ValueError("mask element count must be one or equal logits element count")


def _sum_in_order(values: np.ndarray) -> np.float32:
    total = np.float32(0.0)
    for value in values:
        total = np.float32(total + np.float32(value))
    return total


def mhc_sinkhorn_reference(
    logits: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    iterations: int = 20,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Run the contract sequence with explicit FP32 row/column reductions."""
    mask_mode = validate_contract(logits, mask, iterations=iterations, eps=eps)
    output_dtype = logits.dtype
    n = logits.shape[-1]
    matrix_count = logits.size // (n * n)
    state = np.ascontiguousarray(logits, dtype=np.float32).reshape(matrix_count, n, n)
    state = state.copy()
    if mask_mode == 1:
        state += np.float32(np.ascontiguousarray(mask).reshape(-1)[0])
    elif mask_mode == 2:
        state += np.ascontiguousarray(mask, dtype=np.float32).reshape(matrix_count, n, n)

    eps32 = np.float32(eps)
    for matrix in range(matrix_count):
        for row in range(n):
            maximum = np.float32(state[matrix, row, 0])
            for column in range(1, n):
                value = np.float32(state[matrix, row, column])
                maximum = value if value > maximum else maximum
            for column in range(n):
                state[matrix, row, column] = np.float32(
                    state[matrix, row, column] - maximum
                )

    np.exp(state, out=state)

    for matrix in range(matrix_count):
        for row in range(n):
            row_sum = _sum_in_order(state[matrix, row, :])
            reciprocal = np.float32(np.float32(1.0) / row_sum)
            for column in range(n):
                state[matrix, row, column] = np.float32(
                    np.float32(state[matrix, row, column] * reciprocal) + eps32
                )

        for column in range(n):
            column_sum = np.float32(_sum_in_order(state[matrix, :, column]) + eps32)
            reciprocal = np.float32(np.float32(1.0) / column_sum)
            for row in range(n):
                state[matrix, row, column] = np.float32(
                    state[matrix, row, column] * reciprocal
                )

        for _ in range(1, int(iterations)):
            for row in range(n):
                row_sum = np.float32(_sum_in_order(state[matrix, row, :]) + eps32)
                reciprocal = np.float32(np.float32(1.0) / row_sum)
                for column in range(n):
                    state[matrix, row, column] = np.float32(
                        state[matrix, row, column] * reciprocal
                    )
            for column in range(n):
                column_sum = np.float32(
                    _sum_in_order(state[matrix, :, column]) + eps32
                )
                reciprocal = np.float32(np.float32(1.0) / column_sum)
                for row in range(n):
                    state[matrix, row, column] = np.float32(
                        state[matrix, row, column] * reciprocal
                    )

    return state.reshape(logits.shape).astype(output_dtype)
