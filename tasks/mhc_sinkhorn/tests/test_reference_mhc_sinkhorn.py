"""CPU reference and frozen-contract tests for MhcSinkhorn."""

from __future__ import annotations

import numpy as np
import pytest

from reference_mhc_sinkhorn import mhc_sinkhorn_reference, validate_contract


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("shape_prefix", [(), (3,), (2, 3)])
@pytest.mark.parametrize("mask_mode", [0, 1, 2])
def test_reference_contract_matrix(dtype, n, shape_prefix, mask_mode):
    rng = np.random.default_rng(1000 + n + mask_mode)
    shape = (*shape_prefix, n, n)
    logits = rng.normal(0.0, 2.0, size=shape).astype(dtype)
    mask = None
    if mask_mode == 1:
        mask = np.asarray([0.125], dtype=dtype)
    elif mask_mode == 2:
        mask = rng.normal(0.0, 0.5, size=shape).astype(dtype)

    actual = mhc_sinkhorn_reference(logits, mask)
    assert actual.shape == logits.shape
    assert actual.dtype == logits.dtype
    assert np.all(np.isfinite(actual))
    column_sums = actual.astype(np.float32).sum(axis=-2)
    tolerance = 3.0e-3 if dtype is np.float16 else 2.0e-5
    assert np.allclose(column_sums, 1.0, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("iterations", [1, 2, 7, 20, 100])
@pytest.mark.parametrize("eps", [1.0e-8, 1.0e-6, 1.0e-3])
def test_nondefault_iterations_and_eps(iterations, eps):
    logits = np.linspace(-8.0, 8.0, 64, dtype=np.float32).reshape(8, 8)
    output = mhc_sinkhorn_reference(logits, iterations=iterations, eps=eps)
    assert output.shape == logits.shape
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_extreme_finite_logits_are_stable(dtype):
    logits = np.asarray(
        [[100.0, -100.0, 50.0, -50.0]] * 4,
        dtype=dtype,
    )
    output = mhc_sinkhorn_reference(logits, iterations=3)
    assert np.all(np.isfinite(output))
    assert np.all(output >= 0)


@pytest.mark.parametrize(
    "logits,mask,iterations,eps,error",
    [
        (np.zeros((4,), np.float32), None, 20, 1.0e-6, ValueError),
        (np.zeros((3, 3), np.float32), None, 20, 1.0e-6, ValueError),
        (np.zeros((4, 6), np.float32), None, 20, 1.0e-6, ValueError),
        (np.zeros((4, 4), np.float64), None, 20, 1.0e-6, TypeError),
        (
            np.zeros((4, 4), np.float32),
            np.zeros((4, 4), np.float16),
            20,
            1.0e-6,
            TypeError,
        ),
        (
            np.zeros((4, 4), np.float32),
            np.zeros((2,), np.float32),
            20,
            1.0e-6,
            ValueError,
        ),
        (np.zeros((4, 4), np.float32), None, 0, 1.0e-6, ValueError),
        (np.zeros((4, 4), np.float32), None, 101, 1.0e-6, ValueError),
    ],
)
def test_invalid_contract_rejected(logits, mask, iterations, eps, error):
    with pytest.raises(error):
        validate_contract(logits, mask, iterations=iterations, eps=eps)


def test_scalar_and_constant_full_mask_are_equivalent():
    logits = np.arange(72, dtype=np.float32).reshape(2, 6, 6) / 9.0
    scalar = np.asarray([0.25], dtype=np.float32)
    full = np.full_like(logits, 0.25)
    scalar_output = mhc_sinkhorn_reference(logits, scalar, iterations=4)
    full_output = mhc_sinkhorn_reference(logits, full, iterations=4)
    assert np.array_equal(scalar_output, full_output)


def test_empty_leading_domain_is_preserved():
    logits = np.empty((2, 0, 6, 6), dtype=np.float32)
    output = mhc_sinkhorn_reference(logits)
    assert output.shape == logits.shape
    assert output.size == 0


def test_eps_is_not_artificially_range_restricted():
    logits = np.zeros((4, 4), dtype=np.float32)
    output = mhc_sinkhorn_reference(logits, iterations=1, eps=0.0)
    assert np.all(np.isfinite(output))
