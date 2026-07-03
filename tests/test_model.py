"""Unit tests for zone mapping, value iteration, post-processing and the
alpha invariants. No external data required."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_xt import pos_to_zone, train_xt, postprocess_xt, N_COLS, N_ROWS
from pitch_control import compute_continuous_alpha


def test_pos_to_zone_corners_and_clipping():
    assert pos_to_zone(0.0, 0.0) == 0
    assert pos_to_zone(119.9, 79.9) == N_COLS * N_ROWS - 1
    # out-of-range coordinates clip instead of overflowing
    assert pos_to_zone(-5.0, -5.0) == 0
    assert pos_to_zone(500.0, 500.0) == N_COLS * N_ROWS - 1
    # one cell right and one row up
    assert pos_to_zone(10.1, 10.1) == 1 * N_COLS + 1


def test_value_iteration_matches_analytic_solution():
    # two-state chain: state 0 moves to state 1 half the time and never shoots;
    # state 1 shoots half the time with 20% conversion and never moves.
    T = np.array([[0.0, 0.5], [0.0, 0.0]])
    s = np.array([0.0, 0.5])
    g = np.array([0.0, 0.2])
    xT, convergence = train_xt(T, s, g, max_iter=500, tol=1e-12)
    assert np.isclose(xT[1], 0.1, atol=1e-10)   # 0.5 * 0.2
    assert np.isclose(xT[0], 0.05, atol=1e-10)  # 0.5 * xT[1]
    assert convergence[-1] < 1e-12


def test_postprocess_shapes_and_nonnegativity():
    rng = np.random.RandomState(0)
    grid = np.abs(rng.randn(N_ROWS, N_COLS)) * 0.01
    smoothed, upscaled = postprocess_xt(grid)
    assert smoothed.shape == (N_ROWS, N_COLS)
    assert upscaled.shape == (80, 120)
    assert upscaled.min() >= 0.0


def _toy_situation():
    bc = np.array([60.0, 40.0])
    teammates = np.array([[70.0, 35.0], [55.0, 20.0], [75.0, 60.0]])
    defenders = np.array([[80.0, 40.0], [72.0, 50.0], [68.0, 30.0], [90.0, 40.0], [115.0, 40.0]])
    actual_end = np.array([70.0, 35.0])
    return bc, teammates, defenders, actual_end


def test_alpha_is_never_positive():
    # the optimum is the argmax of the same surface the actual pass is read
    # from, so alpha = actual - optimal can never exceed zero
    bc, tm, df, end = _toy_situation()
    res = compute_continuous_alpha(bc, tm, df, end)
    assert res["spatial_xev"].shape == (80, 120)
    assert res["opt_xev"] >= res["actual_xev"]
    assert res["alpha"] <= 1e-9
    assert 0 <= res["opt_x"] <= 120 and 0 <= res["opt_y"] <= 80


def test_offside_zone_is_penalized_and_not_preferred():
    bc = np.array([55.0, 40.0])
    # one onside option near the carrier, one teammate parked deep offside
    teammates = np.array([[58.0, 35.0], [100.0, 40.0]])
    defenders = np.array([[30.0, 30.0], [35.0, 50.0], [40.0, 40.0], [45.0, 35.0]])
    end = np.array([58.0, 35.0])
    res = compute_continuous_alpha(bc, teammates, defenders, end, attack_right=True)
    # with every defender in their own half the offside line clamps to halfway;
    # beyond it, cells carry negative EPV (turnover cost), never positive value
    assert res["spatial_xev"][:, 62:].max() < 0.0
    # with a legal option available, the optimum never lands offside
    assert res["opt_x"] <= 61.0
    assert res["opt_xev"] > 0.0
