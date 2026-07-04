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


# ---- PFF ↔ StatsBomb integration invariants ----------------------------
# These two tests pin down bugs that once shipped: a duplicated (and wrong)
# SB→PFF mapping in app.py, and a missing y-flip in the coordinate transform.

def test_pff_to_statsbomb_transform():
    from pff_loader import pff_to_statsbomb

    # centre spot maps to centre spot
    assert np.allclose(pff_to_statsbomb(0.0, 0.0), (60.0, 40.0))
    # PFF +y and StatsBomb +y are OPPOSITE: pff (0, +34) is sb y=0,
    # pff (0, -34) is sb y=80 (verified against 752 paired passes of
    # the Final: ~5 yd residual with the flip, 35-70 yd without)
    assert np.allclose(pff_to_statsbomb(-52.5, 34.0), (0.0, 0.0))
    assert np.allclose(pff_to_statsbomb(52.5, -34.0), (120.0, 80.0))
    # x is monotone increasing, y monotone decreasing in pff inputs
    assert pff_to_statsbomb(10.0, 0.0)[0] > pff_to_statsbomb(-10.0, 0.0)[0]
    assert pff_to_statsbomb(0.0, 10.0)[1] < pff_to_statsbomb(0.0, -10.0)[1]


def test_sb_to_pff_mapping_matches_fixtures():
    from pff_loader import SB_TO_PFF, PFF_KEY_MATCHES, DATA_DIR
    import json

    # ground truth derived from PFF Metadata/<id>.json home/away team names
    # and StatsBomb event team names (do not edit without re-verifying both)
    expected = {
        3869685: 10517,  # Argentina vs France
        3869684: 10516,  # Croatia vs Morocco
        3869552: 10515,  # France vs Morocco
        3869519: 10514,  # Argentina vs Croatia
        3869354: 10513,  # England vs France
        3869486: 10512,  # Morocco vs Portugal
        3869321: 10511,  # Netherlands vs Argentina
        3869420: 10510,  # Croatia vs Brazil
    }
    assert SB_TO_PFF == expected
    assert set(SB_TO_PFF.values()) == set(PFF_KEY_MATCHES.keys())

    # when the raw metadata is on disk, re-verify labels against it
    meta_dir = DATA_DIR / "Metadata"
    if meta_dir.exists():
        for pff_id, label in PFF_KEY_MATCHES.items():
            meta_path = meta_dir / f"{pff_id}.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if isinstance(meta, list):
                meta = meta[0]
            assert meta["homeTeam"]["name"] in label, (pff_id, label)
            assert meta["awayTeam"]["name"] in label, (pff_id, label)


# ---- validation methodology helpers ------------------------------------

def test_infer_intended_target_picks_closest_teammate():
    from validate_model import infer_intended_target

    end = np.array([20.0, 20.0])
    tms = np.array([[50.0, 50.0], [22.0, 19.0]])
    target, is_proxy = infer_intended_target(end, tms)
    assert is_proxy
    assert np.allclose(target, [22.0, 19.0])
    # no teammates: keep the recorded end location, unproxied
    target, is_proxy = infer_intended_target(end, np.zeros((0, 2)))
    assert not is_proxy
    assert np.allclose(target, end)


def test_disagreement_cohorts_are_symmetric_and_exclusive():
    import pandas as pd
    from validate_model import disagreement_cohorts

    rows = [
        # targets agree (sep 2 yd) -> excluded even though actual hits both
        dict(opt_x=50, opt_y=40, nearest_x=52, nearest_y=40,
             actual_end_x=51, actual_end_y=40, tag="agree"),
        # disagree, actual at the model target -> went_model
        dict(opt_x=100, opt_y=40, nearest_x=60, nearest_y=40,
             actual_end_x=99, actual_end_y=41, tag="model"),
        # disagree, actual at the nearest teammate -> went_baseline
        dict(opt_x=100, opt_y=40, nearest_x=60, nearest_y=40,
             actual_end_x=61, actual_end_y=39, tag="baseline"),
        # disagree, actual near neither -> excluded
        dict(opt_x=100, opt_y=40, nearest_x=60, nearest_y=40,
             actual_end_x=80, actual_end_y=10, tag="neither"),
    ]
    df = pd.DataFrame(rows)
    went_model, went_baseline = disagreement_cohorts(df)
    assert went_model["tag"].tolist() == ["model"]
    assert went_baseline["tag"].tolist() == ["baseline"]
