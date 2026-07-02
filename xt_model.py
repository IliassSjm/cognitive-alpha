"""
Expected Threat (xT) Model — Trained from World Cup 2022
=========================================================
Markov chain xT trained on 64 World Cup 2022 matches via
value iteration, with Y-axis symmetry mirroring and Gaussian
smoothing. See train_xt.py for the full training pipeline.

Grid (12×8):
  - Row 0 = bottom of pitch (y = 0), Row 7 = top (y = 80)
  - Col 0 = own goal line (x = 0), Col 11 = opponent goal line (x = 120)
  - StatsBomb coordinates: x ∈ [0, 120], y ∈ [0, 80]

Training stats:
  - 234,652 events from 64 matches (doubled via Y-mirror)
  - 222,094 successful moves, 2,988 shots, 390 goals
  - 26,050 turnovers (absorbing states, 12.9% avg rate)
  - Converged after 100 iterations (Δ < 1e-8)
  - Pearson r = 0.93 vs Karun Singh reference

Also provides a continuous 120×80 interpolated surface for
smooth integration with the pitch control meshgrid.
"""

import numpy as np
from pathlib import Path

# ---- Trained xT Grid (12×8) — Smoothed (Gaussian σ=0.8) ----
# Trained via Markov chain value iteration on World Cup 2022 data.
# Y-axis mirrored for symmetry, Gaussian-smoothed to remove spikes.
XT_GRID = np.array([
    #  Col:   0       1       2       3       4       5       6       7       8       9      10      11
    [0.0039, 0.0047, 0.0059, 0.0072, 0.0086, 0.0103, 0.0125, 0.0153, 0.0190, 0.0232, 0.0276, 0.0316],  # Row 0 (y=0..10)
    [0.0045, 0.0053, 0.0065, 0.0079, 0.0094, 0.0112, 0.0134, 0.0165, 0.0209, 0.0274, 0.0348, 0.0390],  # Row 1
    [0.0052, 0.0060, 0.0072, 0.0086, 0.0101, 0.0120, 0.0143, 0.0178, 0.0239, 0.0383, 0.0623, 0.0776],  # Row 2
    [0.0057, 0.0064, 0.0076, 0.0089, 0.0104, 0.0123, 0.0149, 0.0189, 0.0273, 0.0555, 0.1110, 0.1495],  # Row 3 (centre)
    [0.0057, 0.0064, 0.0076, 0.0089, 0.0105, 0.0124, 0.0149, 0.0191, 0.0280, 0.0594, 0.1188, 0.1540],  # Row 4 (centre)
    [0.0052, 0.0060, 0.0073, 0.0087, 0.0102, 0.0121, 0.0145, 0.0181, 0.0250, 0.0423, 0.0697, 0.0825],  # Row 5
    [0.0045, 0.0054, 0.0066, 0.0080, 0.0095, 0.0114, 0.0137, 0.0169, 0.0218, 0.0292, 0.0372, 0.0409],  # Row 6
    [0.0040, 0.0048, 0.0060, 0.0073, 0.0088, 0.0105, 0.0128, 0.0158, 0.0198, 0.0244, 0.0287, 0.0322],  # Row 7 (y=70..80)
])

# Karun Singh reference (Premier League) — kept for comparison
XT_GRID_KARUN = np.array([
    [0.0005, 0.0005, 0.0005, 0.0006, 0.0009, 0.0015, 0.0024, 0.0042, 0.0066, 0.0121, 0.0183, 0.0270],
    [0.0006, 0.0006, 0.0007, 0.0009, 0.0014, 0.0024, 0.0040, 0.0070, 0.0118, 0.0225, 0.0467, 0.0870],
    [0.0006, 0.0007, 0.0008, 0.0010, 0.0016, 0.0029, 0.0051, 0.0095, 0.0170, 0.0371, 0.0869, 0.2000],
    [0.0007, 0.0007, 0.0008, 0.0011, 0.0017, 0.0031, 0.0056, 0.0105, 0.0193, 0.0429, 0.1072, 0.2900],
    [0.0007, 0.0007, 0.0008, 0.0011, 0.0017, 0.0031, 0.0056, 0.0105, 0.0193, 0.0429, 0.1072, 0.2900],
    [0.0006, 0.0007, 0.0008, 0.0010, 0.0016, 0.0029, 0.0051, 0.0095, 0.0170, 0.0371, 0.0869, 0.2000],
    [0.0006, 0.0006, 0.0007, 0.0009, 0.0014, 0.0024, 0.0040, 0.0070, 0.0118, 0.0225, 0.0467, 0.0870],
    [0.0005, 0.0005, 0.0005, 0.0006, 0.0009, 0.0015, 0.0024, 0.0042, 0.0066, 0.0121, 0.0183, 0.0270],
])

# Grid dimensions
N_ROWS = XT_GRID.shape[0]   # 8
N_COLS = XT_GRID.shape[1]   # 12

# Pitch dimensions (StatsBomb)
PITCH_LENGTH = 120.0
PITCH_WIDTH  = 80.0

# Bin sizes
BIN_X = PITCH_LENGTH / N_COLS   # 10.0 yards per column
BIN_Y = PITCH_WIDTH  / N_ROWS   # 10.0 yards per row


# ---- Continuous 120×80 surface (bilinear interpolation from train_xt) ----
_XT_UPSCALED = None   # lazy-loaded

def _load_upscaled() -> np.ndarray:
    """Load the precomputed 120×80 interpolated xT surface."""
    global _XT_UPSCALED
    if _XT_UPSCALED is not None:
        return _XT_UPSCALED

    upscaled_path = Path(__file__).parent / "xt_upscaled_120x80.npy"
    if upscaled_path.exists():
        _XT_UPSCALED = np.load(upscaled_path)
    else:
        # Fallback: zoom the 12×8 grid
        from scipy.ndimage import zoom
        _XT_UPSCALED = zoom(XT_GRID, (PITCH_WIDTH / N_ROWS, PITCH_LENGTH / N_COLS), order=1)
    return _XT_UPSCALED


def lookup_xt(x: float, y: float) -> float:
    """
    Return the xT value for a given (x, y) position on the pitch.

    Uses the continuous 120×80 surface if available, otherwise
    falls back to the 12×8 grid.
    """
    upscaled = _load_upscaled()
    if upscaled is not None and upscaled.shape == (80, 120):
        xi = int(np.clip(x, 0, PITCH_LENGTH - 1))
        yi = int(np.clip(y, 0, PITCH_WIDTH - 1))
        return float(upscaled[yi, xi])

    col = int(np.clip(x // BIN_X, 0, N_COLS - 1))
    row = int(np.clip(y // BIN_Y, 0, N_ROWS - 1))
    return float(XT_GRID[row, col])


def lookup_xt_array(positions: np.ndarray) -> np.ndarray:
    """
    Vectorized xT lookup for an array of positions (N, 2).

    Uses the continuous 120×80 surface for smooth values.
    """
    upscaled = _load_upscaled()
    if upscaled is not None and upscaled.shape == (80, 120):
        xi = np.clip(positions[:, 0].astype(int), 0, 119)
        yi = np.clip(positions[:, 1].astype(int), 0, 79)
        return upscaled[yi, xi]

    cols = np.clip((positions[:, 0] // BIN_X).astype(int), 0, N_COLS - 1)
    rows = np.clip((positions[:, 1] // BIN_Y).astype(int), 0, N_ROWS - 1)
    return XT_GRID[rows, cols]


# ---- Sanity check ----
if __name__ == "__main__":
    test_positions = {
        "Own penalty area (15, 40)":   (15.0, 40.0),
        "Centre circle (60, 40)":      (60.0, 40.0),
        "Edge of box (102, 40)":       (102.0, 40.0),
        "Penalty spot (108, 40)":      (108.0, 40.0),
        "Six-yard box (115, 40)":      (115.0, 40.0),
        "Wide touchline (90, 5)":      (90.0, 5.0),
        "Half-space (95, 25)":         (95.0, 25.0),
    }
    print("xT Grid Sanity Check (Trained on World Cup 2022)")
    for label, (x, y) in test_positions.items():
        xt = lookup_xt(x, y)
        print(f"  {label:35s} → xT = {xt:.4f}")
    print(f"  Grid shape: {XT_GRID.shape}")
    print(f"  Min xT: {XT_GRID.min():.4f}  Max xT: {XT_GRID.max():.4f}")

    upscaled = _load_upscaled()
    if upscaled is not None:
        print(f"  Upscaled shape: {upscaled.shape}")
        print(f"  Upscaled min: {upscaled.min():.4f}  max: {upscaled.max():.4f}")
