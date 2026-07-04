"""
Expected Threat (xT) — Trained from StatsBomb World Cup 2022 Data
===================================================================
Markov chain value iteration on StatsBomb World Cup 2022 event data.

Features:
  • Turnover-aware transitions (failed passes = absorbing states)
  • Y-axis symmetry mirroring (doubles effective sample size)
  • 2D Gaussian smoothing (removes single-event spikes)
  • Bilinear interpolation (12×8 → 120×80 continuous surface)

Pipeline:
  1. Fetch events from 64 World Cup matches (cached to Parquet)
  2. Mirror every event across Y=40 (symmetry assumption)
  3. Build transition matrix T, shot vector s, goal vector g
  4. Value iteration:  xT = T @ xT + s * g
  5. Smooth with Gaussian filter
  6. Upscale to 120×80 via bilinear interpolation
  7. Compare with Karun Singh reference

Reference:
  Karun Singh, "Introducing Expected Threat (xT)", 2018
  https://karun.in/blog/expected-threat.html
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from statsbombpy import sb
import warnings
import json
from pathlib import Path
from xt_model import XT_GRID_KARUN

warnings.filterwarnings("ignore")

# ---- Configuration ----
COMPETITION_ID = 43   # default: FIFA World Cup 2022
SEASON_ID      = 106


def int_flag(argv: list[str], flag: str, default: int) -> int:
    """Parse `--flag N` from argv; fall back to default on absence/garbage."""
    if flag in argv:
        try:
            return int(argv[argv.index(flag) + 1])
        except (IndexError, ValueError):
            print(f"  Invalid value for {flag}, using {default}")
    return default
PITCH_LENGTH   = 120.0
PITCH_WIDTH    = 80.0

N_COLS = 12   # x-axis bins
N_ROWS = 8    # y-axis bins

MAX_ITERATIONS = 2000
CONVERGENCE_THRESHOLD = 1e-8
GAUSSIAN_SIGMA = 0.8   # smoothing kernel width

OUTPUT_DIR = Path(__file__).parent


# ---- 1. Data collection (with Parquet cache) ----
def fetch_all_events(
    competition_id: int = COMPETITION_ID,
    season_id: int = SEASON_ID,
) -> pd.DataFrame:
    """Fetch events from every match of a competition, cache to Parquet."""
    if (competition_id, season_id) == (COMPETITION_ID, SEASON_ID):
        cache_path = OUTPUT_DIR / "xT_training_events.parquet"  # legacy name
    else:
        cache_path = OUTPUT_DIR / f"xT_training_events_c{competition_id}s{season_id}.parquet"
    if cache_path.exists():
        print(f"  Loading cached events from {cache_path.name}")
        return pd.read_parquet(cache_path)

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches["match_id"].tolist()
    print(f"  Fetching events from {len(match_ids)} matches...")

    all_events = []
    for i, mid in enumerate(match_ids):
        try:
            ev = sb.events(match_id=mid)
            ev["match_id"] = mid
            all_events.append(ev)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(match_ids)} matches loaded")
        except Exception as e:
            print(f"    Match {mid} failed: {e}")

    df = pd.concat(all_events, ignore_index=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Cached {len(df):,} events to {cache_path.name}")
    return df


# ---- 2. Zone discretization ----
def pos_to_zone(x: float, y: float, n_cols: int = N_COLS, n_rows: int = N_ROWS) -> int:
    col = int(np.clip(x // (PITCH_LENGTH / n_cols), 0, n_cols - 1))
    row = int(np.clip(y // (PITCH_WIDTH / n_rows), 0, n_rows - 1))
    return row * n_cols + col


# ---- 3. Build transition matrices (with Y-mirror + turnover logic) ----
def build_transition_data(
    events: pd.DataFrame,
    n_cols: int = N_COLS,
    n_rows: int = N_ROWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build T, s, g with:
    • Only successful passes as transitions (turnovers = absorbing)
    • Y-axis mirroring to enforce symmetry & double sample size
    """
    n_zones = n_cols * n_rows

    move_counts     = np.zeros((n_zones, n_zones))
    shot_counts     = np.zeros(n_zones)
    goal_counts     = np.zeros(n_zones)
    action_counts   = np.zeros(n_zones)
    turnover_counts = np.zeros(n_zones)

    # Filter possession actions
    action_types = {"Pass", "Carry", "Shot", "Dribble"}
    rel = events[events["type"].isin(action_types)].copy()
    rel = rel.dropna(subset=["location"])

    # Extract coordinates
    rel["start_x"] = rel["location"].apply(
        lambda loc: loc[0] if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
    )
    rel["start_y"] = rel["location"].apply(
        lambda loc: loc[1] if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2 else np.nan
    )
    rel = rel.dropna(subset=["start_x", "start_y"])

    def get_end_pos(row):
        if row["type"] == "Pass":
            end = row.get("pass_end_location")
            if isinstance(end, (list, tuple, np.ndarray)) and len(end) >= 2:
                return end[0], end[1]
        elif row["type"] == "Carry":
            end = row.get("carry_end_location")
            if isinstance(end, (list, tuple, np.ndarray)) and len(end) >= 2:
                return end[0], end[1]
        return None, None

    end_pos = rel.apply(get_end_pos, axis=1, result_type="expand")
    rel["end_x"] = end_pos[0]
    rel["end_y"] = end_pos[1]

    n_moves = n_shots = n_goals = n_turnovers = 0

    def process_event(sx, sy, ex, ey, ev_type, pass_outcome, dribble_outcome):
        """Process a single event at (sx, sy) → counts."""
        nonlocal n_moves, n_shots, n_goals, n_turnovers
        start_zone = pos_to_zone(sx, sy, n_cols, n_rows)

        if ev_type == "Shot":
            shot_counts[start_zone] += 1
            action_counts[start_zone] += 1
            n_shots += 1
            if pass_outcome == "Goal":  # reusing param for shot_outcome
                goal_counts[start_zone] += 1
                n_goals += 1

        elif ev_type == "Pass":
            action_counts[start_zone] += 1
            if pd.isna(pass_outcome):  # successful pass
                if pd.notna(ex) and pd.notna(ey):
                    end_zone = pos_to_zone(ex, ey, n_cols, n_rows)
                    move_counts[start_zone, end_zone] += 1
                    n_moves += 1
            else:  # failed pass → absorbing state
                turnover_counts[start_zone] += 1
                n_turnovers += 1

        elif ev_type == "Carry":
            action_counts[start_zone] += 1
            if pd.notna(ex) and pd.notna(ey):
                end_zone = pos_to_zone(ex, ey, n_cols, n_rows)
                move_counts[start_zone, end_zone] += 1
                n_moves += 1

        elif ev_type == "Dribble":
            action_counts[start_zone] += 1
            if dribble_outcome == "Complete":
                move_counts[start_zone, start_zone] += 1
                n_moves += 1
            else:
                turnover_counts[start_zone] += 1
                n_turnovers += 1

    # Process each event TWICE: original + Y-mirrored
    for _, row in rel.iterrows():
        sx, sy = row["start_x"], row["start_y"]
        ex, ey = row["end_x"], row["end_y"]
        ev_type = row["type"]
        pass_out = row.get("shot_outcome") if ev_type == "Shot" else row.get("pass_outcome")
        dribble_out = row.get("dribble_outcome")

        # Original event
        process_event(sx, sy, ex, ey, ev_type, pass_out, dribble_out)

        # Y-axis mirror: reflect across Y = 40
        sy_m = PITCH_WIDTH - sy
        ey_m = (PITCH_WIDTH - ey) if pd.notna(ey) else None
        process_event(sx, sy_m, ex, ey_m, ev_type, pass_out, dribble_out)

    # Normalize
    T = np.zeros_like(move_counts)
    for i in range(n_zones):
        if action_counts[i] > 0:
            T[i] = move_counts[i] / action_counts[i]

    s = np.zeros(n_zones)
    g = np.zeros(n_zones)
    for i in range(n_zones):
        if action_counts[i] > 0:
            s[i] = shot_counts[i] / action_counts[i]
        if shot_counts[i] > 0:
            g[i] = goal_counts[i] / shot_counts[i]

    # Sanity: P(move) + P(shot) + P(turnover) ≈ 1
    row_sums = T.sum(axis=1) + s
    absorbing_mass = 1.0 - row_sums
    active_zones = action_counts > 0
    avg_turnover_rate = absorbing_mass[active_zones].mean()

    stats = {
        "n_events": len(rel),
        "n_events_with_mirror": len(rel) * 2,
        "n_moves": n_moves,
        "n_shots": n_shots,
        "n_goals": n_goals,
        "n_turnovers": n_turnovers,
        "avg_turnover_rate": float(avg_turnover_rate),
        "n_zones_with_data": int(np.sum(active_zones)),
        "n_zones_total": n_zones,
        "total_actions": int(action_counts.sum()),
    }
    return T, s, g, stats


# ---- 4. Value iteration ----
def train_xt(T, s, g, max_iter=MAX_ITERATIONS, tol=CONVERGENCE_THRESHOLD):
    """xT(i) = Σ_j T(i,j)·xT(j) + s(i)·g(i)  until convergence."""
    xT = np.zeros(len(s))
    convergence = []
    for it in range(max_iter):
        xT_new = T @ xT + s * g
        delta = np.max(np.abs(xT_new - xT))
        convergence.append(delta)
        xT = xT_new
        if delta < tol:
            print(f"  Converged after {it + 1} iterations (Δ={delta:.2e})")
            break
    else:
        print(f"  Did not converge after {max_iter} iterations (Δ={delta:.2e})")
    return xT, convergence


# ---- 5. Post-processing: smoothing + interpolation ----
def postprocess_xt(
    xT_grid: np.ndarray,
    sigma: float = GAUSSIAN_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1. Gaussian smoothing  (removes single-event spikes)
    2. Bilinear interpolation  (12×8 → 120×80 continuous surface)

    Returns:
        smoothed : (N_ROWS, N_COLS) — smoothed 12×8 grid
        upscaled : (80, 120) — continuous 120×80 surface
    """
    # Step 1: Gaussian smoothing
    smoothed = gaussian_filter(xT_grid, sigma=sigma)

    # Step 2: Bilinear interpolation via RectBivariateSpline
    # Cell centers of the 12×8 grid
    row_centers = np.array([(r + 0.5) * (PITCH_WIDTH / N_ROWS) for r in range(N_ROWS)])
    col_centers = np.array([(c + 0.5) * (PITCH_LENGTH / N_COLS) for c in range(N_COLS)])

    # Build bilinear spline (kx=ky=1 for bilinear)
    spline = RectBivariateSpline(row_centers, col_centers, smoothed, kx=1, ky=1)

    # Evaluate on a continuous 120×80 grid (1-yard resolution)
    y_fine = np.arange(0.5, PITCH_WIDTH, 1.0)   # 80 points
    x_fine = np.arange(0.5, PITCH_LENGTH, 1.0)  # 120 points
    upscaled = spline(y_fine, x_fine)

    # Clamp negatives (interpolation edge effect)
    upscaled = np.maximum(upscaled, 0.0)

    return smoothed, upscaled


# ---- 6. Visualization ----
def plot_xt_comparison(
    trained_grid: np.ndarray,
    smoothed_grid: np.ndarray,
    upscaled_grid: np.ndarray,
    karun_grid: np.ndarray,
    convergence: list[float],
    stats: dict,
    save_path: str | None = None,
):
    """4-panel comparison: Raw, Smoothed+Upscaled, Karun Reference, Convergence."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes.flat:
        ax.set_facecolor("#0e1117")

    vmax = max(trained_grid.max(), karun_grid.max(), upscaled_grid.max())
    cmap = plt.cm.YlOrRd

    # --- Top Left: Raw trained xT (12×8) ---
    ax1 = axes[0, 0]
    im1 = ax1.imshow(trained_grid, cmap=cmap, vmin=0, vmax=vmax,
                     origin="lower", aspect="auto",
                     extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH])
    ax1.set_title("Raw Trained xT (12×8)", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlabel("x (yards)", color="white"); ax1.set_ylabel("y (yards)", color="white")
    ax1.tick_params(colors="white")
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    for r in range(trained_grid.shape[0]):
        for c in range(trained_grid.shape[1]):
            v = trained_grid[r, c]
            if v > 0.03:
                cx = (c + 0.5) * (PITCH_LENGTH / trained_grid.shape[1])
                cy = (r + 0.5) * (PITCH_WIDTH / trained_grid.shape[0])
                ax1.text(cx, cy, f"{v:.3f}", ha="center", va="center",
                        fontsize=6, color="black", fontweight="bold")

    # --- Top Right: Smoothed + Interpolated (120×80) ---
    ax2 = axes[0, 1]
    im2 = ax2.imshow(upscaled_grid, cmap=cmap, vmin=0, vmax=vmax,
                     origin="lower", aspect="auto",
                     extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH])
    ax2.set_title("Smoothed + Interpolated (120×80)", color="white", fontsize=13, fontweight="bold")
    ax2.set_xlabel("x (yards)", color="white"); ax2.set_ylabel("y (yards)", color="white")
    ax2.tick_params(colors="white")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # --- Bottom Left: Karun Singh Reference ---
    ax3 = axes[1, 0]
    im3 = ax3.imshow(karun_grid, cmap=cmap, vmin=0, vmax=vmax,
                     origin="lower", aspect="auto",
                     extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH])
    ax3.set_title("Reference xT (Karun Singh / PL)", color="white", fontsize=13, fontweight="bold")
    ax3.set_xlabel("x (yards)", color="white"); ax3.set_ylabel("y (yards)", color="white")
    ax3.tick_params(colors="white")
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    for r in range(karun_grid.shape[0]):
        for c in range(karun_grid.shape[1]):
            v = karun_grid[r, c]
            if v > 0.03:
                cx = (c + 0.5) * (PITCH_LENGTH / karun_grid.shape[1])
                cy = (r + 0.5) * (PITCH_WIDTH / karun_grid.shape[0])
                ax3.text(cx, cy, f"{v:.3f}", ha="center", va="center",
                        fontsize=6, color="black", fontweight="bold")

    # --- Bottom Right: Convergence + stats ---
    ax4 = axes[1, 1]
    ax4.semilogy(convergence, color="#00e5ff", linewidth=2)
    ax4.set_title("Value Iteration Convergence", color="white", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Iteration", color="white")
    ax4.set_ylabel("Max |Δ xT|", color="white")
    ax4.tick_params(colors="white"); ax4.grid(True, alpha=0.2)

    corr = np.corrcoef(trained_grid.ravel(), karun_grid.ravel())[0, 1]
    stats_text = (
        f"Training Data (with Y-mirror):\n"
        f"  {stats['n_events']:,} raw events\n"
        f"  {stats['n_events_with_mirror']:,} with mirror\n"
        f"  {stats['n_moves']:,} successful moves\n"
        f"  {stats['n_turnovers']:,} turnovers\n"
        f"  {stats['n_shots']:,} shots / {stats['n_goals']:,} goals\n"
        f"  Avg turnover rate: {stats['avg_turnover_rate']:.1%}\n"
        f"\nCorrelation with Karun:\n"
        f"  r = {corr:.4f}"
    )
    ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=9, color="white", va="top", ha="right",
            fontfamily="monospace",
            bbox=dict(facecolor="#1a1a2e", edgecolor="#00e5ff", alpha=0.8))

    fig.suptitle("Expected Threat (xT) — Markov Chain Training",
                color="white", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor="#0e1117", bbox_inches="tight")
        print(f"  Saved comparison plot to {save_path}")
    return fig


# ---- 7. Main pipeline ----
def main():
    import sys
    competition_id = int_flag(sys.argv, "--competition-id", COMPETITION_ID)
    season_id = int_flag(sys.argv, "--season-id", SEASON_ID)
    is_default = (competition_id, season_id) == (COMPETITION_ID, SEASON_ID)
    # non-default competitions write suffixed artifacts so the engine's
    # WC-trained surface (xt_model.py) is never clobbered by accident
    out_suffix = "" if is_default else f"_c{competition_id}s{season_id}"

    print("  Expected Threat (xT) — Markov Chain Training")
    if is_default:
        print("  Dataset: FIFA World Cup 2022 (64 matches)")
    else:
        print(f"  Dataset: StatsBomb competition {competition_id} / season {season_id}")
    print(f"  Grid: {N_COLS}×{N_ROWS} = {N_COLS * N_ROWS} zones")
    print("  Features: Y-mirror | Turnover-aware | Gaussian smooth | Bilinear interp")

    # 1. Fetch data
    print("\n1. Fetching events...")
    events = fetch_all_events(competition_id, season_id)
    print(f"  → {len(events):,} raw events")

    # 2. Build transition matrices (with Y-mirroring)
    print("\n2. Building transition matrices (with Y-axis mirroring)...")
    T, s, g, stats = build_transition_data(events, N_COLS, N_ROWS)
    print(f"  → {stats['n_moves']:,} successful moves, {stats['n_shots']:,} shots, {stats['n_goals']:,} goals")
    print(f"  → {stats['n_turnovers']:,} turnovers (absorbing states)")
    print(f"  → Avg turnover rate: {stats['avg_turnover_rate']:.1%}")
    print(f"  → {stats['n_zones_with_data']}/{stats['n_zones_total']} zones with data")

    # 3. Value iteration
    print("\n3. Training xT via value iteration...")
    xT_flat, convergence = train_xt(T, s, g)
    xT_grid = xT_flat.reshape(N_ROWS, N_COLS)
    print(f"  → Range: [{xT_grid.min():.6f}, {xT_grid.max():.6f}]")

    # 4. Post-processing
    print("\n4. Post-processing (Gaussian σ={:.1f} + bilinear interpolation)...".format(GAUSSIAN_SIGMA))
    smoothed, upscaled = postprocess_xt(xT_grid)
    print(f"  → Smoothed grid: {smoothed.shape}")
    print(f"  → Upscaled surface: {upscaled.shape}")

    # 5. Compare with reference
    print("\n5. Comparing with Karun Singh reference...")
    karun_grid = XT_GRID_KARUN
    corr = np.corrcoef(xT_grid.ravel(), karun_grid.ravel())[0, 1]
    corr_smooth = np.corrcoef(smoothed.ravel(), karun_grid.ravel())[0, 1]
    print(f"  → Raw correlation:      r = {corr:.4f}")
    print(f"  → Smoothed correlation: r = {corr_smooth:.4f}")
    print(f"  → MAE: {np.abs(smoothed - karun_grid).mean():.6f}")

    # 6. Save
    np.save(OUTPUT_DIR / f"xt_trained{out_suffix}.npy", xT_grid)
    np.save(OUTPUT_DIR / f"xt_trained_smooth{out_suffix}.npy", smoothed)
    np.save(OUTPUT_DIR / f"xt_upscaled_120x80{out_suffix}.npy", upscaled)
    with open(OUTPUT_DIR / f"xt_trained{out_suffix}.json", "w") as f:
        json.dump({
            "grid_raw": xT_grid.tolist(),
            "grid_smoothed": smoothed.tolist(),
            "shape": [N_ROWS, N_COLS],
            "stats": stats,
            "convergence_iterations": len(convergence),
            "correlation_raw": float(corr),
            "correlation_smoothed": float(corr_smooth),
            "gaussian_sigma": GAUSSIAN_SIGMA,
        }, f, indent=2)
    print(f"\n6. Saved all outputs to {OUTPUT_DIR}")

    # 7. Visualization
    print("\n7. Generating comparison plot...")
    plot_xt_comparison(xT_grid, smoothed, upscaled, karun_grid, convergence, stats,
                      save_path=str(OUTPUT_DIR / f"xt_trained_comparison{out_suffix}.png"))
    if not is_default:
        print("\n  NOTE: the engine (xt_model.py) still uses the WC-trained grid.")
        print("  For cross-competition comparisons of alpha, keep ONE fixed xT")
        print("  surface; only paste this grid into xt_model.py if you want the")
        print("  whole engine revalued on this competition.")

    # 8. Print trained grid
    print("\n8. Smoothed xT Grid (for xt_model.py):")
    print("XT_GRID = np.array([")
    for r in range(N_ROWS):
        row_str = ", ".join(f"{smoothed[r, c]:.4f}" for c in range(N_COLS))
        print(f"    [{row_str}],  # Row {r}")
    print("])")
    
    return xT_grid, smoothed, upscaled


if __name__ == "__main__":
    main()
