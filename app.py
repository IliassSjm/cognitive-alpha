"""
Cognitive Alpha — Streamlit dashboard
=======================================================
PFF tracking data integration
  • Dual data source: StatsBomb 360 + PFF 30fps Tracking
  • Real-time per-player speeds (from PFF broadcast tracking)
  • Heterogeneous biometrics / 3D ball physics / Survival density
  • Constrained Spatial xEV = PC × xT × xP_pass × xP_sprint × xP_survival
  • α = Actual_xEV − Global_Optimal_xEV

Previous features retained:
  • Multi-match analysis (16 curated World Cup matches)
  • Per-player aggregation
  • Animated tactical replay
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import streamlit as st
from mplsoccer import Pitch
from statsbombpy import sb

from xt_model import lookup_xt, lookup_xt_array, XT_GRID
from pitch_control import (
    compute_continuous_alpha,
    compute_arrival_surfaces,
    compute_pitch_control,
    compute_spatial_xev,
    compute_survival_density,
    compute_decay_penalties,
    find_global_optimal,
    XT_SURFACE,
    X_GRID, Y_GRID,
    PLAYER_SPEEDS,
)

from pff_loader import (
    extract_pff_passes,
    pff_pass_to_spatial,
    load_pff_match,
    load_tracking_window,
    extract_tracking_sequence,
    PFF_KEY_MATCHES,
    PFF_MATCH_LABELS,
    SB_TO_PFF,
)
from tracking_analytics import analyse_match as analyse_tracking_match

# ---- 0. PAGE CONFIG ----
st.set_page_config(
    page_title="Cognitive Alpha",
    page_icon="",
    layout="wide",
)

# ---- 1. CONSTANTS ----
DEFAULT_RADIUS    = 1.5
PRESSING_DISTANCE = 2.0
PRESSING_RADIUS   = 0.5

KEY_MATCHES = {
    3869685: "Final — Argentina vs France",
    3869519: "Semi — Argentina vs Croatia",
    3869552: "Semi — France vs Morocco",
    3869354: "QF — England vs France",
    3869321: "QF — Netherlands vs Argentina",
    3869420: "QF — Croatia vs Brazil",
    3869486: "QF — Morocco vs Portugal",
    3869151: "⚔️ R16 — Argentina vs Australia",
    3869118: "⚔️ R16 — England vs Senegal",
    3869254: "⚔️ R16 — Portugal vs Switzerland",
    3869253: "⚔️ R16 — Brazil vs South Korea",
    3869152: "⚔️ R16 — France vs Poland",
    3869117: "⚔️ R16 — Netherlands vs USA",
    3869219: "⚔️ R16 — Japan vs Croatia",
    3869220: "⚔️ R16 — Morocco vs Spain",
    3869684: "3rd Place — Croatia vs Morocco",
}

# Full tournament selector: curated knockout labels first, then every
# remaining match with PFF coverage (all 64 WC 2022 matches).
MATCH_OPTIONS = dict(KEY_MATCHES)
for _sb_id, _pff_id in SB_TO_PFF.items():
    if _sb_id not in MATCH_OPTIONS:
        MATCH_OPTIONS[_sb_id] = PFF_MATCH_LABELS[_pff_id]

# StatsBomb match ID → PFF game ID mapping: imported from pff_loader.SB_TO_PFF
# (single source of truth, verified against PFF metadata team names)

# ---- 2. CACHED DATA LOADING ----
@st.cache_data(show_spinner="Loading StatsBomb 360 data …")
def load_match_data(match_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    import warnings
    warnings.filterwarnings("ignore")

    events = sb.events(match_id=match_id)
    frames = sb.frames(match_id=match_id, fmt="dataframe")

    mask = (
        (events["type"] == "Pass")
        & (events["pass_outcome"].isna())
        & (events["play_pattern"] == "Regular Play")
    )
    passes = events.loc[mask].copy().reset_index(drop=True)

    merged = passes.merge(
        frames, on="id", how="inner", suffixes=("_event", "_frame")
    )

    valid_ids = merged["id"].unique()
    passes = passes.loc[passes["id"].isin(valid_ids)].reset_index(drop=True)

    # Convert ALL list-typed columns to tuples so Streamlit cache can hash them
    def _lists_to_tuples(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().iloc[:5] if len(df[col].dropna()) > 0 else pd.Series()
                if len(sample) > 0 and any(isinstance(v, list) for v in sample):
                    df[col] = df[col].apply(
                        lambda v: tuple(v) if isinstance(v, list) else v
                    )
        return df

    passes = _lists_to_tuples(passes)
    merged = _lists_to_tuples(merged)

    return passes, merged


@st.cache_data(show_spinner="Loading multi-match data …")
def load_multi_match_data(match_ids: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    import warnings
    warnings.filterwarnings("ignore")

    all_passes, all_merged = [], []
    for mid in match_ids:
        try:
            p, m = load_match_data(mid)
            p = p.copy(); m = m.copy()
            p["match_id"] = mid; m["match_id"] = mid
            all_passes.append(p); all_merged.append(m)
        except Exception:
            continue

    if not all_passes:
        return pd.DataFrame(), pd.DataFrame()
    return (
        pd.concat(all_passes, ignore_index=True),
        pd.concat(all_merged, ignore_index=True),
    )


# ---- 3. Spatial extraction (includes player names for speed lookup) ----
def extract_spatial_for_pass(
    merged: pd.DataFrame, pass_row: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Returns (bc_pos, tm_pos, def_pos, tm_names, def_names)."""
    if "match_id" in merged.columns and "match_id" in pass_row.index:
        frame_rows = merged.loc[
            (merged["id"] == pass_row["id"])
            & (merged["match_id"] == pass_row["match_id"])
        ]
    else:
        frame_rows = merged.loc[merged["id"] == pass_row["id"]]

    actor_rows = frame_rows.loc[frame_rows["actor"] == True]
    if len(actor_rows) == 0:
        ball_carrier_pos = np.array(pass_row["location"])
    else:
        ball_carrier_pos = np.array(actor_rows.iloc[0]["location_frame"])

    teammate_rows = frame_rows.loc[
        (frame_rows["teammate"] == True) & (frame_rows["actor"] == False)
    ]
    teammates_pos = np.array(teammate_rows["location_frame"].tolist())
    tm_names = teammate_rows["player"].astype(str).tolist() if "player" in teammate_rows.columns else []

    defender_rows = frame_rows.loc[frame_rows["teammate"] == False]
    defenders_pos = np.array(defender_rows["location_frame"].tolist())
    def_names = defender_rows["player"].astype(str).tolist() if "player" in defender_rows.columns else []

    return ball_carrier_pos, teammates_pos, defenders_pos, tm_names, def_names


# ---- 4. PITCH FIGURE — Continuous Heatmap ----
def create_spatial_figure(
    ball_carrier_pos: np.ndarray,
    teammates_pos: np.ndarray,
    defenders_pos: np.ndarray,
    actual_recipient_pos: np.ndarray,
    spatial_xev: np.ndarray,
    opt_x: float, opt_y: float, opt_xev: float,
    actual_xev: float, alpha: float,
    passer_name: str, recipient_name: str,
    minute: int, second: int,
    offside_line_x: float | None = None,
) -> plt.Figure:
    """
    Create pitch figure with Spatial xEV heatmap overlay.
    """
    pitch = Pitch(
        pitch_type="statsbomb", pitch_color="#0e1117",
        line_color="#ffffff", line_zorder=3, linewidth=1,
    )
    fig, ax = pitch.draw(figsize=(13, 8.5))
    fig.set_facecolor("#0e1117")

    # Heatmap — Spatial xEV
    ax.imshow(
        spatial_xev,
        extent=[0, 120, 80, 0],
        origin="upper",
        cmap="magma",
        alpha=0.65,
        aspect="auto",
        zorder=2,
        interpolation="bilinear",
    )

    # Identify keepers (closest to each goal)
    # Teammate GK: closest to x=0 (own goal)
    tm_gk_idx = None
    if len(teammates_pos) > 0:
        tm_x = teammates_pos[:, 0]
        min_x_idx = int(np.argmin(tm_x))
        if tm_x[min_x_idx] < 12:  # within 12 yards of goal
            tm_gk_idx = min_x_idx

    # Defender GK: closest to x=120 (opponent goal)
    def_gk_idx = None
    if len(defenders_pos) > 0:
        def_x = defenders_pos[:, 0]
        max_x_idx = int(np.argmax(def_x))
        if def_x[max_x_idx] > 108:
            def_gk_idx = max_x_idx

    # Defenders (outfield)
    def_outfield_mask = np.ones(len(defenders_pos), dtype=bool)
    if def_gk_idx is not None:
        def_outfield_mask[def_gk_idx] = False
    if np.any(def_outfield_mask):
        pitch.scatter(
            defenders_pos[def_outfield_mask, 0],
            defenders_pos[def_outfield_mask, 1], ax=ax,
            color="#e74c3c", s=120, edgecolors="white",
            linewidth=0.8, zorder=5, label="Defenders",
        )
    # Defender GK (orange)
    if def_gk_idx is not None:
        pitch.scatter(
            defenders_pos[def_gk_idx, 0], defenders_pos[def_gk_idx, 1],
            ax=ax, color="#ff9800", s=180, marker="s", edgecolors="white",
            linewidth=1.2, zorder=5, label="GK (Opp)",
        )

    # Teammates (outfield)
    tm_outfield_mask = np.ones(len(teammates_pos), dtype=bool)
    if tm_gk_idx is not None:
        tm_outfield_mask[tm_gk_idx] = False
    if np.any(tm_outfield_mask):
        pitch.scatter(
            teammates_pos[tm_outfield_mask, 0],
            teammates_pos[tm_outfield_mask, 1], ax=ax,
            color="#3498db", s=120, edgecolors="white",
            linewidth=0.8, zorder=5, label="Teammates",
        )
    # Teammate GK (lime green)
    if tm_gk_idx is not None:
        pitch.scatter(
            teammates_pos[tm_gk_idx, 0], teammates_pos[tm_gk_idx, 1],
            ax=ax, color="#76ff03", s=180, marker="s", edgecolors="white",
            linewidth=1.2, zorder=5, label="GK (Own)",
        )
    # Ball carrier (white diamond — distinct from optimal star)
    pitch.scatter(
        ball_carrier_pos[0], ball_carrier_pos[1], ax=ax,
        color="white", s=200, marker="D",
        edgecolors="#f1c40f", linewidth=1.5, zorder=6,
        label="Ball Carrier",
    )

    # Actual pass (white arrow)
    pitch.arrows(
        ball_carrier_pos[0], ball_carrier_pos[1],
        actual_recipient_pos[0], actual_recipient_pos[1], ax=ax,
        color="white", width=2.5, headwidth=6,
        headlength=4, zorder=6, label="Actual Pass",
    )

    # Global Optimal Target — gold crosshair with glow
    ax.scatter(
        opt_x, opt_y,
        s=600, marker="*", c="#FFD700",
        edgecolors="#FFD700", linewidth=1.5, zorder=7,
        label="Optimal Space",
    )
    # Glow ring
    ax.scatter(
        opt_x, opt_y,
        s=1200, marker="o", facecolors="none",
        edgecolors="#FFD700", linewidth=2, alpha=0.5, zorder=6,
    )
    # Dashed cyan line from carrier to optimal
    ax.plot(
        [ball_carrier_pos[0], opt_x],
        [ball_carrier_pos[1], opt_y],
        color="#00e5ff", linewidth=1.8, linestyle="--",
        alpha=0.7, zorder=5,
    )

    # Offside line (red dashed)
    if offside_line_x is not None:
        ax.axvline(
            x=offside_line_x + 1.0, color="#ff4444", linewidth=1.5,
            linestyle="--", alpha=0.6, zorder=4, label="Offside Line",
        )

    # Titles
    sign = "+" if alpha >= 0 else ""
    fig.suptitle(
        f"Spatial α: {sign}{alpha:.4f}"
        f"   |   Optimal xEV: {opt_xev:.4f}"
        f"   |   Actual xEV: {actual_xev:.4f}",
        color="white", fontsize=14, fontweight="bold", y=0.97,
    )
    n_visible = 1 + len(teammates_pos) + len(defenders_pos)  # +1 for carrier
    ax.set_title(
        f"{passer_name} \u2192 {recipient_name}  ({minute}:{second:02d})"
        f"   |   \u2666 = Carrier   \u2605 = Optimal"
        f"   |   {n_visible}/22 visible",
        color="#aaaaaa", fontsize=10, pad=14,
    )

    legend = ax.legend(
        loc="lower left", fontsize=8, framealpha=0.7,
        facecolor="#1a1a2e", edgecolor="#ffffff", labelcolor="white",
    )
    legend.set_zorder(10)
    plt.tight_layout()
    return fig


# ---- 5. SINGLE-PASS ANALYSIS PIPELINE (Continuous) ----
def analyse_pass_continuous(
    passes: pd.DataFrame, merged: pd.DataFrame, pass_idx: int,
    pass_type: str = "Ground Pass",
) -> tuple[plt.Figure, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyse a single pass with the 3D kinematics model.
    Returns (fig, metrics, bc_pos, tm_pos, def_pos, actual_pos).
    """
    pass_row = passes.iloc[pass_idx]
    ball_carrier_pos, teammates_pos, defenders_pos, tm_names, def_names = (
        extract_spatial_for_pass(merged, pass_row)
    )

    empty_metrics = {
        "actual_xev": 0.0, "opt_xev": 0.0, "alpha": 0.0,
        "opt_x": 60.0, "opt_y": 40.0, "actual_pc": 0.0, "actual_xt": 0.0,
        "actual_survival": 1.0, "pass_type": pass_type,
        "spatial_xev": np.zeros((80, 120)),
        "pc_surface": np.zeros((80, 120)),
        "xP_survival": np.ones((80, 120)),
    }

    if len(teammates_pos) == 0 or len(defenders_pos) == 0:
        pitch = Pitch(
            pitch_type="statsbomb", pitch_color="#0e1117",
            line_color="#ffffff", line_zorder=2, linewidth=1,
        )
        fig, ax = pitch.draw(figsize=(13, 8.5))
        fig.set_facecolor("#0e1117")
        ax.set_title("No teammates/defenders visible", color="#aaaaaa")
        return (fig, empty_metrics, np.zeros(2),
                np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(2))

    actual_end = np.array(pass_row["pass_end_location"])
    dists_to_end = np.linalg.norm(teammates_pos - actual_end, axis=1)
    actual_recipient_pos = teammates_pos[int(np.argmin(dists_to_end))]

    # 3D kinematics (vectorized, ~2ms)
    result = compute_continuous_alpha(
        ball_carrier_pos, teammates_pos, defenders_pos, actual_end,
        pass_type=pass_type,
        teammate_names=tm_names,
        defender_names=def_names,
    )

    passer_name = str(pass_row.get("player", "Unknown"))
    recipient_name = str(pass_row.get("pass_recipient", "Unknown"))
    minute = int(pass_row["minute"])
    second = int(pass_row["second"])

    fig = create_spatial_figure(
        ball_carrier_pos, teammates_pos, defenders_pos,
        actual_recipient_pos, result["spatial_xev"],
        result["opt_x"], result["opt_y"], result["opt_xev"],
        result["actual_xev"], result["alpha"],
        passer_name, recipient_name, minute, second,
        offside_line_x=result.get("offside_line_x"),
    )

    return (fig, result, ball_carrier_pos, teammates_pos,
            defenders_pos, actual_recipient_pos)


# ---- 6. BATCH AGGREGATION (Continuous) ----
@st.cache_data(show_spinner="Computing Spatial Alpha for all passes …")
def aggregate_all_passes(
    passes: pd.DataFrame, merged: pd.DataFrame, match_key: str = "",
) -> pd.DataFrame:
    records = []
    for i in range(passes.shape[0]):
        row = passes.iloc[i]
        try:
            bc, tm, df, tm_n, df_n = extract_spatial_for_pass(merged, row)
            if len(tm) == 0 or len(df) == 0:
                continue
            actual_end = np.array(row["pass_end_location"])
            m = compute_continuous_alpha(
                bc, tm, df, actual_end,
                teammate_names=tm_n, defender_names=df_n,
            )
            records.append({
                "player": str(row.get("player", "Unknown")),
                "team": str(row.get("team", "Unknown")),
                "alpha": m["alpha"],
                "actual_xev": m["actual_xev"],
                "opt_xev": m["opt_xev"],
                "actual_pc": m["actual_pc"],
                "actual_xt": m["actual_xt"],
                "actual_survival": m["actual_survival"],
            })
        except Exception:
            continue

    df_records = pd.DataFrame(records)
    if df_records.empty:
        return pd.DataFrame()

    agg = df_records.groupby(["player", "team"]).agg(
        total_passes=("alpha", "count"),
        mean_alpha=("alpha", "mean"),
        min_alpha=("alpha", "min"),
        pct_suboptimal=("alpha", lambda s: (s < -0.005).mean() * 100),
        mean_actual_xev=("actual_xev", "mean"),
        sum_actual_xev=("actual_xev", "sum"),
        mean_opt_xev=("opt_xev", "mean"),
        mean_pc=("actual_pc", "mean"),
    ).reset_index()

    agg = agg.sort_values("mean_alpha", ascending=True).reset_index(drop=True)
    return agg


# ---- 7. ANIMATED TACTICAL REPLAY (continuous version) ----
def create_tactical_animation(
    ball_carrier_pos: np.ndarray,
    teammates_pos: np.ndarray,
    defenders_pos: np.ndarray,
    actual_recipient_pos: np.ndarray,
    spatial_xev: np.ndarray,
    opt_x: float, opt_y: float,
    passer_name: str, recipient_name: str,
    minute: int, second: int,
    n_frames: int = 40,
    fps: int = 10,
) -> bytes:
    """
    Animated GIF with continuous heatmap:
      Phase 1: Empty pitch + players
      Phase 2: Heatmap fades in
      Phase 3: Optimal target appears
      Phase 4: Ball flight
      Phase 5: Final with alpha annotation
    """
    pitch = Pitch(
        pitch_type="statsbomb", pitch_color="#0e1117",
        line_color="#ffffff", line_zorder=3, linewidth=1,
    )
    fig, ax = pitch.draw(figsize=(11, 7))
    fig.set_facecolor("#0e1117")

    # Static elements
    ax.scatter(
        defenders_pos[:, 0], defenders_pos[:, 1],
        c="#e74c3c", s=100, edgecolors="white", linewidth=0.8, zorder=5,
    )
    ax.scatter(
        teammates_pos[:, 0], teammates_pos[:, 1],
        c="#3498db", s=100, edgecolors="white", linewidth=0.8, zorder=5,
    )
    ax.scatter(
        [ball_carrier_pos[0]], [ball_carrier_pos[1]],
        c="#f1c40f", s=300, marker="*", edgecolors="white",
        linewidth=0.8, zorder=6,
    )

    # Dynamic elements
    heatmap_img = [None]
    ball_dot, = ax.plot([], [], "o", color="white", markersize=8, zorder=8)
    opt_marker = [None]
    opt_line = [None]
    pass_trail = [None]
    phase_text = ax.text(
        60, 2, "", fontsize=11, color="#f1c40f", ha="center",
        fontweight="bold", zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0e1117",
                  edgecolor="#f1c40f", alpha=0.9),
    )

    ax.set_title(
        f"{passer_name} → {recipient_name}  ({minute}:{second:02d})",
        color="#aaaaaa", fontsize=10, pad=10,
    )

    def animate(frame):
        progress = frame / max(n_frames - 1, 1)

        # Clean up dynamic elements from previous frame
        if heatmap_img[0] is not None:
            heatmap_img[0].remove()
            heatmap_img[0] = None
        if opt_marker[0] is not None:
            opt_marker[0].remove()
            opt_marker[0] = None
        if opt_line[0] is not None:
            opt_line[0].remove()
            opt_line[0] = None
        if pass_trail[0] is not None:
            pass_trail[0].remove()
            pass_trail[0] = None

        # Phase 1: Freeze frame (0-15%)
        if progress < 0.15:
            phase_text.set_text("⏸ Freeze Frame")
            ball_dot.set_data([ball_carrier_pos[0]], [ball_carrier_pos[1]])

        # Phase 2: Heatmap fades in (15-40%)
        elif progress < 0.40:
            fade = (progress - 0.15) / 0.25
            phase_text.set_text("🗺️ Spatial Value Surface")
            ball_dot.set_data([ball_carrier_pos[0]], [ball_carrier_pos[1]])
            heatmap_img[0] = ax.imshow(
                spatial_xev, extent=[0, 120, 80, 0], origin="upper",
                cmap="magma", alpha=0.65 * fade, aspect="auto",
                zorder=2, interpolation="bilinear",
            )

        # Phase 3: Optimal target appears (40-60%)
        elif progress < 0.60:
            phase_text.set_text("Optimal Spatial Target")
            ball_dot.set_data([ball_carrier_pos[0]], [ball_carrier_pos[1]])
            heatmap_img[0] = ax.imshow(
                spatial_xev, extent=[0, 120, 80, 0], origin="upper",
                cmap="magma", alpha=0.65, aspect="auto",
                zorder=2, interpolation="bilinear",
            )
            pulse = 0.5 + 0.5 * np.sin((progress - 0.4) / 0.2 * np.pi * 4)
            opt_marker[0] = ax.scatter(
                opt_x, opt_y, s=600 + 200 * pulse, marker="*",
                c="#FFD700", edgecolors="#FFD700", linewidth=1.5,
                alpha=0.9, zorder=7,
            )
            opt_line[0], = ax.plot(
                [ball_carrier_pos[0], opt_x],
                [ball_carrier_pos[1], opt_y],
                color="#FFD700", linewidth=2, linestyle="--",
                alpha=0.7, zorder=5,
            )

        # Phase 4: Ball flight (60-80%)
        elif progress < 0.80:
            flight = (progress - 0.60) / 0.20
            phase_text.set_text("Actual Pass")
            ball_pos = ball_carrier_pos + flight * (
                actual_recipient_pos - ball_carrier_pos
            )
            ball_dot.set_data([ball_pos[0]], [ball_pos[1]])
            heatmap_img[0] = ax.imshow(
                spatial_xev, extent=[0, 120, 80, 0], origin="upper",
                cmap="magma", alpha=0.5, aspect="auto",
                zorder=2, interpolation="bilinear",
            )
            opt_marker[0] = ax.scatter(
                opt_x, opt_y, s=600, marker="*",
                c="#FFD700", edgecolors="#FFD700", linewidth=1.5,
                alpha=0.7, zorder=7,
            )
            pass_trail[0], = ax.plot(
                [ball_carrier_pos[0], ball_pos[0]],
                [ball_carrier_pos[1], ball_pos[1]],
                color="white", linewidth=2.5, alpha=0.8, zorder=6,
            )

        # Phase 5: Final state (80-100%)
        else:
            phase_text.set_text("Spatial Arbitrage")
            ball_dot.set_data(
                [actual_recipient_pos[0]], [actual_recipient_pos[1]]
            )
            heatmap_img[0] = ax.imshow(
                spatial_xev, extent=[0, 120, 80, 0], origin="upper",
                cmap="magma", alpha=0.55, aspect="auto",
                zorder=2, interpolation="bilinear",
            )
            opt_marker[0] = ax.scatter(
                opt_x, opt_y, s=800, marker="*",
                c="#FFD700", edgecolors="#FFD700", linewidth=2,
                alpha=0.9, zorder=7,
            )
            opt_line[0], = ax.plot(
                [ball_carrier_pos[0], opt_x],
                [ball_carrier_pos[1], opt_y],
                color="#FFD700", linewidth=2, linestyle="--",
                alpha=0.6, zorder=5,
            )
            pass_trail[0], = ax.plot(
                [ball_carrier_pos[0], actual_recipient_pos[0]],
                [ball_carrier_pos[1], actual_recipient_pos[1]],
                color="white", linewidth=2.5, alpha=0.9, zorder=6,
            )

        return [ball_dot, phase_text]

    anim = FuncAnimation(
        fig, animate, frames=n_frames,
        interval=1000 // fps, blit=False,
    )

    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        anim.save(tmp_path, writer="pillow", fps=fps)
        plt.close(fig)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---- 7b. 30-FPS TRACKING REPLAY ANIMATION ----
TRAIL_LENGTH = 4   # ghosting frames for motion trails


def create_tracking_replay(
    tracking_frames: list[dict],
    spatial_xev: np.ndarray,
    opt_x: float, opt_y: float,
    passer_name: str, recipient_name: str,
    minute: int, second: int,
    is_home_possession: bool = True,
    subsample: int = 3,
    render_fps: int = 10,
) -> bytes:
    """
    Animate 22 tracking dots at 30fps for ~3s before a pass, ending
    with a magma xEV heatmap eruption on the final frames.

    Parameters
    ----------
    tracking_frames : Output of load_tracking_window()
    spatial_xev : (80, 120) xEV surface from pitch control
    opt_x, opt_y : Optimal target coordinates
    is_home_possession : True if passer is on home team
    subsample : Take every Nth tracking frame (3 → 10fps GIF from 30fps data)
    render_fps : GIF output framerate
    """
    if not tracking_frames:
        return b""

    # Subsample tracking frames to manage GIF size
    anim_frames = tracking_frames[::subsample]
    n_track = len(anim_frames)

    # === Precompute dynamic xEV surfaces per frame ===
    spatial_sequence = extract_tracking_sequence(anim_frames, is_home_possession)
    # Build a mapping: frame_idx → spatial_xev surface
    precomputed_xev: dict[int, np.ndarray] = {}
    precomputed_opt: dict[int, tuple[float, float]] = {}
    xev_interval = max(1, n_track // 10)  # compute ~10 surfaces total
    for fi in range(0, n_track, xev_interval):
        if fi < len(spatial_sequence):
            sp = spatial_sequence[fi]
            try:
                r = compute_continuous_alpha(
                    sp["bc_pos"], sp["tm_pos"], sp["def_pos"],
                    sp["tm_pos"][0] if len(sp["tm_pos"]) > 0 else sp["bc_pos"],
                )
                precomputed_xev[fi] = r["spatial_xev"]
                precomputed_opt[fi] = (r["opt_x"], r["opt_y"])
            except Exception:
                pass
    # Always ensure last frame has an xEV surface
    if n_track - 1 not in precomputed_xev and len(spatial_sequence) > 0:
        sp = spatial_sequence[-1]
        try:
            r = compute_continuous_alpha(
                sp["bc_pos"], sp["tm_pos"], sp["def_pos"],
                sp["tm_pos"][0] if len(sp["tm_pos"]) > 0 else sp["bc_pos"],
            )
            precomputed_xev[n_track - 1] = r["spatial_xev"]
            precomputed_opt[n_track - 1] = (r["opt_x"], r["opt_y"])
        except Exception:
            pass
    # Fallback: use the passed-in static surface
    final_xev = precomputed_xev.get(n_track - 1, spatial_xev)
    final_opt = precomputed_opt.get(n_track - 1, (opt_x, opt_y))

    # Reserve extra frames for heatmap eruption + hold
    ERUPTION_FRAMES = 8
    HOLD_FRAMES = 6
    total_frames = n_track + ERUPTION_FRAMES + HOLD_FRAMES

    # ── Build figure ──
    pitch = Pitch(
        pitch_type="statsbomb", pitch_color="#0e1117",
        line_color="#ffffff", line_zorder=3, linewidth=1,
    )
    fig, ax = pitch.draw(figsize=(11, 7))
    fig.set_facecolor("#0e1117")

    # Title
    ax.set_title(
        f"{passer_name} → {recipient_name}  ({minute}:{second:02d})\n"
        f"PFF 30fps Tracking • {len(tracking_frames)} raw frames",
        color="#aaaaaa", fontsize=10, pad=10,
    )

    # Dynamic artist references
    home_scatter = [None]
    away_scatter = [None]
    ball_dot, = ax.plot([], [], "o", color="#f1c40f", markersize=9,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=8)
    trails_home = []
    trails_away = []
    heatmap_img = [None]
    opt_marker = [None]
    opt_line = [None]

    # Timer text
    timer_text = ax.text(
        60, 2, "", fontsize=11, color="#f1c40f", ha="center",
        fontweight="bold", zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0e1117",
                  edgecolor="#f1c40f", alpha=0.9),
    )

    # Speed legend
    ax.text(
        117, 78, "● sprint  ○ jog", fontsize=7, color="#888888",
        ha="right", va="bottom", zorder=10,
    )

    # Identify possession team colors
    att_color = "#00e5ff"  # cyan for attacking
    def_color = "#e74c3c"  # red for defending

    def _clear_dynamic():
        for ref in [home_scatter, away_scatter, heatmap_img, opt_marker, opt_line]:
            if ref[0] is not None:
                ref[0].remove()
                ref[0] = None
        for t in trails_home:
            t.remove()
        trails_home.clear()
        for t in trails_away:
            t.remove()
        trails_away.clear()

    def animate(frame_idx):
        _clear_dynamic()

        # Phase 1: Tracking movement (0 to n_track-1)
        if frame_idx < n_track:
            fr = anim_frames[frame_idx]
            t_offset = fr["time_offset"]

            # Timer countdown
            timer_text.set_text(f"T{t_offset:+.1f}s")

            # Home players
            home = fr["home_players"]
            if home:
                hx = [p[0] for p in home]
                hy = [p[1] for p in home]
                speeds = [p[2] for p in home]
                sizes = [40 + 90 * min(s / 8.0, 1.0) for s in speeds]
                c = att_color if is_home_possession else def_color
                home_scatter[0] = ax.scatter(
                    hx, hy, s=sizes, c=c, edgecolors="white",
                    linewidth=0.6, alpha=0.9, zorder=5,
                )

            # Away players
            away = fr["away_players"]
            if away:
                ax_coords = [p[0] for p in away]
                ay_coords = [p[1] for p in away]
                speeds = [p[2] for p in away]
                sizes = [40 + 90 * min(s / 8.0, 1.0) for s in speeds]
                c = def_color if is_home_possession else att_color
                away_scatter[0] = ax.scatter(
                    ax_coords, ay_coords, s=sizes, c=c, edgecolors="white",
                    linewidth=0.6, alpha=0.9, zorder=5,
                )

            # Ball
            bp = fr["ball"]
            if bp:
                ball_dot.set_data([bp[0]], [bp[1]])
            else:
                ball_dot.set_data([], [])

            # === Dynamic xEV heatmap (precomputed) ===
            # Find nearest precomputed xEV surface
            nearest_xev_idx = 0
            for ki in sorted(precomputed_xev.keys()):
                if ki <= frame_idx:
                    nearest_xev_idx = ki
            if nearest_xev_idx in precomputed_xev and frame_idx >= n_track // 3:
                # Start showing heatmap from 1/3 through tracking
                heatmap_fade = min(1.0, (frame_idx - n_track // 3) / max(n_track // 3, 1))
                heatmap_img[0] = ax.imshow(
                    precomputed_xev[nearest_xev_idx],
                    extent=[0, 120, 80, 0], origin="upper",
                    cmap="magma", alpha=0.45 * heatmap_fade, aspect="auto",
                    zorder=2, interpolation="bilinear",
                )
                # Show dynamic optimal target
                if nearest_xev_idx in precomputed_opt and heatmap_fade > 0.5:
                    ox, oy = precomputed_opt[nearest_xev_idx]
                    opt_marker[0] = ax.scatter(
                        ox, oy, s=400, marker="*",
                        c="#FFD700", edgecolors="#FFD700", linewidth=1.2,
                        alpha=0.7 * heatmap_fade, zorder=7,
                    )

        # Phase 2: Heatmap eruption (n_track to n_track + ERUPTION)
        elif frame_idx < n_track + ERUPTION_FRAMES:
            erupt_progress = (frame_idx - n_track) / max(ERUPTION_FRAMES - 1, 1)
            timer_text.set_text("xEV Surface")

            # Keep final player positions
            last = anim_frames[-1]
            if last["home_players"]:
                hx = [p[0] for p in last["home_players"]]
                hy = [p[1] for p in last["home_players"]]
                c = att_color if is_home_possession else def_color
                home_scatter[0] = ax.scatter(
                    hx, hy, s=100, c=c, edgecolors="white",
                    linewidth=0.8, alpha=0.9, zorder=5,
                )
            if last["away_players"]:
                ax_c = [p[0] for p in last["away_players"]]
                ay_c = [p[1] for p in last["away_players"]]
                c = def_color if is_home_possession else att_color
                away_scatter[0] = ax.scatter(
                    ax_c, ay_c, s=100, c=c, edgecolors="white",
                    linewidth=0.8, alpha=0.9, zorder=5,
                )
            bp = last["ball"]
            if bp:
                ball_dot.set_data([bp[0]], [bp[1]])

            # Heatmap fades in to full
            heatmap_img[0] = ax.imshow(
                final_xev, extent=[0, 120, 80, 0], origin="upper",
                cmap="magma", alpha=0.65 * erupt_progress, aspect="auto",
                zorder=2, interpolation="bilinear",
            )

            # Optimal target pulses in
            if erupt_progress > 0.5:
                pulse = 0.5 + 0.5 * np.sin(erupt_progress * np.pi * 4)
                fx, fy = final_opt
                opt_marker[0] = ax.scatter(
                    fx, fy, s=500 + 200 * pulse, marker="*",
                    c="#FFD700", edgecolors="#FFD700", linewidth=1.5,
                    alpha=0.9, zorder=7,
                )

        # Phase 3: Final hold
        else:
            timer_text.set_text("Spatial Arbitrage")

            last = anim_frames[-1]
            if last["home_players"]:
                hx = [p[0] for p in last["home_players"]]
                hy = [p[1] for p in last["home_players"]]
                c = att_color if is_home_possession else def_color
                home_scatter[0] = ax.scatter(
                    hx, hy, s=100, c=c, edgecolors="white",
                    linewidth=0.8, alpha=0.9, zorder=5,
                )
            if last["away_players"]:
                ax_c = [p[0] for p in last["away_players"]]
                ay_c = [p[1] for p in last["away_players"]]
                c = def_color if is_home_possession else att_color
                away_scatter[0] = ax.scatter(
                    ax_c, ay_c, s=100, c=c, edgecolors="white",
                    linewidth=0.8, alpha=0.9, zorder=5,
                )
            bp = last["ball"]
            if bp:
                ball_dot.set_data([bp[0]], [bp[1]])

            heatmap_img[0] = ax.imshow(
                final_xev, extent=[0, 120, 80, 0], origin="upper",
                cmap="magma", alpha=0.55, aspect="auto",
                zorder=2, interpolation="bilinear",
            )
            fx, fy = final_opt
            opt_marker[0] = ax.scatter(
                fx, fy, s=800, marker="*",
                c="#FFD700", edgecolors="#FFD700", linewidth=2,
                alpha=0.9, zorder=7,
            )

        return [ball_dot, timer_text]

    anim = FuncAnimation(
        fig, animate, frames=total_frames,
        interval=1000 // render_fps, blit=False,
    )

    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        anim.save(tmp_path, writer="pillow", fps=render_fps)
        plt.close(fig)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---- 8. STREAMLIT APP ----
def main():
    st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
        [data-testid="stSidebar"] { background-color: #0e1117; }
        h1 { text-align: center; }

        /* ── Gradient Metric Cards ── */
        [data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.8));
            border: 1px solid rgba(100, 200, 255, 0.15);
            border-radius: 12px;
            padding: 14px 18px;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(0, 229, 255, 0.15);
            border-color: rgba(0, 229, 255, 0.3);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.7;
        }
        [data-testid="stMetricDelta"] > div {
            font-size: 0.85rem;
        }

        /* ── Tab styling ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Cognitive Alpha: Quantifying Spatial Opportunity Cost")
    st.caption(
        "2022 FIFA World Cup  ·  "
        "**Continuous Pitch Control** × **Interpolated xT** = **Spatial xEV**"
    )

    # ── Sidebar ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🏟️ Match")
        match_options = list(MATCH_OPTIONS.items())
        match_labels = [label for _, label in match_options]
        match_ids_list = [mid for mid, _ in match_options]

        selected_match_idx = st.selectbox(
            "Select a match",
            range(len(match_labels)),
            format_func=lambda i: match_labels[i],
            index=0,
        )
        selected_match_id = match_ids_list[selected_match_idx]

        st.markdown("---")
        multi_match = st.checkbox("Multi-Match Mode", value=False)
        if multi_match:
            multi_options = st.multiselect(
                "Matches for aggregation",
                options=match_ids_list,
                format_func=lambda mid: MATCH_OPTIONS[mid],
                default=[3869685, 3869519, 3869552],
            )
        else:
            multi_options = [selected_match_id]

        st.markdown("---")
        st.header("📡 Data Source")
        data_source = st.radio(
            "Choose data source",
            ["StatsBomb 360", "PFF Tracking"],
            index=0,
            horizontal=True,
            help="PFF: real-time speeds from 30fps broadcast tracking.",
        )
        use_pff = data_source == "PFF Tracking"

        st.markdown("---")
        st.header("🏃 Pass Trajectory")
        if not use_pff:
            pass_type = st.radio(
                "Simulate pass type",
                ["Ground Pass", "Lofted Pass"],
                index=0,
                horizontal=True,
                help="Ground: 15 m/s, 30yd decay. Lofted: 11 m/s, 50yd decay.",
            )
        else:
            pass_type = "Ground Pass"  # unused — PFF auto-detects from ball_z
            st.info("PFF mode: pass type auto-detected from ball Z-axis (z > 1.5m = lofted)")
        st.markdown("---")
        with st.expander("⚙️ Model Details", expanded=False):
            st.markdown(
                "**3D kinematics + trained xT**\n\n"
                "• Meshgrid 120×80 (1 yd)\n"
                "• Kinematic acceleration profiles\n"
                "• xT trained via Markov chain (WC 2022)\n"
                "• `α = xEV(actual) − xEV(optimal)`\n\n"
                "`xEV = PC × xT × xP × xS × xSurv`"
            )
        st.markdown("---")
        st.caption("Built with ❤️ for football analytics")

    # ── Load data ──────────────────────────────────────────────────
    pff_passes = None
    passes = pd.DataFrame()
    merged = pd.DataFrame()

    if use_pff:
        selected_pff_id = SB_TO_PFF.get(selected_match_id)
        if selected_pff_id is None:
            st.warning(
                "No PFF tracking coverage for this match. "
                "Choose a QF/Semi/Final/3rd-place match or switch to StatsBomb 360."
            )
            return
        pff_passes = extract_pff_passes(selected_pff_id)
        if pff_passes.empty:
            st.warning("No PFF data loaded.")
            return
    else:
        if multi_match and len(multi_options) > 1:
            passes, merged = load_multi_match_data(tuple(multi_options))
        else:
            passes, merged = load_match_data(selected_match_id)
        if passes.empty:
            st.warning("No data loaded.")
            return

    # ── Tabs ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Spatial Analysis", "Player Rankings", "Tactical Replay",
        "📋 Match Report", "📡 Tracking Deep Dive"
    ])

    # ── Pass Selector (shared state) ──────────────────────────────
    if use_pff:
        labels = []
        for _, row in pff_passes.iterrows():
            minute = int(row["minute"])
            second = int(row["second"])
            pname = str(row["passer_name"])
            rname = str(row["receiver_name"])
            labels.append(f"{minute:3d}:{second:02d}  —  {pname} → {rname}")
        with st.sidebar:
            st.markdown("---")
            st.header("Pass")
            selected_idx = st.selectbox(
                "Select a pass",
                range(len(labels)),
                format_func=lambda i: labels[i],
                index=min(20, len(labels) - 1),
            )
    elif not (multi_match and len(multi_options) > 1):
        labels = []
        for _, row in passes.iterrows():
            minute = int(row["minute"])
            second = int(row["second"])
            player = str(row.get("player", "Unknown"))
            parts = player.split()
            short_name = f"{parts[0][0]}. {parts[-1]}" if len(parts) > 2 else player
            recipient = str(row.get("pass_recipient", "Unknown"))
            r_parts = recipient.split()
            short_r = f"{r_parts[0][0]}. {r_parts[-1]}" if len(r_parts) > 2 else recipient
            labels.append(f"{minute:3d}:{second:02d}  —  {short_name} → {short_r}")

        with st.sidebar:
            st.markdown("---")
            st.header("Pass")
            selected_idx = st.selectbox(
                "Select a pass",
                range(len(labels)),
                format_func=lambda i: labels[i],
                index=0,
            )
    else:
        selected_idx = 0

    # ── TAB 1: Spatial Analysis ──────────────────────────────────
    with tab1:
        if use_pff:
            # ── PFF Tracking Analysis ─────────────────────────────────
            pff_row = pff_passes.iloc[selected_idx]
            spatial = pff_pass_to_spatial(pff_row)

            if len(spatial["tm_pos"]) == 0 or len(spatial["def_pos"]) == 0:
                st.warning("Insufficient player data for this pass.")
            else:
                result = compute_continuous_alpha(
                    spatial["bc_pos"], spatial["tm_pos"], spatial["def_pos"],
                    spatial["actual_end"],
                    pass_type=spatial["pass_type"],
                    teammate_names=spatial["tm_names"],
                    defender_names=spatial["def_names"],
                    teammate_speeds=spatial["tm_speeds"],
                    defender_speeds=spatial["def_speeds"],
                    teammate_velocities=spatial.get("tm_velocities"),
                    attack_right=spatial.get("attack_right", True),
                    passer_velocity=spatial.get("passer_velocity"),
                )

                # Find actual recipient
                dists = np.linalg.norm(
                    spatial["tm_pos"] - spatial["actual_end"], axis=1
                )
                actual_recipient = spatial["tm_pos"][int(np.argmin(dists))]

                fig = create_spatial_figure(
                    spatial["bc_pos"], spatial["tm_pos"], spatial["def_pos"],
                    actual_recipient, result["spatial_xev"],
                    result["opt_x"], result["opt_y"], result["opt_xev"],
                    result["actual_xev"], result["alpha"],
                    pff_row["passer_name"], pff_row["receiver_name"],
                    int(pff_row["minute"]), int(pff_row["second"]),
                )

                # Metrics Row 1
                traj = "🏈 Lofted" if spatial["pass_type"] == "Lofted Pass" else "Ground"
                st.markdown(f"#### 📡 PFF Tracking — {traj}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Actual Pass xEV", f"{result['actual_xev']:.4f}")
                with c2:
                    st.metric("Optimal Spatial xEV", f"{result['opt_xev']:.4f}")
                with c3:
                    st.metric("Spatial α", f"{result['alpha']:+.4f}",
                              delta=f"{result['alpha']:+.4f}",
                              delta_color="normal")

                # Metrics Row 2
                c4, c5, c6 = st.columns(3)
                with c4:
                    st.metric("Actual Pitch Control", f"{result['actual_pc']:.0%}")
                with c5:
                    st.metric("Optimal Location",
                              f"({result['opt_x']:.0f}, {result['opt_y']:.0f})")
                with c6:
                    st.metric("Receiver Survival",
                              f"{result.get('actual_survival', 1.0):.0%}")

                # Metrics Row 3: PFF-exclusive data
                c7, c8, c9 = st.columns(3)
                with c7:
                    st.metric("Passer Speed",
                              f"{pff_row['passer_speed']:.1f} m/s")
                with c8:
                    st.metric("Ball Height (z)",
                              f"{pff_row['ball_z']:.2f} m")
                with c9:
                    has_vel = spatial.get("tm_velocities") is not None
                    st.metric("Body Orientation",
                              "Active (Kinematic Vectors)" if has_vel
                              else "Inactive (Scalar Only)")

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        elif multi_match and len(multi_options) > 1:
            st.info("Switch to single-match mode to analyse individual passes.")
        else:
            result_data = analyse_pass_continuous(
                passes, merged, selected_idx, pass_type=pass_type,
            )
            fig, metrics, bc_pos, tm_pos, def_pos, actual_pos = result_data

            # Metrics: Continuous Spatial Arbitrage
            traj_label = "🏈 Lofted" if pass_type == "Lofted Pass" else "Ground"
            st.markdown(f"#### Continuous Spatial Arbitrage — {traj_label}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Actual Pass xEV", f"{metrics['actual_xev']:.4f}")
            with c2:
                st.metric("Optimal Spatial xEV", f"{metrics['opt_xev']:.4f}")
            with c3:
                st.metric("Spatial α", f"{metrics['alpha']:+.4f}",
                          delta=f"{metrics['alpha']:+.4f}",
                          delta_color="normal")

            # Row 2: Supporting metrics
            c4, c5, c6 = st.columns(3)
            with c4:
                st.metric("Actual Pitch Control", f"{metrics['actual_pc']:.0%}")
            with c5:
                st.metric("Optimal Location",
                          f"({metrics['opt_x']:.0f}, {metrics['opt_y']:.0f})")
            with c6:
                st.metric("Actual xT", f"{metrics['actual_xt']:.4f}")

            # Row 3: v6 survival + trajectory
            c7, c8, c9 = st.columns(3)
            with c7:
                st.metric("Receiver Survival",
                          f"{metrics.get('actual_survival', 1.0):.0%}")
            with c8:
                v_b = "11 m/s" if pass_type == "Lofted Pass" else "15 m/s"
                st.metric("Ball Speed (v_ball)", v_b)
            with c9:
                decay = "50 yd" if pass_type == "Lofted Pass" else "30 yd"
                st.metric("Pass Decay λ", decay)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── TAB 2: Player Rankings ───────────────────────────────────
    with tab2:
        match_desc = (
            "multiple matches" if (multi_match and len(multi_options) > 1)
            else MATCH_OPTIONS.get(selected_match_id, "selected match")
        )
        st.subheader(f"Per-Player Spatial α — {match_desc}")
        st.caption(
            "Sorted by **Mean Spatial α** (ascending = most missed spatial opportunity)."
        )

        match_key = str(sorted(multi_options)) if multi_match else str(selected_match_id)
        agg = aggregate_all_passes(passes, merged, match_key=match_key)

        if agg.empty:
            st.warning("No data available for aggregation.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Players", f"{len(agg)}")
            with m2:
                st.metric("Passes", f"{int(agg['total_passes'].sum())}")
            with m3:
                st.metric("Mean α", f"{agg['mean_alpha'].mean():+.4f}")
            with m4:
                st.metric("Worst", agg.iloc[0]["player"].split()[-1])

            st.markdown("---")

            display_df = agg.rename(columns={
                "player": "Player", "team": "Team",
                "total_passes": "Passes",
                "mean_alpha": "Mean α",
                "min_alpha": "Min α",
                "pct_suboptimal": "% Sub-Opt",
                "mean_actual_xev": "Avg xEV",
                "mean_opt_xev": "Avg Opt xEV",
                "mean_pc": "Avg PC",
            })

            st.dataframe(
                display_df.style.format({
                    "Mean α": "{:+.4f}",
                    "Min α": "{:+.4f}",
                    "% Sub-Opt": "{:.1f}%",
                    "Avg xEV": "{:.4f}",
                    "Avg Opt xEV": "{:.4f}",
                    "Avg PC": "{:.0%}",
                }).background_gradient(
                    subset=["Mean α"],
                    cmap="RdYlGn", vmin=-0.06, vmax=0.005,
                ),
                use_container_width=True, height=600,
            )

            # ── Scouting Archetype Matrix (Scatter Plot) ──────────
            st.markdown("### Scouting Archetype Matrix")
            st.caption(
                "X = Mean α (decision efficiency) · Y = Total xEV (offensive volume). "
                "Players with ≥5 passes only. Colored by team."
            )
            min_passes = 5
            chart_df = agg.loc[agg["total_passes"] >= min_passes].copy()

            # Display name mapping (StatsBomb legal → recognizable)
            DISPLAY_NAMES: dict[str, str] = {
                "Lionel Andrés Messi Cuccittini": "Messi",
                "Kylian Mbappé Lottin": "Mbappé",
                "Antoine Griezmann": "Griezmann",
                "Ousmane Dembélé": "Dembélé",
                "Olivier Giroud": "Giroud",
                "Aurélien Djani Tchouaméni": "Tchouaméni",
                "Adrien Rabiot": "Rabiot",
                "Theo Bernard François Hernández": "T. Hernández",
                "Jules Olivier Koundé": "Koundé",
                "Raphaël Varane": "Varane",
                "Dayotchanculle Upamecano": "Upamecano",
                "Hugo Lloris": "Lloris",
                "Randal Kolo Muani": "Kolo Muani",
                "Marcus Thuram": "Thuram",
                "Kingsley Coman": "Coman",
                "Ángel Fabián Di María Hernández": "Di María",
                "Rodrigo Javier De Paul": "De Paul",
                "Alexis Mac Allister": "Mac Allister",
                "Enzo Jeremías Fernández": "E. Fernández",
                "Nicolás Alejandro Tagliafico": "Tagliafico",
                "Nahuel Molina Lucero": "Molina",
                "Nicolás Hernán Otamendi": "Otamendi",
                "Cristian Gabriel Romero": "Romero",
                "Emiliano Martínez": "E. Martínez",
                "Marcos Javier Acuña": "Acuña",
                "Leandro Daniel Paredes": "Paredes",
                "Gonzalo Ariel Montiel": "Montiel",
                "Julián Álvarez": "J. Álvarez",
                "Lautaro Javier Martínez": "L. Martínez",
                "Germán Alejandro Pezzella": "Pezzella",
            }

            if len(chart_df) > 0:
                fig_sc, ax_sc = plt.subplots(figsize=(12, 8))
                fig_sc.set_facecolor("#0e1117")
                ax_sc.set_facecolor("#0e1117")

                # Dynamic team colors
                teams = chart_df["team"].unique()
                team_palette = {}
                base_colors = [
                    "#00e5ff", "#ff6b6b", "#7c4dff", "#ffd54f",
                    "#69f0ae", "#ff8a65", "#e040fb", "#80deea",
                ]
                for i, t in enumerate(teams):
                    team_palette[t] = base_colors[i % len(base_colors)]

                # Quadrant lines
                median_xev = chart_df["sum_actual_xev"].median()
                ax_sc.axvline(x=0, color="white", linewidth=1, linestyle="--", alpha=0.4)
                ax_sc.axhline(y=median_xev, color="white", linewidth=1, linestyle="--", alpha=0.4)

                # Quadrant labels (large, faint background text)
                x_min = chart_df["mean_alpha"].min()
                x_max = chart_df["mean_alpha"].max()
                y_min = chart_df["sum_actual_xev"].min()
                y_max = chart_df["sum_actual_xev"].max()
                y_pad = (y_max - y_min) * 0.15

                quadrant_labels = [
                    (x_max * 0.5, y_max - y_pad * 0.3, "Elite\nProgressors", "#2ecc71"),
                    (x_min * 0.5, y_max - y_pad * 0.3, "High-Risk\nCreators", "#e67e22"),
                    (x_max * 0.5, y_min + y_pad * 0.3, "Safe\nRetainers", "#3498db"),
                    (x_min * 0.5, y_min + y_pad * 0.3, "Inefficient\nWasteful", "#e74c3c"),
                ]
                for qx, qy, qlabel, qcolor in quadrant_labels:
                    ax_sc.text(
                        qx, qy, qlabel,
                        fontsize=16, fontweight="bold", color=qcolor,
                        alpha=0.15, ha="center", va="center",
                    )

                # Plot dots by team
                for team_name in teams:
                    team_data = chart_df[chart_df["team"] == team_name]
                    ax_sc.scatter(
                        team_data["mean_alpha"],
                        team_data["sum_actual_xev"],
                        c=team_palette[team_name],
                        s=80, alpha=0.85, edgecolors="white",
                        linewidth=0.5, label=team_name, zorder=3,
                    )

                # Annotate with recognizable display names
                for _, row in chart_df.iterrows():
                    display = DISPLAY_NAMES.get(
                        row["player"], row["player"].split()[-1]
                    )
                    ax_sc.annotate(
                        display,
                        (row["mean_alpha"], row["sum_actual_xev"]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color="white", alpha=0.85,
                    )

                ax_sc.set_xlabel("Mean Spatial α (Decision Efficiency)", color="white", fontsize=11)
                ax_sc.set_ylabel("Total xEV Generated (Offensive Volume)", color="white", fontsize=11)
                ax_sc.tick_params(colors="white")
                ax_sc.legend(
                    facecolor="#1a1a2e", edgecolor="#333",
                    labelcolor="white", fontsize=9,
                    loc="upper left",
                )
                ax_sc.spines["top"].set_visible(False)
                ax_sc.spines["right"].set_visible(False)
                ax_sc.spines["bottom"].set_color("#444")
                ax_sc.spines["left"].set_color("#444")

                plt.tight_layout()
                st.pyplot(fig_sc, use_container_width=True)
                plt.close(fig_sc)

    # ── TAB 3: Animated Tactical Replay ──────────────────────────
    with tab3:
        st.subheader("Animated Tactical Replay")

        if use_pff and pff_passes is not None and not pff_passes.empty:
            # ── PFF 30-FPS TRACKING REPLAY ────────────────────────
            st.caption(
                "📡 **PFF 30fps Tracking Replay** — Watch 22 players sprint "
                "across the pitch in the 3 seconds leading up to the pass, "
                "ending with the xEV heatmap eruption."
            )

            if st.button("▶️ Generate 30fps Tracking Replay", type="primary",
                         use_container_width=True):
                pff_row = pff_passes.iloc[selected_idx]
                spatial = pff_pass_to_spatial(pff_row)

                if len(spatial["tm_pos"]) == 0 or len(spatial["def_pos"]) == 0:
                    st.warning("Insufficient player data for this pass.")
                else:
                    # Compute xEV surface
                    result = compute_continuous_alpha(
                        spatial["bc_pos"], spatial["tm_pos"], spatial["def_pos"],
                        spatial["actual_end"],
                        pass_type=spatial["pass_type"],
                        teammate_names=spatial["tm_names"],
                        defender_names=spatial["def_names"],
                        teammate_speeds=spatial["tm_speeds"],
                        defender_speeds=spatial["def_speeds"],
                        teammate_velocities=spatial.get("tm_velocities"),
                        attack_right=spatial.get("attack_right", True),
                        passer_velocity=spatial.get("passer_velocity"),
                    )

                    # Load 30fps tracking window
                    pff_game_id = SB_TO_PFF.get(selected_match_id)
                    pass_start_time = None

                    # Find startTime from raw PFF event data
                    if pff_game_id is not None:
                        import json as _json
                        from pff_loader import DATA_DIR as _PFF_DATA_DIR
                        ev_path = _PFF_DATA_DIR / "Event Data" / f"{pff_game_id}.json"
                        if ev_path.exists():
                            with open(ev_path) as _f:
                                raw_events = _json.load(_f)
                            # Match by passer name + minute
                            target_min = int(pff_row.get("minute", 0))
                            target_sec = int(pff_row.get("second", 0))
                            target_passer = pff_row.get("passer_name", "")
                            for ev in raw_events:
                                pe = ev.get("possessionEvents", {})
                                if not pe or pe.get("possessionEventType") != "PA":
                                    continue
                                gc = pe.get("gameClock", 0)
                                if gc // 60 == target_min and gc % 60 == target_sec:
                                    pass_start_time = ev.get("startTime")
                                    break
                            # Fallback: first matching passer
                            if pass_start_time is None:
                                for ev in raw_events:
                                    pe = ev.get("possessionEvents", {})
                                    if not pe:
                                        continue
                                    if pe.get("passerPlayerName") == target_passer:
                                        pass_start_time = ev.get("startTime")
                                        break

                    if pff_game_id is None or pass_start_time is None:
                        st.warning("Could not resolve tracking timestamp for this pass. "
                                   "Falling back to static replay.")
                        # Fall back to static animation
                        gif_bytes = create_tactical_animation(
                            spatial["bc_pos"], spatial["tm_pos"],
                            spatial["def_pos"], spatial["actual_end"],
                            result["spatial_xev"],
                            result["opt_x"], result["opt_y"],
                            pff_row["passer_name"],
                            pff_row["receiver_name"],
                            int(pff_row["minute"]), int(pff_row["second"]),
                        )
                    else:
                        with st.spinner("Loading 30fps tracking data (~45MB compressed)…"):
                            tracking_frames = load_tracking_window(
                                pff_game_id, pass_start_time,
                            )

                        if not tracking_frames:
                            st.warning("No tracking frames found for this window.")
                            gif_bytes = b""
                        else:
                            # Determine possession side ("team" holds the team
                            # name, e.g. "Argentina" — use the is_home boolean)
                            ge_home = bool(pff_row.get("is_home", False))

                            with st.spinner(
                                f"Rendering {len(tracking_frames)} tracking frames…"
                            ):
                                gif_bytes = create_tracking_replay(
                                    tracking_frames, result["spatial_xev"],
                                    result["opt_x"], result["opt_y"],
                                    pff_row["passer_name"],
                                    pff_row["receiver_name"],
                                    int(pff_row["minute"]),
                                    int(pff_row["second"]),
                                    is_home_possession=ge_home,
                                )

                    if gif_bytes:
                        b64 = base64.b64encode(gif_bytes).decode()
                        st.markdown(
                            f'<img src="data:image/gif;base64,{b64}" '
                            f'style="width:100%; border-radius:8px;">',
                            unsafe_allow_html=True,
                        )

                        st.markdown("---")
                        st.markdown("#### Spatial Arbitrage Insights")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Actual xEV",
                                      f"{result['actual_xev']:.4f}")
                        with c2:
                            st.metric("Optimal xEV",
                                      f"{result['opt_xev']:.4f}")
                        with c3:
                            st.metric("Spatial α",
                                      f"{result['alpha']:+.4f}")
                        with c4:
                            n_frames_raw = len(tracking_frames)
                            st.metric("Tracking Frames",
                                      f"{n_frames_raw} @ 30fps")

        elif multi_match and len(multi_options) > 1:
            st.info("Switch to single-match mode for tactical replay.")
        else:
            if st.button("▶️ Generate Replay", type="primary",
                         use_container_width=True):
                with st.spinner("Rendering spatial animation …"):
                    result_data = analyse_pass_continuous(
                        passes, merged, selected_idx
                    )
                    fig_unused, metrics, bc_pos, tm_pos, def_pos, actual_pos = result_data
                    plt.close(fig_unused)

                    if len(tm_pos) > 0 and len(def_pos) > 0:
                        pass_row = passes.iloc[selected_idx]
                        passer_name = str(pass_row.get("player", "Unknown"))
                        recipient_name = str(pass_row.get("pass_recipient",
                                                          "Unknown"))
                        minute = int(pass_row["minute"])
                        second = int(pass_row["second"])

                        gif_bytes = create_tactical_animation(
                            bc_pos, tm_pos, def_pos, actual_pos,
                            metrics["spatial_xev"],
                            metrics["opt_x"], metrics["opt_y"],
                            passer_name, recipient_name, minute, second,
                        )

                        b64 = base64.b64encode(gif_bytes).decode()
                        st.markdown(
                            f'<img src="data:image/gif;base64,{b64}" '
                            f'style="width:100%; border-radius:8px;">',
                            unsafe_allow_html=True,
                        )

                        st.markdown("---")
                        st.markdown("#### Spatial Arbitrage Insights")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Actual xEV",
                                      f"{metrics['actual_xev']:.4f}")
                        with c2:
                            st.metric("Optimal xEV",
                                      f"{metrics['opt_xev']:.4f}")
                        with c3:
                            st.metric("Spatial α",
                                      f"{metrics['alpha']:+.4f}")
                    else:
                        st.warning("Insufficient data for animation.")

    # ── TAB 4: Match Report ───────────────────────────────────────
    with tab4:
        st.subheader("📋 Auto-Generated Match Report")
        if use_pff:
            st.info("Match Report is available for StatsBomb data. Switch to StatsBomb 360.")
        elif multi_match and len(multi_options) > 1:
            st.info("Switch to single-match mode for the match report.")
        elif passes.empty:
            st.warning("No pass data available.")
        else:
            match_label = MATCH_OPTIONS.get(selected_match_id, "Selected Match")
            st.caption(f"**{match_label}** — Automated spatial analysis of all open-play passes")

            # Compute α for all passes
            report_alphas = []
            for idx in range(len(passes)):
                try:
                    row = passes.iloc[idx]
                    bc, tm, df_pos, _, _ = extract_spatial_for_pass(merged, row)
                    if len(tm) == 0 or len(df_pos) == 0:
                        continue
                    actual_end = np.array(row["pass_end_location"])
                    res = compute_continuous_alpha(bc, tm, df_pos, actual_end, pass_type="Ground Pass")
                    report_alphas.append({
                        "idx": idx,
                        "player": str(row.get("player", "Unknown")),
                        "minute": int(row.get("minute", 0)),
                        "second": int(row.get("second", 0)),
                        "recipient": str(row.get("pass_recipient", "Unknown")),
                        "alpha": res["alpha"],
                        "actual_xev": res["actual_xev"],
                        "opt_xev": res["opt_xev"],
                        "opt_x": res["opt_x"],
                        "opt_y": res["opt_y"],
                    })
                except Exception:
                    continue

            if not report_alphas:
                st.warning("Could not compute α for any passes.")
            else:
                report_df = pd.DataFrame(report_alphas).sort_values("alpha")

                # Summary metrics
                rm1, rm2, rm3, rm4 = st.columns(4)
                with rm1:
                    st.metric("Passes Analysed", f"{len(report_df)}")
                with rm2:
                    st.metric("Mean α", f"{report_df['alpha'].mean():+.5f}")
                with rm3:
                    st.metric("Median α", f"{report_df['alpha'].median():+.5f}")
                with rm4:
                    pct_neg = (report_df['alpha'] < -0.001).mean()
                    st.metric("Suboptimal %", f"{pct_neg:.0%}")

                st.markdown("---")

                # Top 5 worst and best passes
                col_worst, col_best = st.columns(2)

                with col_worst:
                    st.markdown("#### 🔴 Top 5 Most Suboptimal Passes")
                    worst5 = report_df.head(5)
                    for _, r in worst5.iterrows():
                        st.markdown(
                            f"**{r['minute']}:{r['second']:02d}** — "
                            f"{r['player'].split()[-1]} → {r['recipient'].split()[-1]}  \n"
                            f"α = `{r['alpha']:+.5f}` · "
                            f"Actual xEV: `{r['actual_xev']:.4f}` · "
                            f"Optimal: `{r['opt_xev']:.4f}` at ({r['opt_x']:.0f}, {r['opt_y']:.0f})"
                        )
                        st.markdown("---")

                with col_best:
                    st.markdown("#### 🟢 Top 5 Most Efficient Passes")
                    best5 = report_df.tail(5).iloc[::-1]
                    for _, r in best5.iterrows():
                        st.markdown(
                            f"**{r['minute']}:{r['second']:02d}** — "
                            f"{r['player'].split()[-1]} → {r['recipient'].split()[-1]}  \n"
                            f"α = `{r['alpha']:+.5f}` · "
                            f"Actual xEV: `{r['actual_xev']:.4f}` · "
                            f"Optimal: `{r['opt_xev']:.4f}` at ({r['opt_x']:.0f}, {r['opt_y']:.0f})"
                        )
                        st.markdown("---")

                # α distribution histogram
                st.markdown("#### α Distribution")
                fig_hist, ax_hist = plt.subplots(figsize=(10, 3))
                fig_hist.patch.set_facecolor("#0e1117")
                ax_hist.set_facecolor("#0e1117")
                ax_hist.hist(report_df["alpha"], bins=40, color="#00e5ff", alpha=0.8, edgecolor="none")
                ax_hist.axvline(x=0, color="white", linewidth=1, linestyle="--", alpha=0.5)
                ax_hist.axvline(x=report_df["alpha"].mean(), color="#ff4444", linewidth=2,
                               label=f"Mean: {report_df['alpha'].mean():+.5f}")
                ax_hist.set_xlabel("Spatial α", color="white")
                ax_hist.set_ylabel("Count", color="white")
                ax_hist.tick_params(colors="white")
                ax_hist.legend(facecolor="#1a1a2e", edgecolor="#00e5ff", labelcolor="white")
                st.pyplot(fig_hist, use_container_width=True)
                plt.close(fig_hist)

                # Team-level breakdown
                st.markdown("#### Team-Level Summary")
                team_agg = report_df.groupby(
                    report_df["player"].apply(
                        lambda p: passes.loc[passes["player"] == p, "team"].iloc[0]
                        if len(passes.loc[passes["player"] == p]) > 0 else "Unknown"
                    )
                ).agg(
                    passes_count=("alpha", "count"),
                    mean_alpha=("alpha", "mean"),
                    min_alpha=("alpha", "min"),
                    max_alpha=("alpha", "max"),
                ).round(5)
                st.dataframe(team_agg, use_container_width=True)

    # ── TAB 5: Tracking Deep Dive ────────────────────────────────
    with tab5:
        st.subheader("📡 PFF Tracking Deep Dive")
        st.caption("30fps tracking data → Off-Ball Sprint Quality + Pass Availability")

        # SB_TO_PFF is imported at module level from pff_loader

        pff_game_id = SB_TO_PFF.get(selected_match_id)
        if pff_game_id is None:
            st.info("Tracking data available for knockout matches only. "
                    "Select a QF/Semi/Final match.")
        else:
            with st.spinner("Computing tracking analytics (cached after first run…)"):
                try:
                    tracking = analyse_tracking_match(pff_game_id)
                except Exception as e:
                    st.error(f"Failed to load tracking data: {e}")
                    tracking = None

            if tracking:
                sprints = tracking.get("sprints", pd.DataFrame())
                availability = tracking.get("availability", pd.DataFrame())

                # ── Off-Ball Sprint Rankings ─────────────────────────
                st.markdown("#### 🏃 Off-Ball Sprint Quality")
                st.caption(
                    "Sprint distance in the 2.0s window before receiving a pass. "
                    "Identifies who **creates** space, not just who finds it."
                )

                if not sprints.empty:
                    sm1, sm2, sm3 = st.columns(3)
                    with sm1:
                        st.metric("Players Tracked", f"{len(sprints)}")
                    with sm2:
                        st.metric("Top Sprinter",
                                  sprints.iloc[0]["player"].split()[-1])
                    with sm3:
                        st.metric("Best Avg Sprint",
                                  f"{sprints.iloc[0]['mean_sprint_dist']:.1f} m")

                    # Sprint bar chart
                    top_n = min(15, len(sprints))
                    fig_sp, ax_sp = plt.subplots(figsize=(10, 4))
                    fig_sp.patch.set_facecolor("#0e1117")
                    ax_sp.set_facecolor("#0e1117")
                    names = [n.split()[-1] for n in sprints.head(top_n)["player"]]
                    vals = sprints.head(top_n)["mean_sprint_dist"].values
                    colors = ["#00e5ff" if v > 5 else "#64b5f6" for v in vals]
                    ax_sp.barh(range(top_n), vals, color=colors, edgecolor="none")
                    ax_sp.set_yticks(range(top_n))
                    ax_sp.set_yticklabels(names, color="white", fontsize=10)
                    ax_sp.set_xlabel("Mean Sprint Distance (m)", color="white")
                    ax_sp.invert_yaxis()
                    ax_sp.tick_params(colors="white")
                    ax_sp.spines["top"].set_visible(False)
                    ax_sp.spines["right"].set_visible(False)
                    ax_sp.spines["left"].set_color("#333")
                    ax_sp.spines["bottom"].set_color("#333")
                    st.pyplot(fig_sp, use_container_width=True)
                    plt.close(fig_sp)

                    # Full table
                    with st.expander("Full Sprint Data", expanded=False):
                        st.dataframe(
                            sprints.rename(columns={
                                "player": "Player", "team": "Team",
                                "receptions": "Receptions",
                                "mean_sprint_dist": "Avg Sprint (m)",
                                "max_sprint_dist": "Max Sprint (m)",
                                "mean_max_speed": "Avg Peak Speed (m/s)",
                                "mean_sprint_pct": "Sprint % of Window",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    st.warning("No sprint data available.")

                st.markdown("---")

                # ── Pass Availability (Body Orientation) ───────────
                st.markdown("#### 🧠 Pass Availability (Body Orientation Proxy)")
                st.caption(
                    "Velocity-vector body orientation: a player running **away** "
                    "from the ball cannot receive. This solves the omnidirectional-player flaw."
                )

                if not availability.empty:
                    home_avail = availability[availability["team"] == "home"]
                    away_avail = availability[availability["team"] == "away"]

                    am1, am2, am3, am4 = st.columns(4)
                    with am1:
                        st.metric("Home Avg Available",
                                  f"{home_avail['n_available'].mean():.1f} / {home_avail['n_outfield'].mean():.0f}")
                    with am2:
                        st.metric("Home Availability %",
                                  f"{home_avail['availability_pct'].mean():.1f}%")
                    with am3:
                        st.metric("Away Avg Available",
                                  f"{away_avail['n_available'].mean():.1f} / {away_avail['n_outfield'].mean():.0f}")
                    with am4:
                        st.metric("Away Availability %",
                                  f"{away_avail['availability_pct'].mean():.1f}%")

                    # Availability timeline (per-minute)
                    fig_av, ax_av = plt.subplots(figsize=(10, 3))
                    fig_av.patch.set_facecolor("#0e1117")
                    ax_av.set_facecolor("#0e1117")

                    for team, color, label_suffix in [
                        ("home", "#00e5ff", "Home"),
                        ("away", "#ff6b6b", "Away"),
                    ]:
                        t_data = availability[availability["team"] == team]
                        per_min = t_data.groupby("minute")["availability_pct"].mean()
                        ax_av.plot(per_min.index, per_min.values,
                                  color=color, linewidth=1.5, alpha=0.8,
                                  label=f"{label_suffix} availability")

                    ax_av.set_xlabel("Minute", color="white")
                    ax_av.set_ylabel("Availability %", color="white")
                    ax_av.tick_params(colors="white")
                    ax_av.legend(facecolor="#1a1a2e", edgecolor="#333",
                                labelcolor="white")
                    ax_av.spines["top"].set_visible(False)
                    ax_av.spines["right"].set_visible(False)
                    ax_av.spines["left"].set_color("#333")
                    ax_av.spines["bottom"].set_color("#333")
                    st.pyplot(fig_av, use_container_width=True)
                    plt.close(fig_av)

                    # Facing angle distribution
                    fig_fa, ax_fa = plt.subplots(figsize=(10, 3))
                    fig_fa.patch.set_facecolor("#0e1117")
                    ax_fa.set_facecolor("#0e1117")
                    ax_fa.hist(availability["mean_facing_angle"], bins=30,
                              color="#7c4dff", alpha=0.8, edgecolor="none")
                    ax_fa.axvline(x=90, color="#ff4444", linewidth=2,
                                 linestyle="--", label="90° (perpendicular)")
                    ax_fa.set_xlabel("Mean Facing Angle (°)", color="white")
                    ax_fa.set_ylabel("Count", color="white")
                    ax_fa.tick_params(colors="white")
                    ax_fa.legend(facecolor="#1a1a2e", edgecolor="#333",
                                labelcolor="white")
                    ax_fa.spines["top"].set_visible(False)
                    ax_fa.spines["right"].set_visible(False)
                    ax_fa.spines["left"].set_color("#333")
                    ax_fa.spines["bottom"].set_color("#333")
                    st.pyplot(fig_fa, use_container_width=True)
                    plt.close(fig_fa)
                else:
                    st.warning("No availability data.")


if __name__ == "__main__":
    main()
