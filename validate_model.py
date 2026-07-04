"""
Pitch Control Model Validation
===================================================
Quantitative validation of the Cognitive Alpha model.

Features:
  • Possession-ID look-ahead (no cross-possession contamination)
  • Distance-controlled quintile analysis
  • Naive baseline comparison (nearest teammate heuristic)
  • Parallel computation via joblib

Validation axes:
  1. Does α predict actual pass success?
  2. Does α predict downstream outcomes (shots, turnovers)?
  3. Does α outperform naive "nearest teammate" baseline?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsbombpy import sb
from joblib import Parallel, delayed
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from pitch_control import compute_continuous_alpha
from pff_loader import extract_pff_passes, pff_pass_to_spatial, PFF_KEY_MATCHES

OUTPUT_DIR = Path(__file__).parent
COMPETITION_ID = 43
SEASON_ID = 106


# ---- 0. Methodology helpers (unit-tested in tests/test_model.py) ----
def infer_intended_target(
    end_loc: np.ndarray, tm_pos: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """
    Proxy the INTENDED target of a failed pass.

    StatsBomb's pass_end_location for a failed pass is where the ball was
    intercepted or ran out -- mechanically a low-xEV spot. Evaluating the
    decision there leaks the outcome into alpha ("failed passes look like
    bad decisions" partly by construction). The standard proxy is the
    teammate closest to the recorded end location.

    Returns (target_pos, is_proxy).
    """
    if len(tm_pos) == 0:
        return end_loc, False
    dists = np.linalg.norm(tm_pos - end_loc, axis=1)
    return tm_pos[int(np.argmin(dists))].astype(float), True


def disagreement_cohorts(
    df: pd.DataFrame,
    radius: float = 8.0,
    min_separation: float = 8.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Outcome-based model-vs-baseline comparison, restricted to passes where
    the model target and the nearest-teammate target actually DISAGREE
    (further than min_separation apart).

    Rationale: the old comparison split on opt_xev > nearest_teammate_xev,
    but opt_xev is the argmax of the same surface nearest_teammate_xev is
    read from, so opt_xev >= nearest_teammate_xev holds by construction and
    the "baseline" cohort contained only exact ties.

    Returns (went_model, went_baseline): passes whose actual end point lies
    within `radius` yards of exactly one of the two targets. Downstream
    outcome rates (shots, turnovers) of the two cohorts give a fair,
    non-circular baseline comparison.
    """
    d_targets = np.sqrt(
        (df["opt_x"] - df["nearest_x"]) ** 2
        + (df["opt_y"] - df["nearest_y"]) ** 2
    )
    d_opt = np.sqrt(
        (df["actual_end_x"] - df["opt_x"]) ** 2
        + (df["actual_end_y"] - df["opt_y"]) ** 2
    )
    d_near = np.sqrt(
        (df["actual_end_x"] - df["nearest_x"]) ** 2
        + (df["actual_end_y"] - df["nearest_y"]) ** 2
    )
    disagree = d_targets > min_separation
    went_model = disagree & (d_opt <= radius) & (d_near > radius)
    went_baseline = disagree & (d_near <= radius) & (d_opt > radius)
    return df[went_model], df[went_baseline]


# ---- 1. Compute α for all passes in one match ----
def compute_match_alphas(match_id: int) -> pd.DataFrame:
    """Compute α for all open-play passes in a match (with possession-aware look-ahead)."""
    events = sb.events(match_id=match_id)
    frames = sb.frames(match_id=match_id, fmt="dataframe")

    # All passes (successful + failed) for analysis
    mask = (events["type"] == "Pass") & (events["play_pattern"] == "Regular Play")
    passes = events.loc[mask].copy()
    merged = passes.merge(frames, on="id", how="inner", suffixes=("_event", "_frame"))

    results = []
    processed_ids = set()

    for _, pass_row in passes.iterrows():
        pass_id = pass_row["id"]
        if pass_id in processed_ids:
            continue
        processed_ids.add(pass_id)

        try:
            frame_rows = merged.loc[merged["id"] == pass_id]
            if len(frame_rows) == 0:
                continue

            # Ball carrier
            actor_rows = frame_rows.loc[frame_rows["actor"] == True]
            loc = pass_row.get("location")
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue

            if len(actor_rows) > 0:
                actor_loc = actor_rows.iloc[0]["location_frame"]
                bc_pos = np.array(actor_loc, dtype=float)
            else:
                bc_pos = np.array(loc, dtype=float)

            # Teammates
            tm_rows = frame_rows.loc[
                (frame_rows["teammate"] == True) & (frame_rows["actor"] == False)
            ]
            if len(tm_rows) == 0:
                continue
            tm_pos = np.array(tm_rows["location_frame"].tolist(), dtype=float)

            # Defenders
            def_rows = frame_rows.loc[frame_rows["teammate"] == False]
            if len(def_rows) == 0:
                continue
            def_pos = np.array(def_rows["location_frame"].tolist(), dtype=float)

            # Pass end location
            end_loc = pass_row.get("pass_end_location")
            if not isinstance(end_loc, (list, tuple)) or len(end_loc) < 2:
                continue
            actual_end = np.array(end_loc, dtype=float)

            # Outcome (needed now: failed passes get a proxied target)
            pass_outcome = pass_row.get("pass_outcome")
            is_successful = pd.isna(pass_outcome)

            # De-leak: evaluate failed passes at the intended target, not
            # at the interception point (see infer_intended_target).
            end_is_proxy = False
            if not is_successful:
                actual_end, end_is_proxy = infer_intended_target(actual_end, tm_pos)

            # Compute α
            result = compute_continuous_alpha(
                bc_pos, tm_pos, def_pos, actual_end, pass_type="Ground Pass",
            )

            # === Nearest teammate baseline (concordance-ready) ===
            # Store both xEV and position for concordance analysis.
            if len(tm_pos) > 0:
                dists = np.linalg.norm(tm_pos - bc_pos, axis=1)
                nearest_idx = np.argmin(dists)
                nearest_pos = tm_pos[nearest_idx]
                col = int(np.clip(nearest_pos[0] - 0.5, 0, 119))
                row = int(np.clip(nearest_pos[1] - 0.5, 0, 79))
                nearest_xev = float(result["spatial_xev"][row, col])
                nearest_x, nearest_y = float(nearest_pos[0]), float(nearest_pos[1])
            else:
                nearest_xev = 0.0
                nearest_x, nearest_y = 0.0, 0.0

            # === Possession-ID look-ahead ===
            passer_team = pass_row.get("team")
            possession_id = pass_row.get("possession")

            # Look ahead within SAME possession chain only.
            # Sort chronologically -- DataFrame row order is not a contract.
            if possession_id is not None:
                same_poss = events[
                    (events["possession"] == possession_id)
                    & (events["team"] == passer_team)
                ]
                if "index" in same_poss.columns:
                    same_poss = same_poss.sort_values("index", kind="stable")
                else:
                    same_poss = same_poss.sort_values(
                        ["period", "minute", "second"], kind="stable"
                    )
                poss_ids = same_poss["id"].tolist()
                if pass_id in poss_ids:
                    pos = poss_ids.index(pass_id)
                    future_in_poss = same_poss.iloc[pos + 1:]
                    shot_in_poss = any(future_in_poss["type"] == "Shot")
                else:
                    shot_in_poss = False

                # Possession lost = did the NEXT possession belong to other team?
                next_poss_events = events[events["possession"] == possession_id + 1]
                if len(next_poss_events) > 0:
                    next_team = next_poss_events.iloc[0].get("team")
                    possession_lost = (next_team != passer_team)
                else:
                    possession_lost = False
            else:
                shot_in_poss = False
                possession_lost = False

            pass_distance = float(np.linalg.norm(actual_end - bc_pos))

            results.append({
                "match_id": match_id,
                "pass_id": pass_id,
                "passer": pass_row.get("player", "Unknown"),
                "team": passer_team,
                "minute": pass_row.get("minute", 0),
                "possession_id": possession_id,
                "alpha": result["alpha"],
                "opt_xev": result["opt_xev"],
                "actual_xev": result["actual_xev"],
                "nearest_teammate_xev": nearest_xev,
                "opt_x": result["opt_x"],
                "opt_y": result["opt_y"],
                "bc_x": bc_pos[0],
                "actual_end_x": float(actual_end[0]),
                "actual_end_y": float(actual_end[1]),
                "nearest_x": nearest_x,
                "nearest_y": nearest_y,
                "pass_distance": pass_distance,
                "is_successful": is_successful,
                "end_is_proxy": end_is_proxy,
                "shot_in_possession": shot_in_poss,
                "possession_lost": possession_lost,
                "n_defenders": len(def_pos),
                "n_teammates": len(tm_pos),
                "data_source": "StatsBomb",
            })

        except Exception:
            continue

    return pd.DataFrame(results)


# ---- 1b. Compute α for all passes using PFF tracking data ----
def compute_pff_match_alphas(game_id: int) -> pd.DataFrame:
    """
    Compute α using PFF Event + Tracking data.

    Feeds real-time per-player speeds AND velocity vectors
    (body orientation) into compute_continuous_alpha.

    Includes possession-chain look-ahead: scans the next 10 raw events
    to detect shots (SH) and team changes (possession lost).
    """
    import json
    from pff_loader import DATA_DIR

    passes_df = extract_pff_passes(game_id, completed_only=False, with_velocities=True)
    if passes_df.empty:
        return pd.DataFrame()

    # Load the full raw event sequence for possession chain look-ahead
    event_path = DATA_DIR / "Event Data" / f"{game_id}.json"
    raw_events: list[dict] = []
    if event_path.exists():
        with open(event_path) as f:
            raw_events = json.load(f)

    results = []
    for idx, row in passes_df.iterrows():
        try:
            spatial = pff_pass_to_spatial(row)
            if len(spatial["tm_pos"]) == 0 or len(spatial["def_pos"]) == 0:
                continue

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

            # === True baseline: nearest teammate (concordance-ready) ===
            nearest_xev = 0.0
            nearest_x_pff, nearest_y_pff = 0.0, 0.0
            if len(spatial["tm_pos"]) > 0:
                dists = np.linalg.norm(
                    spatial["tm_pos"] - spatial["bc_pos"], axis=1
                )
                nearest_pos = spatial["tm_pos"][np.argmin(dists)]
                col = int(np.clip(nearest_pos[0] - 0.5, 0, 119))
                r = int(np.clip(nearest_pos[1] - 0.5, 0, 79))
                nearest_xev = float(result["spatial_xev"][r, col])
                nearest_x_pff = float(nearest_pos[0])
                nearest_y_pff = float(nearest_pos[1])

            is_successful = row.get("pass_outcome", "") == "C"
            is_home = row.get("team", "") == raw_events[0].get("gameEvents", {}).get("teamName", "") if raw_events else True

            # === PFF Possession Chain Look-ahead ===
            # Find this pass's index in the raw event list by gameClock
            # proximity AND passer name (gameClock alone is ambiguous when
            # two passes share the same second), capped at 2s tolerance.
            shot_in_poss = False
            poss_lost = not is_successful  # default: failed pass = lost
            pass_clock = row.get("minute", 0) * 60 + row.get("second", 0)
            target_passer = str(row.get("passer_name") or "")

            best_raw_idx = None
            best_diff = float("inf")
            for ri, re in enumerate(raw_events):
                pe = re.get("possessionEvents", {})
                if pe and pe.get("possessionEventType") == "PA":
                    if target_passer and (pe.get("passerPlayerName") or "") != target_passer:
                        continue
                    gc = pe.get("gameClock", -1)
                    if gc >= 0:
                        diff = abs(gc - pass_clock)
                        if diff < best_diff:
                            best_diff = diff
                            best_raw_idx = ri
            if best_diff > 2:
                best_raw_idx = None

            if best_raw_idx is not None and is_successful:
                # Scan the next 10 events after this pass
                passer_is_home = raw_events[best_raw_idx].get("gameEvents", {}).get("homeTeam", None)
                lookahead = raw_events[best_raw_idx + 1: best_raw_idx + 11]
                for future_event in lookahead:
                    fe_ge = future_event.get("gameEvents", {})
                    fe_pe = future_event.get("possessionEvents", {})
                    fe_type = fe_pe.get("possessionEventType", "") if fe_pe else ""
                    fe_is_home = fe_ge.get("homeTeam", None)

                    # Team changed → possession lost
                    if fe_is_home is not None and fe_is_home != passer_is_home:
                        poss_lost = True
                        break

                    # Shot event by same team
                    if fe_type == "SH" and fe_is_home == passer_is_home:
                        shot_in_poss = True
                        break

            # === Expert label: resolve better_option player position ===
            better_opt_name = row.get("better_option_name")
            better_opt_x, better_opt_y = np.nan, np.nan
            if better_opt_name and len(spatial["tm_pos"]) > 0:
                # Find the teammate whose name matches
                tm_names = spatial.get("tm_names", [])
                for ti, tname in enumerate(tm_names):
                    if better_opt_name in tname or tname in better_opt_name:
                        better_opt_x = float(spatial["tm_pos"][ti, 0])
                        better_opt_y = float(spatial["tm_pos"][ti, 1])
                        break

            results.append({
                "match_id": game_id,
                "pass_id": f"pff_{game_id}_{idx}",
                "passer": row.get("passer_name", "Unknown"),
                "team": row.get("team", "Unknown"),
                "minute": row.get("minute", 0),
                "possession_id": 0,
                "alpha": result["alpha"],
                "opt_xev": result["opt_xev"],
                "actual_xev": result["actual_xev"],
                "nearest_teammate_xev": nearest_xev,
                "opt_x": result["opt_x"],
                "opt_y": result["opt_y"],
                "bc_x": spatial["bc_pos"][0],
                "actual_end_x": float(spatial["actual_end"][0]),
                "actual_end_y": float(spatial["actual_end"][1]),
                "nearest_x": nearest_x_pff,
                "nearest_y": nearest_y_pff,
                "pass_distance": float(np.linalg.norm(
                    spatial["actual_end"] - spatial["bc_pos"]
                )),
                "is_successful": is_successful,
                # failed PFF passes are already evaluated at the annotated
                # intended target (extract_pff_passes falls back to
                # targetPlayerName when there is no receiver), not at the
                # interception point
                "end_is_proxy": not is_successful,
                "shot_in_possession": shot_in_poss,
                "possession_lost": poss_lost,
                "n_defenders": len(spatial["def_pos"]),
                "n_teammates": len(spatial["tm_pos"]),
                "lines_broken": int(row.get("lines_broken", 0)),
                "better_option_name": better_opt_name,
                "better_option_x": better_opt_x,
                "better_option_y": better_opt_y,
                "data_source": "PFF",
            })

        except Exception:
            continue

    return pd.DataFrame(results)



# ---- 2. Parallel aggregation across matches ----
def compute_validation_data(
    n_matches: int = 10,
    data_source: str = "statsbomb",
) -> pd.DataFrame:
    """
    Compute α across multiple matches.

    data_source: "statsbomb" (freeze frames, no speeds)
                 "pff" (30fps tracking, speeds + velocity vectors)
    """
    suffix = "_pff" if data_source == "pff" else ""
    # n_matches is part of the cache key -- a cache built for 3 matches
    # must not silently satisfy a 10-match run
    cache_path = OUTPUT_DIR / f"validation_data{suffix}_{n_matches}m.parquet"
    if cache_path.exists():
        print(f"  Loading cached {data_source.upper()} validation data")
        return pd.read_parquet(cache_path)

    if data_source == "pff":
        # Use PFF key matches (knockout rounds with tracking data)
        pff_ids = list(PFF_KEY_MATCHES.keys())[:n_matches]
        print(f"  Computing α for {len(pff_ids)} PFF matches (with velocities)...")
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(compute_pff_match_alphas)(gid) for gid in pff_ids
        )
    else:
        matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
        match_ids = matches["match_id"].head(n_matches).tolist()
        print(f"  Computing α for {n_matches} StatsBomb matches in parallel...")
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(compute_match_alphas)(mid) for mid in match_ids
        )

    valid = [df for df in results if len(df) > 0]
    result = pd.concat(valid, ignore_index=True)
    result.to_parquet(cache_path, index=False)
    print(f"  Cached {len(result)} validated passes ({data_source.upper()})")
    return result


# ---- 3. Visualization (with all validation axes) ----
def plot_validation(df: pd.DataFrame, save_path: str | None = None):
    """6-panel validation dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes.flat:
        ax.set_facecolor("#0e1117")

    # === Panel 1: α distribution by outcome ===
    ax1 = axes[0, 0]
    success = df[df["is_successful"]]["alpha"]
    failed = df[~df["is_successful"]]["alpha"]
    ax1.hist(success, bins=50, alpha=0.7, color="#00e5ff",
             label=f"Successful (n={len(success)})", density=True)
    ax1.hist(failed, bins=50, alpha=0.7, color="#ff4444",
             label=f"Failed (n={len(failed)})", density=True)
    ax1.axvline(x=0, color="white", linewidth=1, linestyle="--", alpha=0.5)
    ax1.set_title("α Distribution by Outcome", color="white", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Spatial α", color="white")
    ax1.set_ylabel("Density", color="white")
    ax1.tick_params(colors="white"); ax1.legend(fontsize=9)

    # === Panel 2: α quintile vs outcomes (raw) ===
    ax2 = axes[0, 1]
    df["alpha_q"] = pd.qcut(df["alpha"], q=5, labels=["Q1\nworst", "Q2", "Q3", "Q4", "Q5\nbest"],
                            duplicates="drop")
    q_stats = df.groupby("alpha_q", observed=False).agg(
        success=("is_successful", "mean"),
        shot=("shot_in_possession", "mean"),
        turnover=("possession_lost", "mean"),
        count=("alpha", "count"),
    )
    x = range(len(q_stats))
    w = 0.25
    ax2.bar([i - w for i in x], q_stats["success"], w, color="#00e5ff", label="Pass Success", alpha=0.8)
    ax2.bar(x, q_stats["shot"], w, color="#ffd700", label="Shot in Poss", alpha=0.8)
    ax2.bar([i + w for i in x], q_stats["turnover"], w, color="#ff4444", label="Poss Lost", alpha=0.8)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(q_stats.index, color="white")
    ax2.set_title("Outcomes by α Quintile (Raw)", color="white", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Rate", color="white")
    ax2.tick_params(colors="white"); ax2.legend(fontsize=8)

    # === Panel 3: Distance-controlled quintiles ===
    ax3 = axes[0, 2]
    # Bin passes by distance first, then compute α quintiles within each bin
    df["dist_bin"] = pd.cut(df["pass_distance"], bins=[0, 10, 20, 35, 200],
                           labels=["Short\n(<10yd)", "Medium\n(10-20)", "Long\n(20-35)", "Very Long\n(35+)"])
    dist_ctrl = []
    for dist_label, group in df.groupby("dist_bin", observed=False):
        if len(group) < 20:
            continue
        try:
            group = group.copy()
            group["alpha_q_dist"] = pd.qcut(group["alpha"], q=3,
                                            labels=["Low α", "Mid α", "High α"],
                                            duplicates="drop")
            for aq, sub in group.groupby("alpha_q_dist", observed=False):
                dist_ctrl.append({
                    "distance": dist_label,
                    "alpha_tercile": aq,
                    "success_rate": sub["is_successful"].mean(),
                    "shot_rate": sub["shot_in_possession"].mean(),
                    "n": len(sub),
                })
        except:
            continue

    if dist_ctrl:
        dc = pd.DataFrame(dist_ctrl)
        for i, dist_lab in enumerate(dc["distance"].unique()):
            sub = dc[dc["distance"] == dist_lab]
            colors = ["#ff4444", "#ffd700", "#00e5ff"]
            for j, (_, row) in enumerate(sub.iterrows()):
                ax3.bar(i + (j - 1) * 0.25, row["success_rate"], 0.2,
                       color=colors[j], alpha=0.8,
                       label=row["alpha_tercile"] if i == 0 else "")
        ax3.set_xticks(range(len(dc["distance"].unique())))
        ax3.set_xticklabels(dc["distance"].unique(), color="white", fontsize=9)
    ax3.set_title("Success Rate (Distance-Controlled)", color="white", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Pass Success Rate", color="white")
    ax3.tick_params(colors="white"); ax3.legend(fontsize=8)

    # === Panel 4: Model vs Nearest-TM baseline (disagreements only) ===
    # Fair comparison: only passes where the two targets are >8 yd apart
    # and the player's actual choice matches exactly one of them.
    # (The old version assigned overlapping passes to the model first,
    # which inflated the model cohort whenever the targets coincided.)
    ax4 = axes[1, 0]
    CONCORDANCE_R = 8.0  # yards — radius for "human passed to this target"
    EXPERT_R = 12.0      # yards — radius for AI-vs-expert agreement

    has_coords = all(c in df.columns for c in ["actual_end_x", "actual_end_y", "nearest_x", "nearest_y"])
    if has_coords:
        conc_model, conc_baseline = disagreement_cohorts(
            df, radius=CONCORDANCE_R, min_separation=CONCORDANCE_R,
        )
    else:
        conc_model = pd.DataFrame()
        conc_baseline = pd.DataFrame()

    # --- Outcomes when the player followed the model vs the naive heuristic ---
    has_lines = "lines_broken" in df.columns
    lines_model = conc_model["lines_broken"].mean() if has_lines and len(conc_model) > 0 else 0
    lines_baseline = conc_baseline["lines_broken"].mean() if has_lines and len(conc_baseline) > 0 else 0
    succ_model = conc_model["is_successful"].mean() if len(conc_model) > 0 else 0
    succ_baseline = conc_baseline["is_successful"].mean() if len(conc_baseline) > 0 else 0
    shot_model = conc_model["shot_in_possession"].mean() if len(conc_model) > 0 else 0
    shot_baseline = conc_baseline["shot_in_possession"].mean() if len(conc_baseline) > 0 else 0

    x_b = [0, 1]
    labels_conc = [
        f"Chose model\ntarget\n(n={len(conc_model)})",
        f"Chose nearest\nteammate\n(n={len(conc_baseline)})",
    ]
    w = 0.22
    ax4.bar([x - w for x in x_b], [succ_model, succ_baseline], w,
            color="#00e5ff", label="Pass Success Rate", alpha=0.8)
    ax4.bar(x_b, [shot_model, shot_baseline], w,
            color="#ff8c00", label="Shot in Possession", alpha=0.9)
    ax4.bar([x + w for x in x_b], [lines_model, lines_baseline], w,
            color="#ffd700", label="Avg Lines Broken", alpha=0.9)
    ax4.set_xticks(x_b)
    ax4.set_xticklabels(labels_conc, color="white", fontsize=9)
    ax4.set_title(
        f"Model vs Nearest-TM — disagreements only (sep > {CONCORDANCE_R:.0f} yd)",
        color="white", fontsize=11, fontweight="bold",
    )
    ax4.set_ylabel("Value", color="white")
    ax4.tick_params(colors="white")
    ax4.legend(fontsize=8)

    # --- Expert Scout Agreement % ---
    expert_agreement_pct = 0.0
    n_expert_passes = 0
    if "better_option_x" in df.columns:
        expert_df = df.dropna(subset=["better_option_x", "better_option_y"])
        n_expert_passes = len(expert_df)
        if n_expert_passes > 0:
            expert_dist = np.sqrt(
                (expert_df["opt_x"] - expert_df["better_option_x"]) ** 2
                + (expert_df["opt_y"] - expert_df["better_option_y"]) ** 2
            )
            n_agree = (expert_dist <= EXPERT_R).sum()
            expert_agreement_pct = n_agree / n_expert_passes * 100
            print(f"\n  AI Agreement with PFF Expert Scouts: "
                  f"{expert_agreement_pct:.1f}% ({n_agree}/{n_expert_passes} passes, "
                  f"r ≤ {EXPERT_R:.0f} yd)")

    # === Panel 5: α vs pass distance scatter ===
    ax5 = axes[1, 1]
    sc = ax5.scatter(df["pass_distance"], df["alpha"],
                    c=df["is_successful"].astype(float), cmap="RdYlGn", alpha=0.3, s=8)
    ax5.axhline(y=0, color="white", linewidth=1, linestyle="--", alpha=0.5)
    ax5.set_title("α vs Pass Distance", color="white", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Distance (yards)", color="white")
    ax5.set_ylabel("α", color="white")
    ax5.tick_params(colors="white")
    cbar = plt.colorbar(sc, ax=ax5); cbar.set_label("Success", color="white")
    cbar.ax.tick_params(colors="white")

    # === Panel 6: Summary statistics ===
    ax6 = axes[1, 2]; ax6.axis("off")
    corr = df["alpha"].corr(df["is_successful"].astype(float))
    q1, q5 = q_stats.iloc[0], q_stats.iloc[-1]

    stats_text = (
        f"VALIDATION RESULTS\n{'='*45}\n\n"
        f"Passes analysed:        {len(df):,}\n"
        f"Matches:                {df['match_id'].nunique()}\n"
        f"  Successful:           {len(success):,} ({len(success)/len(df):.1%})\n"
        f"  Failed:               {len(failed):,}\n\n"
        f"Mean α (successful):    {success.mean():+.6f}\n"
        f"Mean α (failed):        {failed.mean():+.6f}\n"
        f"Correlation (α↔succ):   r = {corr:.4f}\n\n"
        f"Q1 (worst α):\n"
        f"  Pass success: {q1['success']:.1%}\n"
        f"  Shot rate:    {q1['shot']:.1%}\n"
        f"  Turnover:     {q1['turnover']:.1%}\n\n"
        f"Q5 (best α):\n"
        f"  Pass success: {q5['success']:.1%}\n"
        f"  Shot rate:    {q5['shot']:.1%}\n"
        f"  Turnover:     {q5['turnover']:.1%}\n\n"
        f"Disagreement cohorts (sep > 8 yd):\n"
        f"  Chose model target (n={len(conc_model)}):\n"
        f"    success {succ_model:.1%} | shot {shot_model:.1%}\n"
        f"  Chose nearest TM (n={len(conc_baseline)}):\n"
        f"    success {succ_baseline:.1%} | shot {shot_baseline:.1%}\n\n"
        f"Expert Scout Agreement:\n"
        f"  {expert_agreement_pct:.1f}% ({n_expert_passes} tagged)\n"
    )
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=9, color="white", va="top", ha="left",
            fontfamily="monospace",
            bbox=dict(facecolor="#1a1a2e", edgecolor="#00e5ff", alpha=0.9))

    fig.suptitle("Cognitive Alpha — Model Validation",
                color="white", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor="#0e1117", bbox_inches="tight")
        print(f"  Saved validation plot to {save_path}")
    return fig


# ---- Main ----
def main():
    import sys
    data_source = "pff" if "--pff" in sys.argv else "statsbomb"
    n_matches = 8 if data_source == "pff" else 10
    print("  Cognitive Alpha — Model Validation")
    print(f"  Data Source: {data_source.upper()}")
    print(f"  Matches: {n_matches}")
    if data_source == "pff":
        print("  Features: PFF Tracking | Real-Time Speeds | Body Orientation Vectors")
    else:
        print("  Features: Possession-ID | Distance-control | Baseline | Parallel")

    print("\n1. Computing α for all passes (parallel)...")
    df = compute_validation_data(n_matches=n_matches, data_source=data_source)
    print(f"  → {len(df)} passes across {df['match_id'].nunique()} matches")

    print("\n2. Generating validation dashboard...")
    suffix = f"_{data_source}" if data_source == "pff" else ""
    plot_validation(df, save_path=str(OUTPUT_DIR / f"model_validation{suffix}.png"))

    # Key findings
    print("\n3. Key Findings:")
    success = df[df["is_successful"]]
    failed = df[~df["is_successful"]]
    print(f"  Mean α (successful): {success['alpha'].mean():+.6f}")
    print(f"  Mean α (failed):     {failed['alpha'].mean():+.6f}")
    print(f"  Correlation (α↔suc): r = {df['alpha'].corr(df['is_successful'].astype(float)):.4f}")

    # Baseline: outcome comparison on disagreement cohorts only.
    # (Splitting on opt_xev > nearest_teammate_xev is degenerate: opt_xev
    # is the argmax of the surface nearest_teammate_xev is read from.)
    went_model, went_baseline = disagreement_cohorts(df)
    if len(went_model) > 0 and len(went_baseline) > 0:
        print(f"  Disagreements — chose model target (n={len(went_model)}): "
              f"shot rate {went_model['shot_in_possession'].mean():.1%}, "
              f"turnover {went_model['possession_lost'].mean():.1%}")
        print(f"  Disagreements — chose nearest TM  (n={len(went_baseline)}): "
              f"shot rate {went_baseline['shot_in_possession'].mean():.1%}, "
              f"turnover {went_baseline['possession_lost'].mean():.1%}")
    else:
        print("  Not enough disagreement passes for a baseline comparison.")

    if data_source == "pff":
        print("\n  Body orientation turn penalty ACTIVE")
        print("  Newtonian kinematics from real-time speeds")
    
    return df


if __name__ == "__main__":
    main()
