"""
PFF World Cup 2022 Tracking Data Loader
========================================
Loads PFF tracking + event data and transforms it into the format
expected by the Cognitive Alpha pitch control engine.

Coordinate Transform:
  PFF:       origin at center circle, meters  (x ∈ [-52.5, 52.5], y ∈ [-34, 34])
  StatsBomb: origin at bottom-left, yards     (x ∈ [0, 120], y ∈ [0, 80])

  sb_x = (pff_x + 52.5) / 105 * 120
  sb_y = 80 - (pff_y + 34.0) / 68 * 80    (PFF +y is opposite to StatsBomb +y)
"""

import json
import bz2
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Constants ----
DATA_DIR = Path(__file__).parent / "External_Data"

PITCH_LENGTH_M = 105.0   # PFF pitch length (meters)
PITCH_WIDTH_M  = 68.0    # PFF pitch width (meters)
SB_X_MAX       = 120.0   # StatsBomb x-range (yards)
SB_Y_MAX       = 80.0    # StatsBomb y-range (yards)

# PFF game IDs for key matches (mapped from StatsBomb match IDs)
PFF_KEY_MATCHES = {
    10517: "Final — Argentina vs France",
    10516: "3rd Place — Croatia vs Morocco",
    10515: "Semi — France vs Morocco",
    10514: "Semi — Argentina vs Croatia",
    10513: "QF — England vs France",
    10512: "QF — Morocco vs Portugal",
    10511: "QF — Netherlands vs Argentina",
    10510: "QF — Croatia vs Brazil",
}

# StatsBomb match ID → PFF game ID
# Single source of truth — verified against PFF Metadata/<id>.json team names
# and StatsBomb event data team names for all 8 matches.
SB_TO_PFF = {
    3869685: 10517,  # Final — Argentina vs France
    3869684: 10516,  # 3rd Place — Croatia vs Morocco
    3869552: 10515,  # Semi — France vs Morocco
    3869519: 10514,  # Semi — Argentina vs Croatia
    3869354: 10513,  # QF — England vs France
    3869486: 10512,  # QF — Morocco vs Portugal
    3869321: 10511,  # QF — Netherlands vs Argentina
    3869420: 10510,  # QF — Croatia vs Brazil
}


# ---- 1. Coordinate transform ----
def pff_to_statsbomb(x_m: float, y_m: float) -> tuple[float, float]:
    """Convert PFF pitch-centered meters to StatsBomb yards.

    PFF +y and StatsBomb +y point in opposite directions, so y is
    flipped. Verified empirically: 752 passes of the Final paired
    across both datasets agree to ~5 yd with the flip (35-70 yd without).
    """
    sb_x = (x_m + PITCH_LENGTH_M / 2) / PITCH_LENGTH_M * SB_X_MAX
    sb_y = SB_Y_MAX - (y_m + PITCH_WIDTH_M / 2) / PITCH_WIDTH_M * SB_Y_MAX
    return sb_x, sb_y


def pff_to_statsbomb_array(positions: list[dict]) -> np.ndarray:
    """Convert list of PFF player dicts → (N, 2) StatsBomb yard array."""
    if not positions:
        return np.zeros((0, 2))
    coords = []
    for p in positions:
        sb_x, sb_y = pff_to_statsbomb(p["x"], p["y"])
        coords.append([sb_x, sb_y])
    return np.array(coords)


# ---- 2. Load match metadata + rosters ----
def load_pff_match(game_id: int) -> dict:
    """Load metadata and rosters for a PFF game."""
    meta_path = DATA_DIR / "Metadata" / f"{game_id}.json"
    roster_path = DATA_DIR / "Rosters" / f"{game_id}.json"

    with open(meta_path) as f:
        metadata = json.load(f)
    if isinstance(metadata, list):
        metadata = metadata[0]

    with open(roster_path) as f:
        rosters = json.load(f)

    # Build jersey→player lookup
    home_team_id = metadata["homeTeam"]["id"]
    away_team_id = metadata["awayTeam"]["id"]

    jersey_to_player = {}
    for r in rosters:
        team_id = r["team"]["id"]
        side = "home" if team_id == home_team_id else "away"
        key = (side, int(r["shirtNumber"]))
        jersey_to_player[key] = {
            "name": r["player"]["nickname"],
            "playerId": r["player"]["id"],
            "position": r["positionGroupType"],
            "team": r["team"]["name"],
        }

    return {
        "metadata": metadata,
        "rosters": rosters,
        "jersey_to_player": jersey_to_player,
        "home_team": metadata["homeTeam"],
        "away_team": metadata["awayTeam"],
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_starts_left": bool(metadata.get("homeTeamStartLeft", True)),
        "fps": metadata.get("fps", 30.0),
        "pitch_length": metadata["stadium"]["pitches"][0]["length"],
        "pitch_width": metadata["stadium"]["pitches"][0]["width"],
    }


# ---- 3. Velocity vector helper ----
VEL_LOOKBACK = 5  # frames (~0.17 s at 30 fps)


def _compute_velocity_vector(
    player_t: dict,
    player_prev: dict | None,
) -> tuple[float, float]:
    """
    Return (vx, vy) in StatsBomb yards/s for one player.

    Algorithm:
      1. Compute displacement in SB space over 5 frames.
      2. Normalise to unit direction vector.
      3. Scale by the scalar speed at frame T.
    If the player wasn't visible 5 frames ago, fall back to (0, 0).
    """
    if player_prev is None:
        return 0.0, 0.0

    # Current and previous positions in SB space
    sx_t, sy_t = pff_to_statsbomb(player_t["x"], player_t["y"])
    sx_p, sy_p = pff_to_statsbomb(player_prev["x"], player_prev["y"])

    dx = sx_t - sx_p
    dy = sy_t - sy_p
    disp = (dx * dx + dy * dy) ** 0.5

    if disp < 1e-6:  # effectively stationary
        return 0.0, 0.0

    # Unit direction × scalar speed  →  velocity vector in SB yards/s
    # PFF speed is m/s; convert to yards/s  (1 m ≈ 1.0936 yd)
    speed_yps = player_t.get("speed", 0.0) * (SB_X_MAX / PITCH_LENGTH_M)
    vx = (dx / disp) * speed_yps
    vy = (dy / disp) * speed_yps
    return vx, vy


# ---- 4. Extract passes from event data (with optional tracking velocity vectors) ----
def extract_pff_passes(
    game_id: int,
    completed_only: bool = True,
    with_velocities: bool = False,
) -> pd.DataFrame:
    """
    Extract pass events with full spatial context.

    When ``with_velocities=True``, loads the 30 fps tracking JSONL.bz2
    to compute per-player velocity vectors from 5-frame displacement.
    This is SLOW (~2-5 min per match) and should only be used for
    validation runs, not for dashboard browsing.

    Returns DataFrame with columns:
      passer_name, receiver_name, team, minute, second,
      ball_height, pass_outcome, pressure_type,
      passer_x, passer_y (StatsBomb yards),
      receiver_x, receiver_y (StatsBomb yards),
      all_teammates (list of dicts with x, y, speed, [vx, vy], name),
      all_defenders (list of dicts with x, y, speed, [vx, vy], name),
      ball_z, passer_speed
    """
    # ── Load event data ────────────────────────────────────────────
    event_path = DATA_DIR / "Event Data" / f"{game_id}.json"
    with open(event_path) as f:
        events = json.load(f)

    # ── Optionally load tracking data for velocity vectors ─────────
    frame_lookup: dict[int, dict] = {}
    if with_velocities:
        tracking_path = DATA_DIR / "Tracking Data" / f"{game_id}.jsonl.bz2"
        if tracking_path.exists():
            with bz2.open(tracking_path, "rt") as tf:
                for line in tf:
                    try:
                        fr = json.loads(line)
                        fn = fr.get("frameNum")
                        if fn is not None:
                            frame_lookup[fn] = fr
                    except json.JSONDecodeError:
                        continue

    match_info = load_pff_match(game_id)
    jersey_map = match_info["jersey_to_player"]

    records = []
    for e in events:
        pe = e.get("possessionEvents", {})
        if not pe or pe.get("possessionEventType") != "PA":
            continue
        if completed_only and pe.get("passOutcomeType") != "C":
            continue

        ge = e["gameEvents"]
        is_home = ge.get("homeTeam", False)
        passer_team = "home" if is_home else "away"
        period = ge.get("period", 1)

        # Passer info
        passer_name = pe.get("passerPlayerName", ge.get("playerName", "Unknown"))
        receiver_name = pe.get("receiverPlayerName", "Unknown")
        game_clock = pe.get("gameClock", 0)
        minute = game_clock // 60
        second = game_clock % 60

        # Ball position
        ball_data = e.get("ball", [{}])
        if isinstance(ball_data, list) and len(ball_data) > 0:
            ball = ball_data[0]
        else:
            ball = ball_data if isinstance(ball_data, dict) else {}
        ball_x_pff = ball.get("x", 0)
        ball_y_pff = ball.get("y", 0)
        ball_z = ball.get("z", 0.0)

        passer_sb_x, passer_sb_y = pff_to_statsbomb(ball_x_pff, ball_y_pff)

        # ── Resolve tracking lookback frame ──────────────────────
        # Event data embed a frameNum (via game_event or top-level).
        start_frame = e.get("frameNum") or ge.get("frameNum")
        # Fallback: match by videoTimeMs
        if start_frame is None and frame_lookup:
            start_time = e.get("startTime")
            if start_time is not None:
                target_ms = int(start_time * 1000)
                best_fn, best_diff = None, float("inf")
                for fn, fr in frame_lookup.items():
                    d = abs(fr.get("videoTimeMs", 0) - target_ms)
                    if d < best_diff:
                        best_diff = d
                        best_fn = fn
                start_frame = best_fn

        # Lookback frame for velocity computation
        prev_frame_data: dict | None = None
        if start_frame is not None:
            prev_frame_data = frame_lookup.get(start_frame - VEL_LOOKBACK)

        # Player arrays — separate teammates from defenders
        home_players = e.get("homePlayers", [])
        away_players = e.get("awayPlayers", [])

        if passer_team == "home":
            tm_raw = home_players
            def_raw = away_players
            tm_side, def_side = "home", "away"
            tm_key_prev = "homePlayers"
            def_key_prev = "awayPlayers"
        else:
            tm_raw = away_players
            def_raw = home_players
            tm_side, def_side = "away", "home"
            tm_key_prev = "awayPlayers"
            def_key_prev = "homePlayers"

        # Build jersey → player-dict lookup from the lookback frame
        tm_prev_by_jersey: dict[str, dict] = {}
        def_prev_by_jersey: dict[str, dict] = {}
        if prev_frame_data is not None:
            for p in prev_frame_data.get(tm_key_prev, []):
                tm_prev_by_jersey[str(p.get("jerseyNum", ""))] = p
            for p in prev_frame_data.get(def_key_prev, []):
                def_prev_by_jersey[str(p.get("jerseyNum", ""))] = p

        # Build teammate list (exclude ball carrier)
        teammates = []
        passer_speed = 0.0
        passer_vx, passer_vy = 0.0, 0.0
        for p in tm_raw:
            pid = p.get("playerId")
            pname = jersey_map.get(
                (tm_side, int(p.get("jerseyNum", 0))), {}
            ).get("name", f"#{p.get('jerseyNum', '?')}")

            if pid == pe.get("passerPlayerId") or pid == ge.get("playerId"):
                passer_speed = p.get("speed", 0.0)
                jersey_str = str(p.get("jerseyNum", ""))
                prev_p = tm_prev_by_jersey.get(jersey_str)
                passer_vx, passer_vy = _compute_velocity_vector(p, prev_p)
                continue  # Skip passer from teammate list

            sb_x, sb_y = pff_to_statsbomb(p["x"], p["y"])
            jersey_str = str(p.get("jerseyNum", ""))
            prev_p = tm_prev_by_jersey.get(jersey_str)
            vx, vy = _compute_velocity_vector(p, prev_p)

            teammates.append({
                "x": sb_x, "y": sb_y,
                "speed": p.get("speed", 0.0),
                "vx": vx, "vy": vy,
                "name": pname,
            })

        # Build defender list
        defenders = []
        for p in def_raw:
            pname = jersey_map.get(
                (def_side, int(p.get("jerseyNum", 0))), {}
            ).get("name", f"#{p.get('jerseyNum', '?')}")
            sb_x, sb_y = pff_to_statsbomb(p["x"], p["y"])
            jersey_str = str(p.get("jerseyNum", ""))
            prev_p = def_prev_by_jersey.get(jersey_str)
            vx, vy = _compute_velocity_vector(p, prev_p)

            defenders.append({
                "x": sb_x, "y": sb_y,
                "speed": p.get("speed", 0.0),
                "vx": vx, "vy": vy,
                "name": pname,
            })

        # Find receiver position (closest teammate to target)
        receiver_pos = None
        receiver_pid = pe.get("receiverPlayerId")
        if receiver_pid:
            for p in tm_raw:
                if p.get("playerId") == receiver_pid:
                    rx, ry = pff_to_statsbomb(p["x"], p["y"])
                    receiver_pos = (rx, ry)
                    break

        # Fallback: use target player position
        if receiver_pos is None and len(teammates) > 0:
            target_name = pe.get("targetPlayerName", "")
            for t in teammates:
                if t["name"] == target_name:
                    receiver_pos = (t["x"], t["y"])
                    break

        if receiver_pos is None:
            continue  # Skip if we can't determine receiver

        records.append({
            "passer_name": passer_name,
            "receiver_name": receiver_name,
            "team": ge.get("teamName", "Unknown"),
            "minute": minute,
            "second": second,
            "ball_height": pe.get("ballHeightType", "G"),
            "pass_outcome": pe.get("passOutcomeType", ""),
            "pressure_type": pe.get("pressureType", "N"),
            "facing_type": e.get("initialTouch", {}).get("facingType", ""),
            "target_facing": pe.get("targetFacingType", ""),
            "receiver_facing": pe.get("receiverFacingType", ""),
            "passer_x": passer_sb_x,
            "passer_y": passer_sb_y,
            "receiver_x": receiver_pos[0],
            "receiver_y": receiver_pos[1],
            "ball_z": ball_z,
            "passer_speed": passer_speed,
            "passer_vx": passer_vx,
            "passer_vy": passer_vy,
            # Expert labels (PFF analyst annotations)
            "lines_broken": len(pe.get("linesBrokenType") or ""),  # A=1, AM=2, etc.
            "better_option_name": pe.get("betterOptionPlayerName") or None,
            "pass_type_pff": pe.get("passType", "S"),
            "all_teammates": teammates,
            "all_defenders": defenders,
            "is_home": is_home,
            "period": period,
            "home_starts_left": match_info.get("home_starts_left", True),
        })

    return pd.DataFrame(records)


# ---- 4. Convert PFF pass → pitch_control input ----
def pff_pass_to_spatial(row: pd.Series) -> dict:
    """
    Convert a PFF pass DataFrame row into the spatial inputs
    expected by compute_continuous_alpha().

    Returns dict with:
      bc_pos, tm_pos, def_pos, tm_names, def_names,
      tm_speeds, def_speeds, tm_velocities, def_velocities,
      actual_end, pass_type
    """
    bc_pos = np.array([row["passer_x"], row["passer_y"]])
    actual_end = np.array([row["receiver_x"], row["receiver_y"]])

    teammates = row["all_teammates"]
    defenders = row["all_defenders"]

    tm_pos = np.array([[t["x"], t["y"]] for t in teammates]) if teammates else np.zeros((0, 2))
    def_pos = np.array([[d["x"], d["y"]] for d in defenders]) if defenders else np.zeros((0, 2))

    tm_names = [t["name"] for t in teammates]
    def_names = [d["name"] for d in defenders]

    # Real-time speeds from PFF tracking (m/s)
    tm_speeds = np.array([t["speed"] for t in teammates]) if teammates else np.array([])
    def_speeds = np.array([d["speed"] for d in defenders]) if defenders else np.array([])

    # Velocity vectors from 5-frame displacement (SB yards/s)
    tm_velocities = (
        np.array([[t["vx"], t["vy"]] for t in teammates])
        if teammates and "vx" in teammates[0]
        else None
    )
    def_velocities = (
        np.array([[d["vx"], d["vy"]] for d in defenders])
        if defenders and "vx" in defenders[0]
        else None
    )

    # 3D Ball Physics: auto-detect lofted pass from ball Z height
    ball_z = row.get("ball_z", 0.0)
    pass_type = "Lofted Pass" if ball_z > 1.5 else "Ground Pass"

    # Attack direction detection from PFF metadata
    # PFF: homeTeamStartLeft=True → home attacks RIGHT in period 1
    is_home = row.get("is_home", True)
    period = int(row.get("period", 1))
    home_starts_left = row.get("home_starts_left", True)

    # Period 1/3/5 (odd) → home starts left; Period 2/4 (even) → sides swap
    home_attacks_right = home_starts_left if period % 2 == 1 else not home_starts_left
    attack_right = home_attacks_right if is_home else not home_attacks_right

    return {
        "bc_pos": bc_pos,
        "tm_pos": tm_pos,
        "def_pos": def_pos,
        "tm_names": tm_names,
        "def_names": def_names,
        "tm_speeds": tm_speeds,
        "def_speeds": def_speeds,
        "tm_velocities": tm_velocities,
        "def_velocities": def_velocities,
        "actual_end": actual_end,
        "pass_type": pass_type,
        "attack_right": attack_right,
        "passer_velocity": np.array([row.get("passer_vx", 0.0), row.get("passer_vy", 0.0)]),
    }


# ---- 6. Load tracking window for animation ----
TRACKING_FPS = 30
REPLAY_SECONDS = 3.0  # seconds before the pass to animate


def load_tracking_window(
    game_id: int,
    pass_start_time: float,
    *,
    window_before: float = REPLAY_SECONDS,
    window_after: float = 0.5,
) -> list[dict]:
    """
    Load tracking frames around a pass event for animation.

    Parameters
    ----------
    game_id : PFF game ID
    pass_start_time : Event startTime in seconds (from PFF event data)
    window_before : Seconds before the pass to capture (default 3.0)
    window_after : Seconds after the pass to capture (default 0.5)

    Returns
    -------
    List of frame dicts, each with:
      - time_offset: float (seconds relative to pass, negative = before)
      - home_players: list of (sb_x, sb_y, speed, jersey)
      - away_players: list of (sb_x, sb_y, speed, jersey)
      - ball: (sb_x, sb_y) or None
    Sorted by time_offset (earliest first).
    """
    tracking_path = DATA_DIR / "Tracking Data" / f"{game_id}.jsonl.bz2"
    if not tracking_path.exists():
        return []

    target_ms = int(pass_start_time * 1000)
    window_start_ms = target_ms - int(window_before * 1000)
    window_end_ms = target_ms + int(window_after * 1000)

    frames: list[dict] = []
    with bz2.open(tracking_path, "rt") as tf:
        for line in tf:
            try:
                fr = json.loads(line)
            except json.JSONDecodeError:
                continue

            vt = fr.get("videoTimeMs", 0)
            if vt < window_start_ms:
                continue
            if vt > window_end_ms:
                break  # tracking frames are chronological

            time_offset = (vt - target_ms) / 1000.0

            def _convert_players(player_list):
                result = []
                for p in (player_list or []):
                    sb_x, sb_y = pff_to_statsbomb(p["x"], p["y"])
                    result.append((
                        sb_x, sb_y,
                        p.get("speed", 0.0),
                        str(p.get("jerseyNum", "?")),
                    ))
                return result

            # Ball position
            balls = fr.get("balls") or fr.get("ballsSmoothed") or []
            ball_pos = None
            if balls:
                b = balls[0]
                ball_pos = pff_to_statsbomb(b.get("x", 0), b.get("y", 0))

            frames.append({
                "time_offset": time_offset,
                "home_players": _convert_players(fr.get("homePlayers")),
                "away_players": _convert_players(fr.get("awayPlayers")),
                "ball": ball_pos,
            })

    return frames


def extract_tracking_sequence(
    tracking_frames: list[dict],
    is_home_possession: bool = True,
) -> list[dict]:
    """
    Convert raw tracking frames into spatial dicts for compute_continuous_alpha().

    For each frame, determines ball carrier as the attacking player
    nearest the ball, then separates teammates and defenders.

    Parameters
    ----------
    tracking_frames : Output of load_tracking_window()
    is_home_possession : True if the attacking team is home

    Returns
    -------
    List of dicts, each with:
      - bc_pos: (2,) ball carrier position
      - tm_pos: (N, 2) teammate positions
      - def_pos: (M, 2) defender positions
      - time_offset: float
    """
    sequence = []
    for fr in tracking_frames:
        ball = fr.get("ball")
        if ball is None:
            continue

        ball_xy = np.array(ball[:2], dtype=float)

        # Attacking and defending player lists
        if is_home_possession:
            att_players = fr.get("home_players", [])
            def_players = fr.get("away_players", [])
        else:
            att_players = fr.get("away_players", [])
            def_players = fr.get("home_players", [])

        if len(att_players) < 2 or len(def_players) < 1:
            continue

        att_xy = np.array([[p[0], p[1]] for p in att_players], dtype=float)
        def_xy = np.array([[p[0], p[1]] for p in def_players], dtype=float)

        # Ball carrier = attacker nearest the ball
        dists_to_ball = np.linalg.norm(att_xy - ball_xy, axis=1)
        bc_idx = int(np.argmin(dists_to_ball))
        bc_pos = att_xy[bc_idx]

        # Teammates = all other attackers
        tm_mask = np.ones(len(att_xy), dtype=bool)
        tm_mask[bc_idx] = False
        tm_pos = att_xy[tm_mask]

        sequence.append({
            "bc_pos": bc_pos,
            "tm_pos": tm_pos,
            "def_pos": def_xy,
            "time_offset": fr["time_offset"],
        })

    return sequence


# ---- Sanity check ----
if __name__ == "__main__":
    print("PFF Loader — Sanity Check")

    # Load Final (Argentina vs France)
    game_id = 10517
    match = load_pff_match(game_id)
    print(f"Match: {match['home_team']['name']} vs {match['away_team']['name']}")
    print(f"  FPS: {match['fps']}")
    print(f"  Pitch: {match['pitch_length']}m × {match['pitch_width']}m")
    print(f"  Roster size: {len(match['rosters'])}")

    # Extract passes
    passes_df = extract_pff_passes(game_id)
    print(f"\n  Completed passes: {len(passes_df)}")
    print(f"  Teams: {passes_df['team'].unique()}")
    print(f"  Ball heights: {passes_df['ball_height'].value_counts().to_dict()}")

    # Test one pass
    row = passes_df.iloc[20]
    print(f"\n  Sample pass: {row['passer_name']} → {row['receiver_name']} ({row['minute']}:{row['second']:02d})")
    print(f"    Passer pos (SB): ({row['passer_x']:.1f}, {row['passer_y']:.1f})")
    print(f"    Receiver pos (SB): ({row['receiver_x']:.1f}, {row['receiver_y']:.1f})")
    print(f"    Passer speed: {row['passer_speed']:.1f} m/s")
    print(f"    Teammates: {len(row['all_teammates'])}")
    print(f"    Defenders: {len(row['all_defenders'])}")
    print(f"    Ball z: {row['ball_z']:.2f}m")

    # Test spatial conversion
    spatial = pff_pass_to_spatial(row)
    print(f"\n  Spatial output:")
    print(f"    bc_pos: {spatial['bc_pos']}")
    print(f"    tm_pos shape: {spatial['tm_pos'].shape}")
    print(f"    def_pos shape: {spatial['def_pos'].shape}")
    print(f"    tm_speeds: {spatial['tm_speeds'][:3]}...")
    print(f"    pass_type: {spatial['pass_type']}")

    # Coordinate sanity: center circle should map to (60, 40)
    cx, cy = pff_to_statsbomb(0, 0)
    print(f"\n  Coord check: PFF (0,0) → SB ({cx:.1f}, {cy:.1f}) [expect 60, 40]")
    gx, gy = pff_to_statsbomb(52.5, 0)
    print(f"  Coord check: PFF (52.5,0) → SB ({gx:.1f}, {gy:.1f}) [expect 120, 40]")
