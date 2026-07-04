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

import bisect
import bz2
import json
import os
import re
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

# StatsBomb match ID → PFF game ID — all 64 World Cup 2022 matches.
# Generated from PFF Metadata/<id>.json + StatsBomb event data;
# verified 1:1 by team names (goal count disambiguates CRO-MAR x2).
SB_TO_PFF = {
    3869685: 10517,  # F — Argentina vs France
    3869684: 10516,  # 3P — Croatia vs Morocco
    3869519: 10514,  # SF — Argentina vs Croatia
    3869552: 10515,  # SF — France vs Morocco
    3869420: 10510,  # QF — Croatia vs Brazil
    3869321: 10511,  # QF — Netherlands vs Argentina
    3869486: 10512,  # QF — Morocco vs Portugal
    3869354: 10513,  # QF — England vs France
    3869117: 10502,  # R16 — Netherlands vs United States
    3869151: 10503,  # R16 — Argentina vs Australia
    3869152: 10504,  # R16 — France vs Poland
    3869118: 10505,  # R16 — England vs Senegal
    3869219: 10506,  # R16 — Japan vs Croatia
    3869253: 10507,  # R16 — Brazil vs South Korea
    3869220: 10508,  # R16 — Morocco vs Spain
    3869254: 10509,  # R16 — Portugal vs Switzerland
    3857267: 3844,  # GS3 — Ecuador vs Senegal
    3857294: 3845,  # GS3 — Netherlands vs Qatar
    3857261: 3846,  # GS3 — Wales vs England
    3857278: 3847,  # GS3 — Iran vs United States
    3857257: 3848,  # GS3 — Australia vs Denmark
    3857275: 3849,  # GS3 — Tunisia vs France
    3857264: 3850,  # GS3 — Poland vs Argentina
    3857260: 3851,  # GS3 — Saudi Arabia vs Mexico
    3857296: 3852,  # GS3 — Croatia vs Belgium
    3857276: 3853,  # GS3 — Canada vs Morocco
    3857255: 3854,  # GS3 — Japan vs Spain
    3857292: 3855,  # GS3 — Costa Rica vs Germany
    3857293: 3856,  # GS3 — Ghana vs Uruguay
    3857262: 3857,  # GS3 — South Korea vs Portugal
    3857256: 3858,  # GS3 — Serbia vs Switzerland
    3857280: 3859,  # GS3 — Cameroon vs Brazil
    3857273: 3828,  # GS2 — Wales vs Iran
    3857301: 3829,  # GS2 — Qatar vs Senegal
    3857274: 3830,  # GS2 — Netherlands vs Ecuador
    3857272: 3831,  # GS2 — England vs United States
    3857288: 3832,  # GS2 — Tunisia vs Australia
    3857297: 3833,  # GS2 — Poland vs Saudi Arabia
    3857266: 3834,  # GS2 — France vs Denmark
    3857289: 3835,  # GS2 — Argentina vs Mexico
    3857295: 3836,  # GS2 — Japan vs Costa Rica
    3857283: 3837,  # GS2 — Belgium vs Morocco
    3857281: 3838,  # GS2 — Croatia vs Canada
    3857263: 3839,  # GS2 — Spain vs Germany
    3857259: 3840,  # GS2 — Cameroon vs Serbia
    3857299: 3841,  # GS2 — South Korea vs Ghana
    3857269: 3842,  # GS2 — Brazil vs Switzerland
    3857270: 3843,  # GS2 — Portugal vs Uruguay
    3857286: 3814,  # GS1 — Qatar vs Ecuador
    3857285: 3812,  # GS1 — Senegal vs Netherlands
    3857271: 3813,  # GS1 — England vs Iran
    3857282: 3815,  # GS1 — United States vs Wales
    3857300: 3816,  # GS1 — Argentina vs Saudi Arabia
    3857254: 3817,  # GS1 — Denmark vs Tunisia
    3857265: 3818,  # GS1 — Mexico vs Poland
    3857279: 3819,  # GS1 — France vs Australia
    3857277: 3820,  # GS1 — Morocco vs Croatia
    3857284: 3821,  # GS1 — Germany vs Japan
    3857291: 3822,  # GS1 — Spain vs Costa Rica
    3857268: 3823,  # GS1 — Belgium vs Canada
    3857290: 3824,  # GS1 — Switzerland vs Cameroon
    3857287: 3825,  # GS1 — Uruguay vs South Korea
    3857298: 3826,  # GS1 — Portugal vs Ghana
    3857258: 3827,  # GS1 — Brazil vs Serbia
}

# PFF game ID → display label, ordered knockouts-first.
PFF_MATCH_LABELS = {
    10517: "F — Argentina vs France",
    10516: "3P — Croatia vs Morocco",
    10514: "SF — Argentina vs Croatia",
    10515: "SF — France vs Morocco",
    10510: "QF — Croatia vs Brazil",
    10511: "QF — Netherlands vs Argentina",
    10512: "QF — Morocco vs Portugal",
    10513: "QF — England vs France",
    10502: "R16 — Netherlands vs United States",
    10503: "R16 — Argentina vs Australia",
    10504: "R16 — France vs Poland",
    10505: "R16 — England vs Senegal",
    10506: "R16 — Japan vs Croatia",
    10507: "R16 — Brazil vs South Korea",
    10508: "R16 — Morocco vs Spain",
    10509: "R16 — Portugal vs Switzerland",
    3844: "GS3 — Ecuador vs Senegal",
    3845: "GS3 — Netherlands vs Qatar",
    3846: "GS3 — Wales vs England",
    3847: "GS3 — Iran vs United States",
    3848: "GS3 — Australia vs Denmark",
    3849: "GS3 — Tunisia vs France",
    3850: "GS3 — Poland vs Argentina",
    3851: "GS3 — Saudi Arabia vs Mexico",
    3852: "GS3 — Croatia vs Belgium",
    3853: "GS3 — Canada vs Morocco",
    3854: "GS3 — Japan vs Spain",
    3855: "GS3 — Costa Rica vs Germany",
    3856: "GS3 — Ghana vs Uruguay",
    3857: "GS3 — South Korea vs Portugal",
    3858: "GS3 — Serbia vs Switzerland",
    3859: "GS3 — Cameroon vs Brazil",
    3828: "GS2 — Wales vs Iran",
    3829: "GS2 — Qatar vs Senegal",
    3830: "GS2 — Netherlands vs Ecuador",
    3831: "GS2 — England vs United States",
    3832: "GS2 — Tunisia vs Australia",
    3833: "GS2 — Poland vs Saudi Arabia",
    3834: "GS2 — France vs Denmark",
    3835: "GS2 — Argentina vs Mexico",
    3836: "GS2 — Japan vs Costa Rica",
    3837: "GS2 — Belgium vs Morocco",
    3838: "GS2 — Croatia vs Canada",
    3839: "GS2 — Spain vs Germany",
    3840: "GS2 — Cameroon vs Serbia",
    3841: "GS2 — South Korea vs Ghana",
    3842: "GS2 — Brazil vs Switzerland",
    3843: "GS2 — Portugal vs Uruguay",
    3814: "GS1 — Qatar vs Ecuador",
    3812: "GS1 — Senegal vs Netherlands",
    3813: "GS1 — England vs Iran",
    3815: "GS1 — United States vs Wales",
    3816: "GS1 — Argentina vs Saudi Arabia",
    3817: "GS1 — Denmark vs Tunisia",
    3818: "GS1 — Mexico vs Poland",
    3819: "GS1 — France vs Australia",
    3820: "GS1 — Morocco vs Croatia",
    3821: "GS1 — Germany vs Japan",
    3822: "GS1 — Spain vs Costa Rica",
    3823: "GS1 — Belgium vs Canada",
    3824: "GS1 — Switzerland vs Cameroon",
    3825: "GS1 — Uruguay vs South Korea",
    3826: "GS1 — Portugal vs Ghana",
    3827: "GS1 — Brazil vs Serbia",
}


# ---- 0b. Tracking file resolution (two folders on disk) ----
TRACKING_SUBDIRS = ("Tracking Data", "Tracking Data 2")


def resolve_tracking_path(game_id: int) -> Path | None:
    """Return the tracking file for a game, searching both tracking folders."""
    for sub in TRACKING_SUBDIRS:
        p = DATA_DIR / sub / f"{game_id}.jsonl.bz2"
        if p.exists():
            return p
    return None


def pff_matches_with_tracking() -> list[int]:
    """All PFF game IDs with a tracking file on disk, knockouts first."""
    return [gid for gid in PFF_MATCH_LABELS if resolve_tracking_path(gid) is not None]


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
        # None for matches without extra time; True means the home team
        # attacks right in period 3 (verified empirically on 10511/10517)
        "home_starts_left_et": metadata.get("homeTeamStartLeftExtraTime"),
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
    # Only frames within ±0.6 s of a pass are ever looked up (the pass
    # frame plus the 5-frame velocity lookback), so stream the file and
    # keep just those windows: ~10x faster than parsing every frame and
    # avoids holding the whole match (multi-GB) in memory.
    frame_lookup: dict[int, dict] = {}
    if with_velocities:
        t_path = resolve_tracking_path(game_id)
        if t_path is not None:
            windows = sorted(
                (e["startTime"] * 1000.0 - 600.0, e["startTime"] * 1000.0 + 600.0)
                for e in events
                if (e.get("possessionEvents") or {}).get("possessionEventType") == "PA"
                and e.get("startTime") is not None
            )
            merged: list[list[float]] = []
            for lo, hi in windows:
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(hi, merged[-1][1])
                else:
                    merged.append([lo, hi])
            window_lows = [w[0] for w in merged]
            vt_pattern = re.compile(r'"videoTimeMs":\s*([0-9.]+)')

            with bz2.open(t_path, "rt") as tf:
                for line in tf:
                    m = vt_pattern.search(line)
                    if not m:
                        continue
                    vt = float(m.group(1))
                    i = bisect.bisect_right(window_lows, vt) - 1
                    if i < 0 or vt > merged[i][1]:
                        continue
                    try:
                        fr = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fn = fr.get("frameNum")
                    if fn is not None:
                        frame_lookup[fn] = fr

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
            "home_starts_left_et": match_info.get("home_starts_left_et"),
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

    if period >= 3:
        # Extra time: ends are re-drawn at the ET coin toss, so regulation
        # parity is not a rule. PFF metadata records the ET orientation
        # explicitly; fall back to parity when the field is absent.
        hsl_et = row.get("home_starts_left_et")
        hsl_base = bool(hsl_et) if hsl_et is not None else home_starts_left
        home_attacks_right = hsl_base if period == 3 else not hsl_base
    else:
        # Period 1 → homeTeamStartLeft; Period 2 → sides swap
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
    t_path = resolve_tracking_path(game_id)
    if t_path is None:
        return []

    target_ms = int(pass_start_time * 1000)
    window_start_ms = target_ms - int(window_before * 1000)
    window_end_ms = target_ms + int(window_after * 1000)

    frames: list[dict] = []
    with bz2.open(t_path, "rt") as tf:
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
