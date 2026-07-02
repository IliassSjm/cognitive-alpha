"""
PFF Tracking Analytics — Targeted Deep Dive
=============================================
Computes 2 tactical analytics from 30fps PFF tracking data:

1. Off-Ball Sprint Quality — Receiver sprint distance before reception
2. Pass Availability Model — Body orientation proxy from velocity vectors

Ruthlessly scoped: only metrics that directly complete the physics engine
or produce unique scouting value. No macro-tactical team metrics.

Dependencies: numpy, pandas
Data: PFF JSONL.bz2 tracking files at 30fps + Event Data JSON
"""

import bz2
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from pff_loader import (
    DATA_DIR,
    pff_to_statsbomb as pff_to_sb,
)

CACHE_DIR = Path(__file__).parent / "tracking_cache"
CACHE_DIR.mkdir(exist_ok=True)

FPS = 30


# ---- Frame Streaming — Efficient bz2 JSONL reader ----
def stream_tracking_frames(
    game_id: int,
    subsample: int = 1,
    max_frames: int | None = None,
) -> list[dict]:
    """
    Stream tracking frames from a compressed JSONL file.

    Args:
        game_id: PFF game ID
        subsample: Take every Nth frame (1 = all, 6 = 5fps)
        max_frames: Cap total frames loaded (None = all)

    Returns:
        List of frame dicts with parsed player/ball data
    """
    path = DATA_DIR / "Tracking Data" / f"{game_id}.jsonl.bz2"
    if not path.exists():
        raise FileNotFoundError(f"No tracking data for game {game_id}")

    frames = []
    count = 0
    with bz2.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i % subsample != 0:
                continue
            try:
                frame = json.loads(line)
                frames.append(frame)
                count += 1
                if max_frames and count >= max_frames:
                    break
            except json.JSONDecodeError:
                continue

    return frames


def frames_to_arrays(frames: list[dict]) -> dict:
    """
    Convert raw frames to numpy arrays for vectorized computation.

    Returns dict with:
        timestamps: (N,) game clock times
        periods: (N,) period numbers
        home_pos: (N, 11, 2) home player positions [x, y] in meters
        away_pos: (N, 11, 2) away player positions
        home_speed: (N, 11) home player speeds m/s
        away_speed: (N, 11) away player speeds m/s
        home_jerseys: list of jersey numbers
        away_jerseys: list of jersey numbers
        ball_pos: (N, 3) ball [x, y, z] in meters
    """
    n = len(frames)
    home_pos = np.full((n, 11, 2), np.nan)
    away_pos = np.full((n, 11, 2), np.nan)
    home_speed = np.full((n, 11), np.nan)
    away_speed = np.full((n, 11), np.nan)
    ball_pos = np.full((n, 3), np.nan)
    timestamps = np.zeros(n)
    periods = np.zeros(n, dtype=int)

    # Discover jersey order from first frame
    first = frames[0]
    home_jerseys = [p.get("jerseyNum", "?") for p in first.get("homePlayers", [])]
    away_jerseys = [p.get("jerseyNum", "?") for p in first.get("awayPlayers", [])]

    for i, frame in enumerate(frames):
        timestamps[i] = frame.get("periodGameClockTime", 0)
        periods[i] = frame.get("period", 1)

        for j, p in enumerate(frame.get("homePlayers", [])[:11]):
            home_pos[i, j] = [p.get("x", np.nan), p.get("y", np.nan)]
            home_speed[i, j] = p.get("speed", np.nan)

        for j, p in enumerate(frame.get("awayPlayers", [])[:11]):
            away_pos[i, j] = [p.get("x", np.nan), p.get("y", np.nan)]
            away_speed[i, j] = p.get("speed", np.nan)

        balls = frame.get("balls", frame.get("ballsSmoothed", [{}]))
        if isinstance(balls, list) and len(balls) > 0:
            b = balls[0]
        elif isinstance(balls, dict):
            b = balls
        else:
            b = {}
        ball_pos[i] = [b.get("x", np.nan), b.get("y", np.nan), b.get("z", np.nan)]

    return {
        "timestamps": timestamps,
        "periods": periods,
        "home_pos": home_pos,
        "away_pos": away_pos,
        "home_speed": home_speed,
        "away_speed": away_speed,
        "home_jerseys": home_jerseys,
        "away_jerseys": away_jerseys,
        "ball_pos": ball_pos,
    }


# ---- 1. OFF-BALL SPRINT QUALITY ----
def compute_offball_sprints(
    frames: list[dict],
    game_id: int,
    lookback_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    For each pass reception event, compute the receiver's sprint distance
    in the 2.0 seconds (60 frames) before the ball arrives.

    This tells us WHO CREATES SPACE (not just who finds it).

    Scouting value: Players with high mean sprint distance are active
    space creators, not passengers.

    Uses PFF Event Data with tracking frame lookup.
    """
    lookback_frames = int(lookback_seconds * FPS)

    # Load event data
    event_path = DATA_DIR / "Event Data" / f"{game_id}.json"
    if not event_path.exists():
        return pd.DataFrame()

    with open(event_path) as f:
        events = json.load(f)

    # Load roster mapping
    from pff_loader import load_pff_match
    match_info = load_pff_match(game_id)
    jersey_map = match_info["jersey_to_player"]

    # Build frame lookup by frame number
    frame_lookup = {}
    for frame in frames:
        fn = frame.get("frameNum")
        if fn is not None:
            frame_lookup[fn] = frame

    results = []
    for e in events:
        pe = e.get("possessionEvents", {})
        if not pe or pe.get("possessionEventType") != "PA":
            continue
        if pe.get("passOutcomeType") != "C":
            continue

        receiver_name = pe.get("receiverPlayerName", "Unknown")
        receiver_id = pe.get("receiverPlayerId")

        # Get the start frame from the game event
        ge = e.get("gameEvents", {})
        start_time = e.get("startTime")

        # Find the frame closest to this time
        is_home = ge.get("homeTeam", False)
        player_key = "homePlayers" if is_home else "awayPlayers"
        team_side = "home" if is_home else "away"

        # Find receiver's jersey number using player ID from event data
        receiver_jersey = None
        # First try: from event data itself (players have playerId)
        event_players = e.get(player_key, [])
        for p in event_players:
            if p.get("playerId") == receiver_id:
                receiver_jersey = str(p.get("jerseyNum", ""))
                break

        # Second try: from roster
        if not receiver_jersey:
            for (side, jersey), info in jersey_map.items():
                if side == team_side and (
                    info.get("playerId") == receiver_id
                    or info.get("name") == receiver_name
                ):
                    receiver_jersey = str(jersey)
                    break

        if not receiver_jersey:
            continue

        # Find the tracking frame closest to the event start_time
        # We need to scan the tracking frames for this
        start_frame = None
        if start_time:
            # The tracking frame "videoTimeMs" = start_time * 1000
            target_ms = start_time * 1000
            best_diff = float("inf")
            for fn, fr in frame_lookup.items():
                diff = abs(fr.get("videoTimeMs", 0) - target_ms)
                if diff < best_diff:
                    best_diff = diff
                    start_frame = fn

        if start_frame is None:
            continue

        # Compute sprint distance in lookback window
        sprint_dist = 0.0
        max_speed = 0.0
        n_sprint_frames = 0
        n_valid = 0

        for fn in range(start_frame - lookback_frames, start_frame):
            frame = frame_lookup.get(fn)
            if frame is None:
                continue

            players = frame.get(player_key, [])
            for p in players:
                if str(p.get("jerseyNum")) == receiver_jersey:
                    spd = p.get("speed", 0)
                    sprint_dist += spd / FPS  # distance = speed * dt
                    max_speed = max(max_speed, spd)
                    n_valid += 1
                    if spd > 5.5:  # sprint threshold m/s
                        n_sprint_frames += 1
                    break

        game_clock = pe.get("gameClock", 0)
        team_name = ge.get("teamName", team_side)

        results.append({
            "player": receiver_name,
            "team": team_name,
            "minute": game_clock // 60 if isinstance(game_clock, (int, float)) else 0,
            "sprint_distance_m": round(sprint_dist, 2),
            "max_speed_ms": round(max_speed, 2),
            "sprint_frames": n_sprint_frames,
            "sprint_pct": round(
                n_sprint_frames / max(n_valid, 1) * 100, 1
            ),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        # Aggregate per player
        agg = df.groupby("player").agg(
            team=("team", "first"),
            receptions=("sprint_distance_m", "count"),
            mean_sprint_dist=("sprint_distance_m", "mean"),
            max_sprint_dist=("sprint_distance_m", "max"),
            mean_max_speed=("max_speed_ms", "mean"),
            mean_sprint_pct=("sprint_pct", "mean"),
        ).round(2).sort_values("mean_sprint_dist", ascending=False).reset_index()
        return agg
    return df


# ---- 2. PASS AVAILABILITY MODEL (Velocity-Vector Body Orientation) ----
def compute_pass_availability(
    data: dict,
    velocity_window: int = 5,  # frames for velocity estimation (0.17s)
    angle_threshold: float = 90.0,  # degrees — "facing" toward ball
    min_separation: float = 3.0,  # meters from nearest defender
) -> pd.DataFrame:
    """
    Body orientation proxy solves the "omnidirectional player" flaw.

    A player running full speed AWAY from the ball cannot instantly
    receive a pass. This metric identifies who is ACTUALLY available.

    Available = facing within angle_threshold of ball AND
                ≥ min_separation meters from nearest defender

    Output: per-frame availability for each team.
    Can be used as angular penalty: cos(θ) applied to xEV surface.
    """
    ball = data["ball_pos"][:, :2]
    n_frames = len(data["timestamps"])
    results = []

    for team_label, tm_pos, opp_pos in [
        ("home", data["home_pos"], data["away_pos"]),
        ("away", data["away_pos"], data["home_pos"]),
    ]:
        for i in range(velocity_window, n_frames, 6):  # 5fps sampling
            n_available = 0
            n_outfield = 0
            facing_angles = []

            for j in range(11):
                curr = tm_pos[i, j]
                prev = tm_pos[i - velocity_window, j]

                if np.isnan(curr[0]) or np.isnan(prev[0]):
                    continue

                n_outfield += 1

                # Velocity direction (body orientation proxy)
                vel = curr - prev
                vel_norm = np.linalg.norm(vel)

                # Skip ball carrier
                to_ball = ball[i] - curr
                to_ball_norm = np.linalg.norm(to_ball)
                if to_ball_norm < 2.0:
                    continue

                to_ball_dir = to_ball / to_ball_norm

                # Angle between velocity and ball direction
                if vel_norm > 0.3:  # minimum movement threshold
                    vel_dir = vel / vel_norm
                    cos_angle = np.clip(np.dot(vel_dir, to_ball_dir), -1, 1)
                    angle = np.degrees(np.arccos(cos_angle))
                    facing_ball = angle < angle_threshold
                    facing_angles.append(angle)
                else:
                    facing_ball = True  # Stationary ≈ open to receive
                    facing_angles.append(0)

                # Defender separation
                opp_xy = opp_pos[i]
                valid_opp = ~np.isnan(opp_xy[:, 0])
                if valid_opp.any():
                    dists = np.linalg.norm(opp_xy[valid_opp] - curr, axis=1)
                    min_dist = dists.min()
                else:
                    min_dist = 99.0

                if facing_ball and min_dist >= min_separation:
                    n_available += 1

            if n_outfield > 0:
                results.append({
                    "time": float(data["timestamps"][i]),
                    "minute": int(data["timestamps"][i] // 60),
                    "period": int(data["periods"][i]),
                    "team": team_label,
                    "n_available": n_available,
                    "n_outfield": n_outfield,
                    "availability_pct": round(n_available / n_outfield * 100, 1),
                    "mean_facing_angle": round(np.mean(facing_angles), 1) if facing_angles else 0,
                })

    return pd.DataFrame(results)


# ---- MASTER PIPELINE — Compute targeted analytics for a match ----
def analyse_match(
    game_id: int,
    use_cache: bool = True,
) -> dict:
    """
    Run targeted tracking analytics for a match.

    Returns dict with keys:
        sprints: per-player off-ball sprint quality
        availability: per-frame pass availability (body orientation)
    """
    # Check for cached results
    if use_cache:
        cached = {}
        all_cached = True
        for key in ["sprints", "availability"]:
            p = CACHE_DIR / f"{game_id}_{key}.parquet"
            if p.exists():
                cached[key] = pd.read_parquet(p)
            else:
                all_cached = False
                break
        if all_cached:
            print(f"  Loaded cached analytics for game {game_id}")
            return cached

    print(f"\n  Loading tracking data for game {game_id}...")
    frames = stream_tracking_frames(game_id, subsample=1)
    print(f"  → {len(frames)} frames at 30fps")

    data = frames_to_arrays(frames)

    # 1. Off-ball sprints
    print("  1. Computing off-ball sprint quality...")
    sprints = compute_offball_sprints(frames, game_id)
    print(f"     → {len(sprints)} players analysed")

    # 2. Pass availability
    print("  2. Computing pass availability (body orientation)...")
    availability = compute_pass_availability(data)
    print(f"     → {len(availability)} availability snapshots")

    # Cache results
    result = {"sprints": sprints, "availability": availability}
    for key, df in result.items():
        if not df.empty:
            df.to_parquet(CACHE_DIR / f"{game_id}_{key}.parquet", index=False)

    print(f"  Cached analytics to {CACHE_DIR}")
    return result


# ---- CLI Entry Point ----
if __name__ == "__main__":
    import sys

    game_id = int(sys.argv[1]) if len(sys.argv) > 1 else 10517
    print(f"  PFF Tracking Analytics — Game {game_id}")
    print(f"  Metrics: Off-Ball Sprints | Pass Availability (Body Orientation)")

    results = analyse_match(game_id, use_cache=False)
    print("  RESULTS SUMMARY")

    if not results["sprints"].empty:
        s = results["sprints"]
        print(f"\n  Off-Ball Sprint Quality: {len(s)} players")
        print(f"  ┌──────────────────────────────────────────────────────┐")
        print(f"  │ {'Player':<25} │ {'Receptions':>10} │ {'Avg Sprint':>10} │ {'Max Speed':>9} │")
        print(f"  ├──────────────────────────────────────────────────────┤")
        for _, r in s.head(10).iterrows():
            print(f"  │ {r['player']:<25.25} │ {r['receptions']:>10} │ {r['mean_sprint_dist']:>8.1f} m │ {r['mean_max_speed']:>7.1f} m/s│")
        print(f"  └──────────────────────────────────────────────────────┘")
    else:
        print("\n  No off-ball sprint data (event data field mismatch?)")

    if not results["availability"].empty:
        a = results["availability"]
        for team in a["team"].unique():
            ta = a[a["team"] == team]
            print(f"\n  {team.title()} Pass Availability:")
            print(f"    Avg available players: {ta['n_available'].mean():.1f} / {ta['n_outfield'].mean():.0f}")
            print(f"    Avg availability:      {ta['availability_pct'].mean():.1f}%")
            print(f"    Avg facing angle:      {ta['mean_facing_angle'].mean():.1f}°")
    print("  Done!")
