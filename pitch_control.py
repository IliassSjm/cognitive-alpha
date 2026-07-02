"""
Continuous pitch control and spatial xEV engine.

Based on William Spearman's pitch control model (2018), extended with
per-player sprint speeds, a ground/lofted pass toggle, and a post-receipt
survival penalty ("hospital pass" discount).

All computations are strictly vectorized via NumPy broadcasting
over a 120×80 meshgrid — zero Python for-loops over cells.

Pipeline:
  1. Kinematic time-to-arrival surfaces (heterogeneous speeds)
  2. Pitch Control probability via logistic sigmoid
  3. Post-receipt survival density (xP_survival)
  4. Pass & Sprint decay penalties (exponential)
  5. Constrained Spatial xEV = PC × xT × xP_pass × xP_sprint × xP_survival
  6. Global optimal spatial target = argmax(Constrained_xEV)
"""

import numpy as np
from scipy.ndimage import zoom
from xt_model import XT_GRID

# ---- Physical constants ----
YARD_TO_METER    = 0.9144

# Ball physics (toggled by UI)
V_BALL_GROUND    = 15.0    # m/s — ground pass
V_BALL_LOFTED    = 11.0    # m/s — lofted pass (Z-axis hang time)
PASS_DECAY_GROUND = 60.0   # yards — Gaussian σ for ground pass distance
PASS_DECAY_LOFTED = 90.0   # yards — Gaussian σ for lofted pass distance

# Player defaults
DEFAULT_ATT_SPEED  = 8.0   # m/s — generic attacker sprint
DEFAULT_DEF_SPEED  = 8.2   # m/s — generic defender sprint
KEEPER_SPEED       = 6.0   # m/s — goalkeeper

REACTION_TIME      = 0.35  # seconds — defender reaction delay
SIGMOID_K          = 3.0   # sigmoid steepness
SPRINT_DECAY_LAMBDA = 15.0 # yards — Gaussian σ for sprint feasibility
MAX_RECEIVE_DIST   = 25.0  # yards — hard cutoff: no teammate within this = 0 xEV
PLAYER_ACCEL       = 3.0   # m/s² — human sprint acceleration
FORWARD_BIAS       = 2.0   # sigmoid steepness for forward-pass preference

# Survival density  (calibrated: softened to reward threat creation)
SURVIVAL_TAU       = 0.5   # seconds — pressure half-life
SURVIVAL_LAMBDA    = 0.15  # scaling factor (was 0.4 — crushed aggressive passes)

PITCH_X            = 120   # yards
PITCH_Y            = 80    # yards


# ---- Heterogeneous Player Speed Database  (The Mbappé Factor) ----
# Source: GPS tracking estimates (FIFA/Opta/public)
# Keys = StatsBomb full player name
PLAYER_SPEEDS: dict[str, float] = {
    # France — attackers
    "Kylian Mbappé Lottin":               9.8,
    "Ousmane Dembélé":                    9.4,
    "Randal Kolo Muani":                  9.0,
    "Antoine Griezmann":                  8.5,
    "Olivier Giroud":                     7.8,
    # France — midfield
    "Aurélien Djani Tchouaméni":          8.6,
    "Adrien Rabiot":                      8.3,
    "Eduardo Camavinga":                  8.8,
    "Youssouf Fofana":                    8.5,
    # France — defence
    "Theo Bernard François Hernández":    9.3,
    "Jules Koundé":                       9.0,
    "Raphaël Varane":                     8.5,
    "Dayotchanculle Upamecano":           8.7,
    "Ibrahima Konaté":                    8.9,
    "William Saliba":                     8.6,
    # France — keeper
    "Hugo Lloris":                        KEEPER_SPEED,

    # Argentina — attackers
    "Lionel Andrés Messi Cuccittini":     7.5,
    "Julián Álvarez":                     8.6,
    "Paulo Bruno Exequiel Dybala":        8.0,
    "Lautaro Javier Martínez":            8.3,
    # Argentina — midfield
    "Enzo Fernandez":                     8.4,
    "Rodrigo Javier De Paul":             8.6,
    "Alexis Mac Allister":                8.3,
    "Leandro Daniel Paredes":             7.6,
    "Exequiel Alejandro Palacios":        8.5,
    # Argentina — defence
    "Nicolás Hernán Otamendi":            7.2,
    "Cristian Gabriel Romero":            8.5,
    "Nicolás Alejandro Tagliafico":       8.2,
    "Nahuel Molina Lucero":               8.8,
    "Gonzalo Ariel Montiel":              8.3,
    "Marcos Javier Acuña":                8.1,
    "Lisandro Martínez":                  8.0,
    # Argentina — keeper
    "Damián Emiliano Martínez":           KEEPER_SPEED,
    "Ángel Fabián Di María Hernández":    8.8,

    # England
    "Kyle Walker":                        9.3,
    "Raheem Shaquille Sterling":          9.1,
    "Bukayo Saka":                        9.0,
    "Phil Foden":                         8.7,
    "Jude Bellingham":                    8.8,

    # Brazil
    "Vinícius José Paixão de Oliveira Júnior": 9.5,
    "Neymar da Silva Santos Junior":      8.6,
    "Richarlison de Andrade":             8.8,

    # Croatia
    "Luka Modrić":                        7.6,

    # Morocco
    "Achraf Hakimi Mouh":                 9.2,

    # Netherlands
    "Memphis Depay":                      8.9,
    "Denzel Justus Morris Dumfries":      9.1,

    # Portugal
    "Cristiano Ronaldo dos Santos Aveiro": 8.2,
}


def _get_player_speed(name: str | None, is_defender: bool = False) -> float:
    """Look up player speed; fall back to positional default."""
    if name and name in PLAYER_SPEEDS:
        return PLAYER_SPEEDS[name]
    return DEFAULT_DEF_SPEED if is_defender else DEFAULT_ATT_SPEED


# ---- 1. Pre-compute meshgrid (module-level, computed once) ----
_x = np.arange(0.5, PITCH_X, 1.0)   # cell centres: 0.5, 1.5, ..., 119.5
_y = np.arange(0.5, PITCH_Y, 1.0)   # cell centres: 0.5, 1.5, ..., 79.5
X_GRID, Y_GRID = np.meshgrid(_x, _y)   # both shape (80, 120)
GRID_COORDS = np.stack([X_GRID, Y_GRID], axis=-1)  # (80, 120, 2)


# ---- 2. Continuous xT surface (interpolated + risk-scaled at import time) ----
_XT_RAW = zoom(XT_GRID, (PITCH_Y / XT_GRID.shape[0],
                          PITCH_X / XT_GRID.shape[1]),
               order=3)   # bicubic → (80, 120)
_XT_RAW = np.clip(_XT_RAW, 0.0, None)

# Risk-appetite scaling: expand the variance near the box.
# Power 1.5 EXPANDS the gap between low-xT midfield and high-xT box.
# Linear scale 10.0 makes it competitive with Gaussian decay penalties.
XT_SURFACE = (np.power(_XT_RAW, 1.5)) * 10.0


# ---- 3. Vectorized distance helpers ----
def _distances_to_grid(pos: np.ndarray) -> np.ndarray:
    """(2,) → (80, 120) Euclidean distances in yards."""
    dx = X_GRID - pos[0]
    dy = Y_GRID - pos[1]
    return np.sqrt(dx * dx + dy * dy)


def _multi_distances_to_grid(positions: np.ndarray) -> np.ndarray:
    """(N, 2) → (N, 80, 120) Euclidean distances in yards."""
    diff = positions[:, np.newaxis, np.newaxis, :] - GRID_COORDS[np.newaxis, :, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _kinematic_time(
    dist_m: np.ndarray,
    v0: np.ndarray,
    v_max: np.ndarray,
    a: float = PLAYER_ACCEL,
) -> np.ndarray:
    """
    Vectorized Newtonian time-to-arrival with acceleration.

    Players accelerate from v0 → v_max at constant a (m/s²),
    then cruise at v_max for any remaining distance.

    Parameters
    ----------
    dist_m  : (N, 80, 120)  distances in meters
    v0      : (N, 1, 1)     initial speed per player (m/s)
    v_max   : (N, 1, 1)     max sprint speed per player (m/s)
    a       : float          acceleration constant (m/s²)

    Returns
    -------
    t_arrive : (N, 80, 120)  time to each cell in seconds
    """
    # Distance required to reach v_max
    d_accel = (v_max ** 2 - v0 ** 2) / (2.0 * a)          # (N, 1, 1)

    # Time spent in the acceleration phase (to reach v_max)
    t_accel = (v_max - v0) / a                              # (N, 1, 1)

    # Case 1: d <= d_accel (still accelerating when reaching cell)
    # Solve d = v0·t + 0.5·a·t² → t = (-v0 + √(v0² + 2ad)) / a
    discriminant = v0 ** 2 + 2.0 * a * dist_m               # (N, 80, 120)
    t_case1 = (-v0 + np.sqrt(np.maximum(discriminant, 0.0))) / a

    # Case 2: d > d_accel (accelerate to v_max, then cruise)
    t_case2 = t_accel + (dist_m - d_accel) / v_max

    return np.where(dist_m <= d_accel, t_case1, t_case2)


# ---- 4. Arrival surfaces  (heterogeneous speeds) ----
def compute_arrival_surfaces(
    ball_carrier_pos: np.ndarray,
    teammates_pos: np.ndarray,
    defenders_pos: np.ndarray,
    *,
    v_ball: float = V_BALL_GROUND,
    teammate_names: list[str] | None = None,
    defender_names: list[str] | None = None,
    teammate_speeds: np.ndarray | None = None,
    defender_speeds: np.ndarray | None = None,
    teammate_velocities: np.ndarray | None = None,
    reaction_time: float = REACTION_TIME,
) -> dict:
    """
    Compute time-to-arrival surfaces with per-player speeds
    and body-orientation turn penalty.

    When PFF real-time speeds are available, uses Newtonian kinematics
    (acceleration from v₀ to v_max at 3.0 m/s²).
    Speed priority: real-time array > name lookup > positional default.

    teammate_velocities : (N_tm, 2) velocity vectors (m/s) from tracking.
        When provided, applies a directional turn penalty: players running
        away from a target cell take up to 1.75× longer to arrive there.

    Returns dict with keys:
      t_ball, t_team_min, t_def_min    — each (80, 120)
      t_def_all                        — (N_def, 80, 120) for survival calc
      pass_dist_grid, sprint_dist_grid — each (80, 120), in yards
    """
    # Ball
    pass_dist_grid = _distances_to_grid(ball_carrier_pos)
    t_ball = (pass_dist_grid * YARD_TO_METER) / v_ball

    # ── Teammates (heterogeneous) ─────────────────────────────────
    if len(teammates_pos) > 0:
        tm_dist_all = _multi_distances_to_grid(teammates_pos)  # (N, 80, 120) yards
        tm_dist_m = tm_dist_all * YARD_TO_METER                # (N, 80, 120) meters

        # Resolve max sprint speed (name lookup or default)
        if teammate_names and len(teammate_names) == len(teammates_pos):
            tm_vmax = np.array([
                _get_player_speed(n, is_defender=False) for n in teammate_names
            ])
        else:
            tm_vmax = np.full(len(teammates_pos), DEFAULT_ATT_SPEED)

        # Resolve initial speed (PFF real-time or assume standing start at v_max)
        if teammate_speeds is not None and len(teammate_speeds) == len(teammates_pos):
            tm_v0 = np.clip(teammate_speeds, 0.0, tm_vmax)  # clip to [0, v_max]
            # Kinematic arrival: accelerate from v0 → v_max
            t_team_all = _kinematic_time(
                tm_dist_m,
                tm_v0[:, np.newaxis, np.newaxis],
                tm_vmax[:, np.newaxis, np.newaxis],
            )
        else:
            # StatsBomb mode: constant speed (no acceleration needed)
            t_team_all = tm_dist_m / tm_vmax[:, np.newaxis, np.newaxis]

        # ── Body Orientation Turn Penalty ─────────────────────────
        # If velocity vectors are provided, penalize arrival time for
        # players whose movement direction diverges from the target cell.
        # cos_sim = dot(vel_dir, target_dir) in [-1, +1]
        # turn_penalty = 1.0 + 0.75 * (1.0 - cos_sim) in [1.0, 2.5]
        # → Running toward cell: penalty ≈ 1.0 (no slowdown)
        # → Running away from cell: penalty ≈ 1.75 (75% slower)
        if teammate_velocities is not None and len(teammate_velocities) == len(teammates_pos):
            # Velocity direction: (N, 2)
            vel_norms = np.linalg.norm(teammate_velocities, axis=1, keepdims=True)
            # Only apply to moving players (> 0.5 m/s)
            moving_mask = vel_norms.squeeze() > 0.5
            vel_dir = np.where(
                vel_norms > 0.5,
                teammate_velocities / vel_norms,
                0.0,
            )  # (N, 2)

            # Direction from each player to every grid cell: (N, 80, 120, 2)
            # GRID_COORDS is (80, 120, 2) in StatsBomb yards; player pos in yards
            target_vec = (
                GRID_COORDS[np.newaxis, :, :, :]
                - teammates_pos[:, np.newaxis, np.newaxis, :]
            )  # (N, 80, 120, 2)
            target_norms = np.linalg.norm(target_vec, axis=-1, keepdims=True)
            target_norms = np.maximum(target_norms, 1e-6)
            target_dir = target_vec / target_norms  # (N, 80, 120, 2)

            # Cosine similarity: (N, 80, 120)
            cos_sim = np.sum(
                vel_dir[:, np.newaxis, np.newaxis, :] * target_dir, axis=-1
            )  # (N, 80, 120)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)

            # Turn penalty: 1.0 (aligned) → 1.75 (opposite)
            turn_penalty = 1.0 + 0.75 * (1.0 - cos_sim)  # (N, 80, 120)

            # Only apply to moving players; stationary = no penalty
            turn_penalty[~moving_mask] = 1.0

            t_team_all = t_team_all * turn_penalty

        t_team_min = np.min(t_team_all, axis=0)
        sprint_dist_grid = np.min(tm_dist_all, axis=0)
    else:
        t_team_all = np.full((1, PITCH_Y, PITCH_X), 999.0)
        t_team_min = np.full((PITCH_Y, PITCH_X), 999.0)
        sprint_dist_grid = np.full_like(t_ball, 999.0)

    # ── Defenders (heterogeneous) ─────────────────────────────────
    if len(defenders_pos) > 0:
        def_dist_all = _multi_distances_to_grid(defenders_pos)  # (N, 80, 120)
        def_dist_m = def_dist_all * YARD_TO_METER               # meters

        if defender_names and len(defender_names) == len(defenders_pos):
            def_vmax = np.array([
                _get_player_speed(n, is_defender=True) for n in defender_names
            ])
        else:
            def_vmax = np.full(len(defenders_pos), DEFAULT_DEF_SPEED)

        if defender_speeds is not None and len(defender_speeds) == len(defenders_pos):
            def_v0 = np.clip(defender_speeds, 0.0, def_vmax)
            t_def_all = _kinematic_time(
                def_dist_m,
                def_v0[:, np.newaxis, np.newaxis],
                def_vmax[:, np.newaxis, np.newaxis],
            ) + reaction_time
        else:
            t_def_all = def_dist_m / def_vmax[:, np.newaxis, np.newaxis] + reaction_time

        t_def_min = np.min(t_def_all, axis=0)
    else:
        t_def_all = np.full((1, PITCH_Y, PITCH_X), 999.0)
        t_def_min = np.full((PITCH_Y, PITCH_X), 999.0)

    return {
        "t_ball": t_ball,
        "t_team_min": t_team_min,
        "t_def_min": t_def_min,
        "t_def_all": t_def_all,
        "pass_dist_grid": pass_dist_grid,
        "sprint_dist_grid": sprint_dist_grid,
    }


# ---- 5. Pitch Control probability surface ----
def compute_pitch_control(
    t_ball: np.ndarray,
    t_team_min: np.ndarray,
    t_def_min: np.ndarray,
    k: float = SIGMOID_K,
) -> np.ndarray:
    """PC = sigmoid(k * (t_def - max(t_ball, t_team)))."""
    arrival_team = np.maximum(t_ball, t_team_min)
    time_margin = t_def_min - arrival_team
    return 1.0 / (1.0 + np.exp(-k * time_margin))


# ---- 6. Post-receipt survival density  ("Hospital Pass" penalty) ----
def compute_survival_density(
    t_ball: np.ndarray,
    t_team_min: np.ndarray,
    t_def_all: np.ndarray,
    *,
    tau: float = SURVIVAL_TAU,
    lam: float = SURVIVAL_LAMBDA,
) -> np.ndarray:
    """
    Vectorized post-receipt pressure from every defender.

    t_arrival = max(t_ball, t_team)     — moment ball is controlled
    margin_i  = t_def_i - t_arrival      — how late each defender arrives
    pressure_i = exp(-margin / tau)       if margin > 0 else 0
    total_pressure = Σ pressure_i
    xP_survival = exp(-λ * total_pressure)

    All broadcasting — zero loops over cells.
    Returns shape (80, 120).
    """
    t_arrival = np.maximum(t_ball, t_team_min)                    # (80, 120)
    # margin for each defender: (N_def, 80, 120) - (80, 120) via broadcast
    margin_after_receipt = t_def_all - t_arrival[np.newaxis, :, :]  # (N, 80, 120)

    # Pressure: only from defenders arriving AFTER receipt (margin > 0)
    pressure_per_def = np.where(
        margin_after_receipt > 0,
        np.exp(-margin_after_receipt / tau),
        0.0,
    )  # (N, 80, 120)

    total_pressure = np.sum(pressure_per_def, axis=0)  # (80, 120)
    xP_survival = np.exp(-lam * total_pressure)         # (80, 120)
    return xP_survival


# ---- 7. Spatial decay penalties ----
def compute_decay_penalties(
    pass_dist_grid: np.ndarray,
    sprint_dist_grid: np.ndarray,
    pass_lambda: float = PASS_DECAY_GROUND,
    sprint_lambda: float = SPRINT_DECAY_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian (squared) decay — short/medium passes are NOT penalized.
    Only extreme distances get suppressed.

    xP_pass   = exp(-0.5 * (d / σ)²)
    xP_sprint = exp(-0.5 * (d / σ)²)
    """
    xP_pass = np.exp(-0.5 * (pass_dist_grid / pass_lambda) ** 2)
    xP_sprint = np.exp(-0.5 * (sprint_dist_grid / sprint_lambda) ** 2)
    return xP_pass, xP_sprint


# ---- 7b. Passing Lane Interception Probability  (Ghost Ball Fix) ----
INTERCEPTION_SIGMA = 1.5    # yards — Gaussian kernel width for lane occlusion
LOFTED_INTERCEPT_MULT = 0.2 # 80% attenuation for lofted passes (ball in air)


def compute_interception_probability(
    ball_carrier_pos: np.ndarray,
    defenders_pos: np.ndarray,
    pass_type: str = "Ground Pass",
) -> np.ndarray:
    """
    Vectorized passing lane occlusion: P(ball survives all defenders).

    For every cell C on the 120×80 grid, the passing ray is P₀ → C.
    Each defender D is projected onto this ray.  If the perpendicular
    distance to the lane is small, the defender can intercept.

    xP_intercept = Π_d (1 − exp(−0.5 × (d_ortho / σ)²))

    Lofted passes attenuate interception probability by 80% (ball in air).

    Parameters
    ----------
    ball_carrier_pos : (2,)    — passer position [x, y]
    defenders_pos    : (N, 2)  — defender positions
    pass_type        : str     — "Ground Pass" or "Lofted Pass"

    Returns
    -------
    xP_intercept : (80, 120) — probability the pass survives all defenders
    """
    if len(defenders_pos) == 0:
        return np.ones((PITCH_Y, PITCH_X))

    P0 = ball_carrier_pos  # (2,)

    # Passing ray to every cell: v = C - P0   →  (80, 120, 2)
    v = GRID_COORDS - P0[np.newaxis, np.newaxis, :]  # (80, 120, 2)

    # dot(v, v) for each cell: (80, 120)
    v_dot_v = np.sum(v * v, axis=-1)  # (80, 120)
    v_dot_v = np.maximum(v_dot_v, 1e-6)  # avoid division by zero

    # For each defender: w = D - P0   →  (N, 2)
    N = len(defenders_pos)
    w = defenders_pos - P0[np.newaxis, :]  # (N, 2)

    # Project each defender onto every ray: t = dot(w, v) / dot(v, v)
    # w: (N, 1, 1, 2),  v: (1, 80, 120, 2)  →  dot product: (N, 80, 120)
    w_dot_v = np.sum(
        w[:, np.newaxis, np.newaxis, :] * v[np.newaxis, :, :, :],
        axis=-1,
    )  # (N, 80, 120)

    t = w_dot_v / v_dot_v[np.newaxis, :, :]  # (N, 80, 120)

    # Clamp t ∈ [0.05, 0.95]: only defenders BETWEEN passer and target
    t = np.clip(t, 0.05, 0.95)  # (N, 80, 120)

    # Closest point on each ray to each defender: P_closest = P0 + t * v
    # (N, 80, 120, 2)
    P_closest = (
        P0[np.newaxis, np.newaxis, np.newaxis, :]
        + t[:, :, :, np.newaxis] * v[np.newaxis, :, :, :]
    )  # (N, 80, 120, 2)

    # Orthogonal distance: || D - P_closest ||
    D_expanded = defenders_pos[:, np.newaxis, np.newaxis, :]  # (N, 1, 1, 2)
    dist_ortho = np.sqrt(
        np.sum((D_expanded - P_closest) ** 2, axis=-1)
    )  # (N, 80, 120)

    # Gaussian interception probability per defender
    interception_prob = np.exp(
        -0.5 * (dist_ortho / INTERCEPTION_SIGMA) ** 2
    )  # (N, 80, 120)

    # Lofted passes: ball is in the air → much harder to intercept mid-flight
    if pass_type == "Lofted Pass":
        interception_prob = interception_prob * LOFTED_INTERCEPT_MULT

    # Survival = product of (1 - P_intercept) over all defenders
    xP_intercept = np.prod(1.0 - interception_prob, axis=0)  # (80, 120)

    return xP_intercept


# ---- 8. Constrained Spatial xEV surface + global optimal ----
def compute_spatial_xev(
    pc_surface: np.ndarray,
    xP_pass: np.ndarray,
    xP_sprint: np.ndarray,
    xP_survival: np.ndarray,
    xP_intercept: np.ndarray | None = None,
    xt_surface: np.ndarray | None = None,
    ball_carrier_x: float | None = None,
    offside_line_x: float | None = None,
    attack_right: bool = True,
    sprint_dist_grid: np.ndarray | None = None,
    passer_pos: np.ndarray | None = None,
    passer_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Constrained Spatial Net xEV = (P_success × Reward) - (P_failure × Turnover_Cost)

    Bidirectional EPV: losing the ball at (X, Y) gives the opponent the ball
    attacking the other way. Turnover cost = flipped xT × 1.2 counter-attack mult.

    Parameters
    ----------
    attack_right : bool
        True = team attacks toward x=120, False = toward x=0.
    sprint_dist_grid : (80, 120) distance from each cell to nearest teammate.
        Used for hard reachability cutoff (MAX_RECEIVE_DIST).
    passer_pos : (2,) passer position for body orientation penalty.
    passer_velocity : (2,) passer velocity vector (vx, vy) in m/s.
        Penalizes passes requiring large body turns.
    """
    if xt_surface is None:
        xt_surface = XT_SURFACE

    # If attacking LEFT, flip the xT surface so the goal at x=0 has high xT
    if not attack_right:
        xt_surface = np.flip(xt_surface, axis=(0, 1))

    # 1. Total Probability of Success (with lane interception)
    p_success = pc_surface * xP_pass * xP_sprint * xP_survival
    if xP_intercept is not None:
        p_success = p_success * xP_intercept
    p_failure = 1.0 - p_success

    # 2. Opponent's Expected Threat (Turnover Cost)
    xt_opponent = np.flip(xt_surface, axis=(0, 1))
    COUNTER_ATTACK_MULT = 1.20
    turnover_cost = xt_opponent * COUNTER_ATTACK_MULT

    # 3. Net Expected Value (EPV)
    base = (p_success * xt_surface) - (p_failure * turnover_cost)

    # 4. Forward Progression Bias (direction-aware)
    if ball_carrier_x is not None:
        if attack_right:
            progression = (X_GRID - ball_carrier_x) / 60.0
        else:
            progression = (ball_carrier_x - X_GRID) / 60.0
        w_forward = 1.0 / (1.0 + np.exp(-FORWARD_BIAS * progression * 5.0))
        w_forward = np.maximum(w_forward, 0.15)
        base = np.where(base > 0, base * w_forward, base)

    # 5. Offside mask (direction-aware)
    if offside_line_x is not None and ball_carrier_x is not None:
        if attack_right:
            offside_mask = X_GRID > (offside_line_x + 1.0)
        else:
            offside_mask = X_GRID < (offside_line_x - 1.0)
        base = np.where(offside_mask, -turnover_cost, base)

    # 6. Teammate reachability mask — hard cutoff
    if sprint_dist_grid is not None:
        unreachable = sprint_dist_grid > MAX_RECEIVE_DIST
        base = np.where(unreachable, -np.abs(turnover_cost), base)

    # 7. Passer body orientation penalty
    if passer_velocity is not None and passer_pos is not None:
        passer_speed = np.linalg.norm(passer_velocity)
        if passer_speed > 0.5:  # only penalize if passer is moving
            # Direction from passer to each cell
            cell_dir = GRID_COORDS - passer_pos[np.newaxis, np.newaxis, :]  # (80,120,2)
            cell_dist = np.sqrt(np.sum(cell_dir ** 2, axis=-1, keepdims=True))
            cell_dir_norm = cell_dir / np.maximum(cell_dist, 1e-6)
            passer_dir = passer_velocity / passer_speed
            # cos(angle) between passer facing and cell direction
            cos_angle = np.sum(
                cell_dir_norm * passer_dir[np.newaxis, np.newaxis, :], axis=-1
            )  # (80, 120)
            # Penalty: forward=1.0, sideways=0.5, backward=0.35
            body_penalty = 0.5 + 0.5 * np.clip(cos_angle, -0.3, 1.0)
            base = base * body_penalty

    return base


def find_global_optimal(spatial_xev: np.ndarray) -> tuple[float, float, float]:
    """Find the (x, y) cell with the highest Spatial xEV."""
    idx = np.unravel_index(np.argmax(spatial_xev), spatial_xev.shape)
    row, col = idx
    return float(_x[col]), float(_y[row]), float(spatial_xev[row, col])


# ---- 9. High-level API: Continuous Cognitive Alpha  (v5) ----
def compute_continuous_alpha(
    ball_carrier_pos: np.ndarray,
    teammates_pos: np.ndarray,
    defenders_pos: np.ndarray,
    actual_end_pos: np.ndarray,
    *,
    pass_type: str = "Ground Pass",
    teammate_names: list[str] | None = None,
    defender_names: list[str] | None = None,
    teammate_speeds: np.ndarray | None = None,
    defender_speeds: np.ndarray | None = None,
    teammate_velocities: np.ndarray | None = None,
    attack_right: bool = True,
    passer_velocity: np.ndarray | None = None,
) -> dict:
    """
    Compute v6 constrained continuous spatial Cognitive Alpha.

    Constrained_xEV = PC × xT × xP_pass × xP_sprint × xP_survival
    α = Actual_xEV − Global_Optimal_xEV

    Parameters
    ----------
    pass_type : "Ground Pass" | "Lofted Pass"
        Toggles v_ball and pass_decay_lambda.
    teammate_names, defender_names : optional player name lists
        Enables heterogeneous speed lookup.
    teammate_speeds, defender_speeds : optional np.ndarray
        Real-time per-player speeds (m/s) from PFF tracking.
        When provided, overrides name-based lookup.
    teammate_velocities : optional np.ndarray (N, 2)
        Velocity vectors (vx, vy) in m/s from PFF tracking.
        Enables body-orientation turn penalty on arrival times.
    """
    # 3D ball physics toggle
    if pass_type == "Lofted Pass":
        v_ball = V_BALL_LOFTED
        pass_decay = PASS_DECAY_LOFTED
    else:
        v_ball = V_BALL_GROUND
        pass_decay = PASS_DECAY_GROUND

    # Arrival surfaces (heterogeneous speeds + body orientation)
    arrivals = compute_arrival_surfaces(
        ball_carrier_pos, teammates_pos, defenders_pos,
        v_ball=v_ball,
        teammate_names=teammate_names,
        defender_names=defender_names,
        teammate_speeds=teammate_speeds,
        defender_speeds=defender_speeds,
        teammate_velocities=teammate_velocities,
    )
    t_ball = arrivals["t_ball"]
    t_team = arrivals["t_team_min"]
    t_def  = arrivals["t_def_min"]

    # Pitch control
    pc_surface = compute_pitch_control(t_ball, t_team, t_def)

    # Survival density (uses per-defender arrival times)
    xP_survival = compute_survival_density(
        t_ball, t_team, arrivals["t_def_all"],
    )

    # Decay penalties (pass_decay adapts to ground/lofted)
    xP_pass, xP_sprint = compute_decay_penalties(
        arrivals["pass_dist_grid"], arrivals["sprint_dist_grid"],
        pass_lambda=pass_decay,
    )

    # Offside line: direction-aware
    # Attacking RIGHT: offside = 2nd-to-last defender from the right (high X)
    # Attacking LEFT:  offside = 2nd-to-last defender from the left (low X)
    if attack_right:
        def_x_sorted = np.sort(defenders_pos[:, 0])[::-1]  # descending
        if len(def_x_sorted) >= 2:
            offside_line_x = max(float(def_x_sorted[1]), 60.0)
        else:
            offside_line_x = None
    else:
        def_x_sorted = np.sort(defenders_pos[:, 0])  # ascending
        if len(def_x_sorted) >= 2:
            offside_line_x = min(float(def_x_sorted[1]), 60.0)
        else:
            offside_line_x = None

    # Lane interception (Ghost Ball Fix)
    xP_intercept = compute_interception_probability(
        ball_carrier_pos, defenders_pos, pass_type=pass_type,
    )

    # Constrained Spatial xEV (direction-aware + reachability + body orientation)
    spatial_xev = compute_spatial_xev(
        pc_surface, xP_pass, xP_sprint, xP_survival,
        xP_intercept=xP_intercept,
        ball_carrier_x=float(ball_carrier_pos[0]),
        offside_line_x=offside_line_x,
        attack_right=attack_right,
        sprint_dist_grid=arrivals["sprint_dist_grid"],
        passer_pos=ball_carrier_pos,
        passer_velocity=passer_velocity,
    )

    # Global optimal — constrained to reachable cells (within MAX_RECEIVE_DIST)
    sprint_dist = arrivals["sprint_dist_grid"]  # (80, 120)
    reachable = sprint_dist <= MAX_RECEIVE_DIST

    # Mask unreachable cells to -inf for argmax search
    constrained_xev = np.where(reachable, spatial_xev, -np.inf)

    if np.any(np.isfinite(constrained_xev)):
        opt_x, opt_y, opt_xev = find_global_optimal(constrained_xev)
    else:
        # Fallback: all cells unreachable → pick best teammate position directly
        tm_xevs = []
        for tp in teammates_pos:
            c = int(np.clip(tp[0] - 0.5, 0, PITCH_X - 1))
            r = int(np.clip(tp[1] - 0.5, 0, PITCH_Y - 1))
            tm_xevs.append(float(spatial_xev[r, c]))
        best_tm = int(np.argmax(tm_xevs))
        opt_x = float(teammates_pos[best_tm, 0])
        opt_y = float(teammates_pos[best_tm, 1])
        opt_xev = tm_xevs[best_tm]

    # Actual pass xEV
    ax_col = int(np.clip(actual_end_pos[0] - 0.5, 0, PITCH_X - 1))
    ax_row = int(np.clip(actual_end_pos[1] - 0.5, 0, PITCH_Y - 1))
    actual_xev = float(spatial_xev[ax_row, ax_col])
    actual_pc = float(pc_surface[ax_row, ax_col])
    actual_xt = float(XT_SURFACE[ax_row, ax_col])
    actual_survival = float(xP_survival[ax_row, ax_col])

    alpha = actual_xev - opt_xev

    return {
        # Surfaces
        "pc_surface": pc_surface,
        "spatial_xev": spatial_xev,
        "xt_surface": XT_SURFACE,
        "xP_pass": xP_pass,
        "xP_sprint": xP_sprint,
        "xP_survival": xP_survival,
        "xP_intercept": xP_intercept,
        # Optimal target
        "opt_x": opt_x,
        "opt_y": opt_y,
        "opt_xev": opt_xev,
        # Actual pass
        "actual_xev": actual_xev,
        "actual_pc": actual_pc,
        "actual_xt": actual_xt,
        "actual_survival": actual_survival,
        # Meta
        "pass_type": pass_type,
        "alpha": alpha,
        "offside_line_x": offside_line_x,
    }


# ---- Sanity check ----
if __name__ == "__main__":
    import time

    print("v5 Pitch Control — 3D Kinematics Sanity Check")

    bc = np.array([60.0, 40.0])
    tm = np.array([[75.0, 40.0], [80.0, 30.0], [70.0, 55.0], [85.0, 45.0]])
    df = np.array([[68.0, 42.0], [78.0, 38.0], [72.0, 50.0], [82.0, 35.0]])
    actual = np.array([75.0, 40.0])

    tm_names = ["Kylian Mbappé Lottin", "Ousmane Dembélé", "Antoine Griezmann", "Olivier Giroud"]
    df_names = ["Nicolás Hernán Otamendi", "Cristian Gabriel Romero", None, None]

    for pt in ["Ground Pass", "Lofted Pass"]:
        t0 = time.perf_counter()
        r = compute_continuous_alpha(
            bc, tm, df, actual,
            pass_type=pt,
            teammate_names=tm_names,
            defender_names=df_names,
        )
        dt = time.perf_counter() - t0
        print(f"\n  [{pt}] — {dt*1000:.1f} ms")
        print(f"    Optimal: ({r['opt_x']:.1f}, {r['opt_y']:.1f})  xEV={r['opt_xev']:.6f}")
        print(f"    Actual:  xEV={r['actual_xev']:.6f}  Survival={r['actual_survival']:.2%}")
        print(f"    α = {r['alpha']:+.6f}")
