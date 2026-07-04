# Cognitive Alpha

![ci](https://github.com/IliassSjm/cognitive-alpha/actions/workflows/ci.yml/badge.svg)

Spatial decision quality in football. For every open-play pass of the 2022 FIFA World Cup, the model measures the gap between the pass the player actually chose and the best option that was physically available:

```
alpha = xEV(actual target) − xEV(optimal target)
```

where the expected value of a target location combines five surfaces evaluated on a continuous 120×80 grid: pitch control, expected threat, pass completion probability, receiver sprint feasibility, and a post-receipt survival penalty (the "hospital pass" discount). Alpha near zero means the player found the best option; strongly negative alpha flags a missed opportunity. An interactive Streamlit dashboard replays any pass with the full surface, the optimum, and the actual choice.

![Demo — Messi to Mac Allister, 2022 final](figures/demo.gif)

*Above: rendered by the dashboard's animation engine — Messi to Mac Allister in extra time of the final. The full star is the model's optimal target, the hollow one the actual choice.*

## Components

**Expected Threat, trained from scratch** (`train_xt.py`). Markov-chain value iteration on StatsBomb event data: 125,566 open-play possession actions (251,132 after Y-axis mirroring) drawn from the 234,652-event log of all 64 matches. Failed passes are treated as absorbing states, so the transition matrix is turnover-aware. Converges in 100 iterations (Δ < 1e-8). The raw 12×8 grid correlates at r = 0.85 with Karun Singh's published Premier League grid, r = 0.93 after Gaussian smoothing — close enough to be sane, different enough to carry World Cup-specific structure. Smoothed and bilinearly interpolated to a continuous surface.

**Pitch control** (`pitch_control.py`). Spearman-style (2018): kinematic time-to-arrival with reaction time and acceleration, squashed through a logistic sigmoid, fully vectorised over the grid (no per-cell Python loops). Extensions: per-player sprint speeds for known players (Mbappé is not a median defender), a ground/lofted ball toggle with distinct speeds and decay kernels, an offside mask that assigns negative turnover value to any cell beyond the second-last defender, and the survival penalty as exponential pressure decay.

**Data fusion** (`pff_loader.py`, `tracking_analytics.py`). Two sources under one coordinate system: StatsBomb 360 freeze-frames (via `statsbombpy`) and PFF FC 30 fps broadcast tracking, which supplies real player velocities and body orientation for the matches where it exists. The tracking data (~5 GB) is used under PFF's research access program and is not redistributed here; the code expects it under `External_Data/`.

**Dashboard** (`app.py`). `streamlit run app.py` — all 64 World Cup matches (knockouts first), pass-by-pass surfaces, animated pre-pass tracking replays at 30 fps, per-player decision-quality aggregation.

## Validation

`validate_model.py` runs four checks (`--pff` for the tracking-enhanced version):

- Alpha quintiles vs outcomes: pass completion, shot within the same possession chain, and possession loss, with look-ahead strictly bounded to the same possession. Failed passes are evaluated at their *intended* target (nearest teammate to the recorded end point; PFF's annotated target), never at the interception point — otherwise the outcome leaks into alpha.
- The same analysis controlled for pass distance (terciles within distance bins), since long passes have both lower alpha and lower completion.
- A nearest-teammate baseline restricted to true disagreements: passes where the model target and the nearest teammate are more than 8 yards apart and the player's actual choice matches exactly one of them. Comparing downstream outcomes of these two cohorts avoids the circularity of scoring the baseline with the model's own surface.
- Agreement with PFF's human scout annotations: where scouts tagged a "better option" on a pass, the model's optimum is compared to the scout's suggestion.

![Validation](figures/model_validation_pff.png)

Headline numbers on the full tournament (all 64 matches, 66,348 passes with 30 fps tracking context): the top-alpha tercile posts the highest completion rate in three of four distance bins (91.1% vs 84.6% on 10–20 yd passes; the 35+ yd bin inverts — a stated limitation). On disagreement passes (n = 25,104), players who chose the model's target generated more shots within the possession (6.1% vs 4.8%, z = 4.4, p < 0.001) and broke 2.4x as many defensive lines (0.19 vs 0.08 per pass) at a lower completion rate (81.0% vs 89.5%) — the model prices risk/reward, not safety. Its optimum lands within 12 yards of the PFF scout's tagged "better option" on 33.9% of the 224 annotated passes.

One limitation to state plainly: even with intended-target evaluation, alpha embeds the completion probability of the chosen target, so the raw alpha–success correlation (r ≈ 0.02 on de-leaked data) is uninformative by itself. The distance-controlled and downstream-outcome checks carry the weight, and the scout-agreement axis is the most independent of the four.

## Other competitions

The engine is competition-agnostic in StatsBomb 360 mode. The dashboard's "Other StatsBomb 360" picker lists every open-data competition with freeze frames at runtime (Euro 2024, Women's Euro 2025, AFCON 2023, ...) — nothing to hardcode. The CLI tools take explicit IDs (discover them via `sb.competitions()`):

```bash
# validate on another tournament (StatsBomb source; 53/315 = Women's Euro 2025)
python3 validate_model.py --competition-id 53 --season-id 315 --n-matches 31

# retrain xT on that tournament (writes suffixed artifacts, never clobbers WC)
python3 train_xt.py --competition-id 53 --season-id 315
```

Two caveats. PFF tracking exists only for World Cup 2022, so other competitions run on freeze frames without real player velocities. And cross-competition alpha comparisons should keep the single WC-trained xT surface — retrained grids land in suffixed files and are opt-in by design.

## Layout

```
app.py                  Streamlit dashboard (entry point)
pitch_control.py        pitch control + spatial xEV engine
pff_loader.py           PFF tracking/event loader, coordinate transform
tracking_analytics.py   30 fps off-ball metrics (sprints, orientation)
train_xt.py             xT training (value iteration)
xt_model.py             trained grid + continuous lookup
validate_model.py       validation suite
xt_trained.{json,npy}   trained artifacts (committed, small)
figures/                output plots
External_Data/          StatsBomb + PFF raw data (git-ignored)
```

## Running

```bash
pip install -r requirements.txt
python train_xt.py        # rebuilds xT from StatsBomb open data (cached to parquet)
python validate_model.py  # StatsBomb validation; add --pff for tracking-enhanced
streamlit run app.py
```

StatsBomb World Cup 2022 event data is open and fetched automatically. The PFF tracking files are required only for the tracking-enhanced paths.
