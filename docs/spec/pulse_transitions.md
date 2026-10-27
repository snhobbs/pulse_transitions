# pulse_transitions — Spec (v0.3)

Reverse-engineered from v0.2.0 (README, `theory.md`, `docs/IEEE_181_2003_SIMPLIFIED.md`, `docs/IEEE_Std_181-2003_simplified.md`, the code and its tests), then revised for a new architecture. The public API may break; this is released as 0.3.0.

## Goal

A Python implementation of MATLAB's *Pulse and Transition Metrics* (`statelevels`, `midcross`, `risetime`, `falltime`, `slewrate`, `overshoot`, `undershoot`, `settlingtime`, `pulsewidth`, `pulseperiod`, `pulsesep`, `dutycycle`) whose algorithms follow IEEE Std 181-2003. Section numbers below (§5.x) refer to `docs/IEEE_Std_181-2003_simplified.md`. It also measures single edges of multi-level traces (TDR), which neither MATLAB nor IEEE 181 cover.

Users are engineers analysing scope captures, simulated step responses and TDR traces: 1e3 to 1e7 samples, often noisy, sometimes non-uniformly sampled.

## Inputs

- A waveform is `y` samples with either an `x` (time) array of the same length, strictly increasing, or a sample rate `fs` (then `x = arange(n) / fs`). Non-uniform `x` is allowed; never resample it.
- NaN samples are dropped (with their `x`) before analysis.
- Levels are `(low, high)` with `low < high`. Reference levels are given as percentages of the amplitude, default `(10, 50, 90)`.

## Features

### State levels (§5.2)
- Histogram method (§5.2.1): bins over `[y_min, y_max]`, split the histogram at the midpoint (or user fractions `f1, f2`), state level is the mode (default) or mean of each half. Bin count default 100.
- Alternatives (§5.2.2): peak magnitude (`y_min`, `y_max`), and initial/final value (mean of the first/last `n` samples).
- Returns the two levels plus the histogram used, so it can be plotted.

### Reference levels and instants (§5.3.2, §5.3.3)
- Percent reference level: `y_x = low + A * x / 100`.
- Reference level instants by linear interpolation between the bracketing samples. For each transition, the 50 % instant is found first; other instants are the crossings nearest to it (§5.3.3 default).

### Transitions
- Find every transition in the waveform: a transition goes from one state's boundary to the other's without returning (hysteresis between the lower and upper reference levels, so noise near a threshold does not create extra edges).
- Each transition records polarity, its reference-level instants (low, mid, high) and the sample indices that bracket it.
- An optional `bounds=(x_start, x_end)` restricts analysis to a window; with no levels given, levels then come from that window.

### Per-transition metrics
- **Rise/fall time** (§5.3.4): duration between the lower and upper reference instants, per transition. `risetime` covers rising transitions, `falltime` falling ones.
- **Slew rate**: `(y_upper - y_lower) / duration` per transition, signed (MATLAB `slewrate`). Not the maximum gradient.
- **Mid-crossing**: the 50 % instants of all transitions.
- **Overshoot / undershoot** (§5.3.5): in the pre- and post-transition aberration regions (`3 × t_10-90` long by default, clipped at the neighbouring transition), as percent of amplitude, with the instant and value of the extreme. Zero when the extreme doesn't pass the state boundary.
- **Settling time** (§5.3.7): from the 50 % instant to the last crossing of the state band around the final state level (band default ±2 % of amplitude), interpolated. NaN when the waveform hasn't settled before the next transition or the end of the record.

### Pulse metrics (§5.4) — new
- **Pulse width** (§5.4.1): mid-reference instant of a transition to the next transition of opposite polarity. Polarity of the pulse selectable (positive = rise then fall).
- **Pulse period** (§5.4.2): between same-polarity mid-reference instants of consecutive pulses.
- **Pulse separation** (§5.4.3): from the second transition of one pulse to the first of the next.
- **Duty cycle** (§5.4.4): pulse width / period, per pulse.
- Pulses with no complete period (first/last) give no period or duty cycle, rather than a guess.

### Single edge of a multi-level trace (TDR)
- Given an `x` position `around` and an optional `max_distance`, measure the nearest edge (rising or falling) between the flat parts on either side of it, or between given levels.
- Edge finding must scale with sample rate and edge length: in v0.2 it smoothed over a fixed 5 samples, so a slow edge sampled densely (an edge spanning 2000 samples in a 1M-sample trace with 1 % noise) was not found at all.
- Raises `EdgeNotFoundError` with a message that says what is missing: no edge near `around`, no flat region on one side, or the levels aren't crossed.

### Plateau flatness
- Standard deviation of `y` within a window, optionally normalised by the final level. `None`/NaN when the window has fewer than 5 samples or the final level is ~0. (Used for TDR plateau ripple.)

### Two interfaces
- A Pythonic API returning frozen dataclasses (per-transition records, a pulse record, a summary of all metrics for a step response).
- A MATLAB-named facade (`pulse_transitions.matlab`: `statelevels`, `risetime`, …) taking `(y, fs | x, ...)` and returning MATLAB-shaped results (arrays of values, plus instants and levels). It calls the same core and adds no logic.

## Error contract (carried from v0.2)

- `EdgeNotFoundError` when an operation that must return an edge (single-edge measurement) cannot. It stays a subclass of `ValueError`; it no longer needs to be an `IndexError`.
- Functions that return all transitions or pulses return empty results, not errors, when there are none.
- Unknown keyword arguments raise `TypeError` (no `**kwargs` pass-through anywhere).
- `levels` with `low >= high`, reference percentages outside `[0, 100]` or not increasing, `x` not increasing or a length mismatch: `ValueError` naming the argument.

## Known v0.2 defects to fix

1. Overshoot and undershoot are computed over the whole record, not the aberration regions; direction is guessed from the last sample.
2. Settling time returns the absolute time of the last out-of-band sample (band around `y[-1]`, not the state level) rather than a duration from the 50 % instant.
3. Slew rate is the maximum point gradient of the whole record, not per transition.
4. Pulse width is `fall.end - rise.start` (outer threshold crossings), not mid-reference to mid-reference.
5. Duty cycle is total high time over record length, not width over period; it isn't exported.
6. The histogram level method takes the two largest smoothed peaks instead of splitting the histogram (§5.2.1); "derivative" is only an alias for it.
7. Two edge-finding algorithms (a hysteresis state machine and a slope-run detector) with different results; two near-identical "first edge" functions.
8. `CrossingDetectionSettings.filter_*` filters only a copy used to locate edges; `window` is unused.
9. `**kwargs` threading through most functions; MATLAB-style functions use `x` for the signal.
10. Per-sample Python loops: `detect_edges` takes ~1 s for 1M samples.

## Architecture goals

- Split the 1,100-line `impl.py` by concern: levels, reference levels and crossings, transitions, aberrations and settling, pulses, multi-level edges, public API, MATLAB facade.
- One transition finder shared by every feature (including the TDR path, which then only differs in how it gets its levels and window).
- Pure functions over numpy arrays inside; validation and conversion of inputs once, at the public boundary.
- No logging at error level in library code; no silent `None` where the contract says raise.
- Keep the `src/` layout and `pyproject.toml` packaging.

## Performance

Measured on v0.2.0 (pulse train, 10 periods, 20 edges, 1 % noise; best of 3):

| n samples | levels | detect_edges | risetime | measure_edge |
|---|---|---|---|---|
| 1e4 | 0.3 ms | 11 ms | 3.6 ms | 3.6 ms |
| 1e5 | 3.0 ms | 91 ms | 37 ms | fails (no edge found) |
| 1e6 | 14 ms | 1009 ms | 373 ms | fails |

Requirements:
- No Python loop over samples; loops over transitions are fine.
- Finding all transitions of a 1e6-sample, 20-edge waveform: ≤ 100 ms on the same machine (≥ 10× faster). Level detection no slower than v0.2.
- Work scales linearly with sample count.
- A `benchmarks/` script reproduces the table above for v0.3 and prints the v0.2 figures beside it.

## Calling interface

```python
import numpy as np
import pulse_transitions as pt

t = np.linspace(0, 10e-6, 100_000)
y = ...  # scope capture

levels = pt.state_levels(y)                      # (low, high) + histogram
transitions = pt.transitions(y, x=t, levels=levels.levels)
for tr in transitions:
    print(tr.polarity, tr.duration, tr.mid_instant)

step = pt.step_metrics(y, x=t)                   # rise, overshoot, settling, ... for a step
pulses = pt.pulses(y, x=t)                       # width, period, separation, duty cycle

edge = pt.measure_edge(y, x=t, around=2.5e-6, reference=(20, 80))   # one edge of a TDR trace

from pulse_transitions import matlab
r, lt, ut, ll, ul = matlab.risetime(y, t)        # MATLAB-shaped
```

Names and shapes in this example are indicative; spec-expander settles them.

## Out of scope

- Compound waveform parsing (§5.5), impulse-like waveforms (§5.6), aberration analysis against a trapezoidal reference (§5.3.6), jitter (§5.9). They may come later as revisions.
- Plotting (an example script may plot).
