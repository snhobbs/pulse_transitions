# pulse_transitions — Expanded Spec (v0.3)

## Goal

A Python implementation of MATLAB's _Pulse and Transition Metrics_ (`statelevels`, `midcross`, `risetime`, `falltime`, `slewrate`, `overshoot`, `undershoot`, `settlingtime`, `pulsewidth`, `pulseperiod`, `pulsesep`, `dutycycle`) whose algorithms follow IEEE Std 181-2003, plus single-edge measurement of multi-level (TDR) traces. Section numbers §5.x refer to `docs/IEEE_Std_181-2003_simplified.md`. Users analyse scope captures, simulated step responses and TDR traces of 1e3 to 1e7 samples, often noisy, sometimes non-uniformly sampled. Replaces v0.2.0; the public API breaks and the release is 0.3.0.

Source: `docs/spec/pulse_transitions.md` (brief spec, reverse-engineered from v0.2.0).

## Module layout

```
src/pulse_transitions/
├── __init__.py        # [simple]  public re-exports only, plus the `matlab` submodule
├── _types.py          # [simple]  Polarity, result dataclasses, exception hierarchy
├── _validate.py       # [simple]  input coercion and checks at the public boundary (raises InputError)
├── _reference.py      # [simple]  percent reference levels (§5.3.2), interpolated instants (§5.3.3)
├── _levels.py         # [simple]  state levels: histogram mode/mean (§5.2.1), peak, endpoints (§5.2.2)
├── _transitions.py    # [complex] vectorised hysteresis transition finder, the one shared by every feature
├── _aberrations.py    # [complex] overshoot/undershoot regions (§5.3.5), settling time (§5.3.7)
├── _pulses.py         # [simple]  pulses from alternating transitions: width, period, separation, duty (§5.4)
├── _multilevel.py     # [complex] scale-adaptive edge location and flat-level measurement for TDR traces
├── _plateau.py        # [simple]  flatness of a window
├── _api.py            # [complex] public functions: validate once, compose the internal modules
└── matlab.py          # [simple]  MATLAB-named facade over _api, returns numpy arrays, no logic of its own
tests/
├── signals.py         # deterministic signal generators shared by the tests (no I/O, so no fakes)
├── test__types.py
├── test__validate.py
├── test__reference.py
├── test__levels.py
├── test__transitions.py
├── test__aberrations.py
├── test__pulses.py
├── test__multilevel.py
├── test__plateau.py
├── test__api.py
├── test_matlab.py
└── test_performance.py   # timing clauses §C22–§C24, skipped unless PT_PERF=1
benchmarks/
└── benchmark.py       # [simple]  prints v0.3 timings beside the v0.2 figures from the brief spec
examples/
└── second_order_system.py  # [simple]  rewritten for the v0.3 API (closed-form step response, matplotlib plot)
pyproject.toml         # [simple]  version 0.3.0, requires-python >=3.10
requirements.txt       # [simple]  numpy only
```

The v0.2 modules (`common.py`, `impl.py`, `transient_response.py`, `matpulse.py`), the v0.2 tests and `examples/predicted_metrics.py` are deleted from this branch before stage 2; they remain on `develop`.

Run the tests with: `PYTHONPATH=src python -m unittest discover -s tests -t . -v`

## Third-party libraries

**Normal (used):**

- `numpy` (>=1.24): arrays, `histogram`, `searchsorted`, `flatnonzero`, `median`. The only runtime dependency.
- `matplotlib`: the example script only, not a package dependency.

**Dropped:** `scipy`. v0.2 used it for a Butterworth pre-filter, `find_peaks` on the histogram, `gaussian_filter1d` and `interp1d` resampling. None of those survive the redesign: hysteresis replaces filtering, §5.2.1 splitting replaces peak-finding, and non-uniform `x` is never resampled.

**Needs approval:** none.

**Looked at and rejected:**

- `python-control` (`control.step_info`): model-based step metrics with a 2 % band and 10-90 rise time, but no state-level histogram, no per-transition or pulse metrics, and a heavy dependency for one function.
- `stepanalyzer`: old, a thin extension of python-control.
- No other PyPI package implements IEEE 181 or the MATLAB pulse metrics; the closest match is this library's own v0.2.

## Dependency rules

- `_types.py`: stdlib, numpy. No internal imports.
- `_validate.py`: stdlib, numpy, `_types`.
- `_reference.py`: numpy, `_types`.
- `_levels.py`: numpy, `_types`.
- `_transitions.py`: numpy, `_types`, `_reference`.
- `_aberrations.py`: numpy, `_types`, `_reference`.
- `_pulses.py`: stdlib (`math`), `_types`.
- `_multilevel.py`: numpy, `_types`, `_reference`, `_transitions`.
- `_plateau.py`: numpy.
- `_api.py`: numpy, every module above except `matlab`.
- `matlab.py`: numpy, `_api`, `_types`. Never imports a private module other than these two.
- `__init__.py`: `_api`, `_types`, `matlab`.

Internal modules receive clean float64 arrays (validated by `_api` through `_validate`) and do no input checking of their own.

## Public API

All public functions take the waveform `y` first and everything else as keyword-only, so a v0.2-style `f(x, y)` call fails loudly instead of swapping axes. `x` and `fs` describe the time axis: give `x` (same length as `y`, strictly increasing, any spacing), or `fs` (then `x = arange(n) / fs`), or neither (then `x = arange(n)`). Reference levels are percentages `(lower, mid, upper)` of the amplitude, default `(10.0, 50.0, 90.0)`. `bounds=(start, end)` restricts the analysis to `start <= x <= end`; detected levels then come from that window.

Floating-point results in the examples are exact to within 1e-9 unless written with `≈`.

### Types (`_types.py`)

```python
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

class Polarity(IntEnum):
    FALLING = -1
    RISING = 1

class PulseTransitionsError(ValueError): ...           # base of every library error
class InputError(PulseTransitionsError): ...           # invalid argument; message names the argument
class LevelError(PulseTransitionsError): ...           # two state levels cannot be determined
class EdgeNotFoundError(PulseTransitionsError): ...    # an edge that must exist doesn't; message says what is missing

@dataclass(frozen=True)
class StateLevels:
    low: float
    high: float
    method: str                                          # "mode" | "mean" | "peak" | "endpoints"
    histogram: np.ndarray | None = field(default=None, compare=False, repr=False)   # counts, histogram methods only
    bin_edges: np.ndarray | None = field(default=None, compare=False, repr=False)
    @property
    def levels(self) -> tuple[float, float]: ...        # (low, high)
    @property
    def amplitude(self) -> float: ...                    # high - low

@dataclass(frozen=True)
class Transition:
    polarity: Polarity
    lower_instant: float      # instant of the lower reference level crossing
    mid_instant: float        # instant of the mid reference level crossing
    upper_instant: float      # instant of the upper reference level crossing
    lower_level: float        # absolute reference levels used
    mid_level: float
    upper_level: float
    start_index: int          # last sample in the pre-transition state
    end_index: int            # first sample in the post-transition state
    mid_method: str = "median"   # how mid_instant was chosen: "median" | "first" | "fit" (rev 5)
    @property
    def start(self) -> float: ...       # min(lower_instant, upper_instant)
    @property
    def end(self) -> float: ...         # max(lower_instant, upper_instant)
    @property
    def duration(self) -> float: ...    # end - start  (§5.3.4)
    @property
    def slew_rate(self) -> float: ...   # (upper_level - lower_level) / (upper_instant - lower_instant); negative when falling

@dataclass(frozen=True)
class Aberration:
    percent: float     # of |amplitude|; 0.0 when the extreme doesn't pass the state boundary; NaN when the region has no samples
    instant: float     # x of the extreme sample (first one on ties); NaN when the region is empty
    value: float       # y of the extreme sample; NaN when the region is empty

@dataclass(frozen=True)
class TransitionMetrics:
    transition: Transition
    levels: tuple[float, float]          # (low, high)
    pre_overshoot: Aberration
    pre_undershoot: Aberration
    post_overshoot: Aberration
    post_undershoot: Aberration
    settling_time: float                 # from mid_instant; NaN when not settled

@dataclass(frozen=True)
class Pulse:
    polarity: Polarity        # RISING: a positive pulse (rise then fall)
    first: Transition
    second: Transition
    width: float              # second.mid_instant - first.mid_instant
    period: float             # to the next pulse's first.mid_instant; NaN for the last pulse
    separation: float         # second.mid_instant to the next pulse's first.mid_instant; NaN for the last pulse
    duty_cycle: float         # width / period; NaN for the last pulse

@dataclass(frozen=True)
class EdgeMeasurement:
    transition: Transition
    levels: tuple[float, float]   # (bottom, top) the edge was measured between, whatever its direction
    @property
    def duration(self) -> float: ...   # transition.duration
```

Examples:

```python
Transition(Polarity.RISING, 1.1, 1.5, 1.9, 0.1, 0.5, 0.9, 1, 2).duration    # → 0.8 (≈, 1e-12)
Transition(Polarity.FALLING, 4.9, 4.5, 4.1, 0.1, 0.5, 0.9, 4, 5).slew_rate   # → -1.0 (≈)
Transition(Polarity.FALLING, 4.9, 4.5, 4.1, 0.1, 0.5, 0.9, 4, 5).start       # → 4.1
StateLevels(0.0, 2.0, "peak").amplitude                                       # → 2.0
issubclass(EdgeNotFoundError, ValueError)                                     # → True
```

### `reference_levels`

```python
def reference_levels(
    levels: tuple[float, float],
    reference: tuple[float, float, float] = (10.0, 50.0, 90.0),
) -> tuple[float, float, float]:
    """Absolute (lower, mid, upper) reference levels: low + (high - low) * p / 100  (§5.3.2)."""
```

```python
reference_levels((0.0, 2.0))                        # → (0.2, 1.0, 1.8)
reference_levels((-1.0, 1.0), (20.0, 50.0, 80.0))   # → (-0.6, 0.0, 0.6)
reference_levels((1.0, 1.0))                        # raises InputError  (§E4)
reference_levels((0.0, 1.0), (50.0, 10.0, 90.0))    # raises InputError  (§E5)
```

Internal helper in `_reference.py` (not exported), vectorised over arrays:

```python
def interpolate_instant(x0, y0, x1, y1, level) -> np.ndarray | float:
    """x0 + (x1 - x0) * (level - y0) / (y1 - y0); x0 where y1 == y0  (§5.3.3)."""
interpolate_instant(0.49, 0.4, 0.51, 0.6, 0.5)    # → 0.5 (≈)
interpolate_instant(2.0, 0.0, 3.0, 1.2, 0.9)      # → 2.75
interpolate_instant(1.0, 0.3, 2.0, 0.3, 0.3)      # → 1.0
```

### `state_levels`

```python
def state_levels(
    y: ArrayLike, *,
    x: ArrayLike | None = None, fs: float | None = None,
    bounds: tuple[float, float] | None = None,
    method: str = "mode",                      # "mode" | "mean" | "peak" | "endpoints"
    bins: int = 100,
    split: tuple[float, float] = (0.5, 0.5),   # §5.2.1 f1, f2
    endpoint_samples: int = 100,
) -> StateLevels:
```

- `mode`/`mean`: `bins` equal bins over `[y_min, y_max]` (`numpy.histogram`, last bin closed). Lower sub-histogram = bins `0 .. floor(f1*bins) - 1`, upper = bins `ceil(f2*bins) .. bins - 1`. `mode`: the level is the **mean of the samples in the most populated bin** of each sub-histogram (first bin on ties). `mean`: the mean of all samples in each sub-histogram.
- `peak`: `(y_min, y_max)`. `endpoints`: means of the first and last `endpoint_samples` samples, sorted into `(low, high)`.

```python
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0.5]
state_levels(y, bins=4)                  # → StateLevels(low=0.0, high=1.0, method="mode")
state_levels(y, bins=4, method="mean")   # → StateLevels(low=0.0, high=0.9166666666666666, method="mean")
state_levels([0, 3, -1, 2], method="peak")                          # → low=-1.0, high=3.0
state_levels([1, 1, 5, 0, 0], method="endpoints", endpoint_samples=2)  # → low=0.0, high=1.0
state_levels([2.0, 2.0, 2.0])            # raises LevelError  (§E7)
state_levels([1, 2, 3, 4, 5], method="endpoints", endpoint_samples=3)  # raises InputError  (§E9)
state_levels(y, method="median")         # raises InputError  (§E8)
```

With `bins=4` the bin edges are `[0, .25, .5, .75, 1]`: four 0s in bin 0, the 0.5 in bin 2, five 1s in bin 3.

### `transitions`

```python
def transitions(
    y: ArrayLike, *,
    x: ArrayLike | None = None, fs: float | None = None,
    levels: tuple[float, float] | None = None,        # None: state_levels(method="mode") of the analysed data
    reference: tuple[float, float, float] = (10.0, 50.0, 90.0),
    bounds: tuple[float, float] | None = None,
    polarity: Polarity | None = None,                  # None: both
    mid_method: str = "median",                        # "median" | "first" | "fit"
) -> tuple[Transition, ...]:
```

Algorithm (all vectorised):

1. `(yl, ym, yu) = reference_levels(levels, reference)`. State of each sample: `-1` if `y <= yl`, `+1` if `y >= yu`, else `0`.
2. Take the samples with non-zero state, in order. Every place where the state differs from the previous non-zero sample is a transition from sample `i` (`start_index`, last of the old state) to sample `j` (`end_index`, first of the new state). Polarity is the new state.
3. Lower and upper instants (rev 3). Each is the **median** crossing of its reference level within a guarded window, interpolated with `interpolate_instant`. Write `L0`/`L100` for the pre-/post-transition state levels (`low`/`high` when rising, swapped when falling).
   - Start-side level (lower when rising, upper when falling): crossings in segments `a .. j-1`, where `a` is the last sample at or before `i` that is at or beyond `L0` (`y <= low` when rising, `y >= high` when falling), not before the previous transition's `end_index`; `a = i` if there is none.
   - End-side level: crossings in segments `i .. b-1`, where `b` is the first sample at or after `j` at or beyond `L100`, not after the next transition's `start_index`; `b = j` if there is none.
   - A segment `k, k+1` crosses the lower level where `y > yl` changes, and the upper level where `y >= yu` changes (matching the state definitions). Each window holds an odd number of crossings.
   - Without state levels (`state_levels=None` in the internal call) `a = i` and `b = j`. On an edge that crosses each level once, every variant gives the same single crossing, which is §5.3.3's "crossing nearest the mid instant".
4. Mid-level crossings: within samples `i .. j`, every segment `k, k+1` where `y >= ym` changes is a crossing, interpolated. There is always an odd number of them (the edge starts on one side and ends on the other). `mid_method="median"` (default) takes the middle one; `"first"` takes the first (§5.3.3's stated default). On a clean edge there is one crossing and both agree.
5. (rev 5) `mid_method="fit"`: least-squares straight line `y = m*x + c` through samples `i .. j` inclusive (computed with `x` centred on its mean over those samples, for precision); `mid_instant = (ym - c) / m`, clamped to `[start, end]` (the transition's lower/upper instants, so §C9 holds). If `m` isn't finite or doesn't have the transition's sign (`m <= 0` rising, `m >= 0` falling), the median crossing of step 4 is used instead and the `Transition` records `mid_method="median"`; otherwise it records `"fit"`. The fit treats the edge as linear between the reference levels: it is exact on a straight ramp and on a 2-sample step (where it equals the interpolated crossing), and least affected by noise on a near-linear edge, but on a curved edge it differs from the crossing (an RC step `1 - exp(-t/τ)` sampled at 1/τ = 0.01: fit 80.6, crossing 69.3 = τ·ln 2 after the step). Lower and upper instants don't depend on `mid_method`. One loop over transitions is allowed here (§C25).

```python
y = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0]          # x = 0..10
transitions(y, levels=(0, 1))
# → (Transition(RISING,  lower_instant=1.1, mid_instant=1.5, upper_instant=1.9, 0.1, 0.5, 0.9, start_index=1, end_index=2),
#    Transition(FALLING, lower_instant=4.9, mid_instant=4.5, upper_instant=4.1, 0.1, 0.5, 0.9, start_index=4, end_index=5),
#    Transition(RISING,  6.1, 6.5, 6.9, 0.1, 0.5, 0.9, 6, 7),
#    Transition(FALLING, 9.9, 9.5, 9.1, 0.1, 0.5, 0.9, 9, 10))     (instants ≈ 1e-12)
transitions(y, levels=(0, 1), polarity=Polarity.FALLING)   # → the 2nd and 4th above
transitions(y)                                               # → same four (detected levels are exactly (0, 1))
transitions(y, fs=2.0, levels=(0, 1))[0].mid_instant         # → 0.75
transitions([0, 0, 0.6, 0, 0, 1, 1], levels=(0, 1))          # → one RISING transition, lower_instant 4.1 (the 0.6 runt is ignored)
transitions([1, 1, 1], levels=(0, 1))                        # → ()
noisy = [0, 0.45, 0.55, 0.45, 0.55, 1]                       # 50 % crossed at 1.5, 2.5, 3.5
transitions(noisy, levels=(0, 1))[0]                         # → lower_instant 0.2222222222, mid_instant 2.5, upper_instant 4.7777777778, mid_method "median"
transitions(noisy, levels=(0, 1), mid_method="first")[0].mid_instant   # → 1.5
dither = [0, 0.12, 0.08, 0.12, 0.5, 0.88, 0.92, 0.88, 1]     # 10 % crossed at 0.8333, 1.5, 2.5; 90 % at 5.5, 6.5, 7.1667
t = transitions(dither, levels=(0, 1))[0]                    # (rev 3)
t.lower_instant, t.upper_instant, t.start_index, t.end_index   # → 1.5, 6.5, 2, 6
transitions(noisy, levels=(0, 1), mid_method="fit")[0].mid_instant     # → 2.5 (rev 5; symmetric noise: fit and median agree)
asym = [0, 0.3, 0.4, 0.9, 1]                                 # i = 0, j = 3; line through (0,0), (1,.3), (2,.4), (3,.9): m = 0.28
transitions(asym, levels=(0, 1))[0].mid_instant              # → 2.2  (median crossing)
transitions(asym, levels=(0, 1), mid_method="fit")[0]        # → mid_instant 1.8571428571 (13/7), mid_method "fit"; lower_instant 0.3333333333, upper_instant 3.0 unchanged
transitions([0, 0, 1, 1], levels=(0, 1), mid_method="fit")[0].mid_instant   # → 1.5 (2-sample step: same as the crossing)
slump = [0.1, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.9]       # fitted slope -0.0195804196 < 0 on a rising edge
transitions(slump, levels=(0, 1), mid_method="fit")[0]       # → mid_instant 4.5, mid_method "median" (fallback)
early = [0, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 1]     # unclamped fit -0.55, before start 0.1123595506
transitions(early, levels=(0, 1), mid_method="fit")[0].mid_instant   # → 0.1123595506 (clamped to start = 0.1 / 0.89)
transitions(noisy, levels=(0, 1), mid_method="slope")        # raises InputError  (§E15)
transitions(y, levels=(1, 0))                                # raises InputError  (§E4)
```

### `transition_metrics` and `step_metrics`

```python
def transition_metrics(
    y: ArrayLike, *,
    x: ArrayLike | None = None, fs: float | None = None,
    levels: tuple[float, float] | None = None,
    reference: tuple[float, float, float] = (10.0, 50.0, 90.0),
    bounds: tuple[float, float] | None = None,
    aberration_factor: float = 3.0,     # region length in transition durations (§5.3.5)
    state_tolerance: float = 2.0,       # percent of |amplitude|: state boundary and settling band
    mid_method: str = "median",
) -> tuple[TransitionMetrics, ...]:

def step_metrics(...same parameters...) -> TransitionMetrics:
    """Metrics of the single transition in the analysed data. Raises EdgeNotFoundError when there is
    no transition or more than one; the message says how many were found and suggests `bounds=` or
    `transition_metrics()`."""
```

For each transition with duration `d`, amplitude `A = high - low`, tolerance `tol = state_tolerance / 100 * A`, direction `s = +1` rising / `-1` falling, pre-state level `Lpre` (low when rising) and post-state level `Lpost`:

- Pre region: `x` in `[start - aberration_factor*d, start]`, but not before the previous transition's `end` or the record's first sample. Post region: `[end, end + aberration_factor*d]`, not after the next transition's `start` or the last sample. Both inclusive.
- **Overshoot** is the excursion past the region's state level _in the direction of travel_; **undershoot** is the excursion past it _against_ the direction of travel. Overshoot extreme: the sample maximising `s*y`; undershoot extreme: the sample minimising `s*y` (first on ties). `percent = 100 * |y_ext - L| / |A|` if the extreme is beyond `L ± tol` on that side, else `0.0`. `instant`, `value` are the extreme sample's, even when `percent` is 0.
- **Settling time** (§5.3.7): band `Lpost ± tol`. Search samples from `start_index` up to (exclusive) the next transition's `start_index + 1`, or the end of the record. `k` = last sample outside the band. If `k` is the last sample searched: NaN (not settled). Otherwise interpolate the instant the waveform crosses the band edge nearest `y[k]` between samples `k` and `k+1`; settling time = that instant − `mid_instant`.

```python
y = [0, 0, 0, 1.2, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0]     # x = 0..11
m = step_metrics(y, levels=(0, 1))
m.transition.lower_instant, m.transition.mid_instant, m.transition.upper_instant  # → 2.0833333333, 2.4166666667, 2.75
m.post_overshoot      # → Aberration(percent=20.0, instant=3.0, value=1.2)      (≈ 1e-9)
m.post_undershoot     # → Aberration(percent=0.0, instant=4.0, value=1.0)
m.pre_overshoot       # → Aberration(percent=0.0, instant=1.0, value=0.0)
m.pre_undershoot      # → Aberration(percent=0.0, instant=1.0, value=0.0)
m.settling_time       # → 4.3833333333  (last out-of-band sample x=6 (0.9), band edge 0.98 crossed at 6.8)
```

Post region is `[2.75, 4.75]` (samples 3, 4); the 0.9 at x=6 equals the upper reference level, so it is still the high state and starts no new transition.

```python
step_metrics([1 - v for v in y], levels=(0, 1)).post_overshoot   # → Aberration(percent=20.0, instant=3.0, value=-0.2)  (falling: overshoot is below low)
transition_metrics([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], levels=(0, 1))   # settling_time of each of the 4 → 0.48 (≈)
step_metrics([0, 0, 0], levels=(0, 1))                             # raises EdgeNotFoundError  (§E10)
step_metrics([0, 0, 1, 1, 1, 0, 0], levels=(0, 1))                 # raises EdgeNotFoundError  (§E10, 2 transitions found)
step_metrics([0, 0, 1, 1, 1, 0, 0], levels=(0, 1), bounds=(0, 3)).transition.mid_instant   # → 1.5
step_metrics([0, 0, 1, 1], levels=(0, 1), aberration_factor=0)    # raises InputError  (§E15)
transition_metrics([0, 0, 0, 1, 1, 1.5], levels=(0, 1))[0].settling_time   # → NaN (last sample out of band)
```

### `pulses`

```python
def pulses(
    y: ArrayLike, *,
    x: ArrayLike | None = None, fs: float | None = None,
    levels: tuple[float, float] | None = None,
    reference: tuple[float, float, float] = (10.0, 50.0, 90.0),
    bounds: tuple[float, float] | None = None,
    polarity: Polarity = Polarity.RISING,
    mid_method: str = "median",
) -> tuple[Pulse, ...]:
```

Transitions strictly alternate (§C5). A pulse starts at every transition `i` of the given polarity that has a following transition `i+1`. `period = t[i+2].mid - t[i].mid` and `separation = t[i+2].mid - t[i+1].mid` when `t[i+2]` exists, else NaN; `duty_cycle = width / period`.

```python
y = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0]
pulses(y, levels=(0, 1))
# → (Pulse(RISING, T1, T2, width=3.0, period=5.0, separation=2.0, duty_cycle=0.6),
#    Pulse(RISING, T3, T4, width=3.0, period=nan, separation=nan, duty_cycle=nan))
pulses(y, levels=(0, 1), polarity=Polarity.FALLING)
# → (Pulse(FALLING, T2, T3, width=2.0, period=5.0, separation=3.0, duty_cycle=0.4),)
pulses([0, 0, 1, 1], levels=(0, 1))   # → ()   (a rise with no fall is not a pulse)
```

### `measure_edge`

```python
def measure_edge(
    y: ArrayLike, *,
    around: float,
    x: ArrayLike | None = None, fs: float | None = None,
    max_distance: float = math.inf,
    reference: tuple[float, float, float] = (10.0, 50.0, 90.0),
    levels: tuple[float, float] | None = None,   # None: medians of the flat parts either side
    mid_method: str = "median",
) -> EdgeMeasurement:
```

Algorithm (`_multilevel.py`):

1. Noise: `sigma = 1.4826 * median(|d - median(d)|) / sqrt(2)`, `d = diff(y)`.
2. Fewer than 16 samples: §E11. For scales `s = 1, 2, 4, …` while `s <= n // 8`: `D = y[s:] - y[:-s]`; steep where `|D| > 0` and `|D| >= max(8 * sqrt(2) * sigma, 0.1 * max|D|)`. Runs of consecutive steep positions with the same sign are candidate edges spanning samples `[first, last + s]`; same-sign runs closer than `s` samples merge. A run's **peak** is its largest `|D|`; its distance from `around` is 0 when `around` is inside its x-span.
3. Scale selection (rev 3). Walk up the scales, tracking `R_s`, the run nearest `around` at scale `s` when it lies within `max_distance`. Stop at the first scale `s` that has an `R_s` and where, at `2s`, any of these holds:
   - **resolved:** `R_2s` exists and its peak is less than 1.5 × the peak of `R_s` (doubling the span no longer raises the difference much: the whole edge fits in `s`);
   - **merged:** `R_2s` contains the centre of another run of scale `s` (going up would join this edge to a neighbour);
   - **lost:** there is no `R_2s`, or `2s > n // 8`.
   (rev 4) The resolved and merged tests apply only once the peak of `R_s` is at least 2 × the noise threshold `8 * sqrt(2) * sigma`. Below that the run is noise-limited (one slow edge shows as fragments or a lone noise spike), so the walk carries on; "lost" still applies.
   The chosen edge is `R_s`; its neighbours at the same scale bound its window. If no scale has an `R_s`: §E11. (Rev 2 took the smallest scale with any run near `around`, which locked onto noise fragments of slow edges.)
4. Levels (unless given): medians of the flat samples before and after the run, between it and the neighbouring run (or record end), limited to the `max(3 * run_length, 20)` samples nearest the run. Fewer than 3 samples on a side: §E12. If the after-median isn't beyond the before-median in the run's direction: §E13.
5. Run `transitions` on the window with those levels; return the transition of the run's polarity whose mid instant is nearest the run centre. None: §E13.

```python
x = np.arange(0, 14, 0.01)
y = np.clip((x - 2) / 2, 0, 1) + 2 * np.clip((x - 6) / 2, 0, 1)    # 0 → 1 at 2..4, 1 → 3 at 6..8
r = measure_edge(y, x=x, around=7.0)
r.levels                     # → (1.0, 3.0)   (≈ 1e-6)
r.transition.polarity        # → Polarity.RISING
r.transition.lower_instant, r.transition.upper_instant   # → 6.2, 7.8  (≈ 1e-6)
r.duration                   # → 1.6 (≈ 1e-6)
measure_edge(y, x=x, around=7.0, reference=(20, 50, 80)).duration   # → 1.2 (≈ 1e-6)
measure_edge(4 - y, x=x, around=7.0).transition.polarity            # → Polarity.FALLING, levels (1.0, 3.0)
measure_edge(y, x=x, around=11.0, max_distance=1.0)                 # raises EdgeNotFoundError  (§E11)
measure_edge(np.ones(100), around=50.0)                             # raises EdgeNotFoundError  (§E11)
```

### `flatness`

```python
def flatness(
    y: ArrayLike, *,
    bounds: tuple[float, float],
    x: ArrayLike | None = None, fs: float | None = None,
    normalize: bool = True,
) -> float:
    """Population std-dev of y within bounds; divided by |mean of the record's last 10 samples| when normalize.
    NaN when the window has fewer than 5 samples, or when normalizing by a final level below 1e-12."""
```

```python
y = np.full(20, 2.0); y[5:10] = [2.0, 2.2, 1.8, 2.0, 2.0]
flatness(y, bounds=(5, 9))                    # → 0.0632455532  (0.1264911064 / 2)
flatness(y, bounds=(5, 9), normalize=False)   # → 0.1264911064
flatness(y, bounds=(5, 8))                    # → nan (4 samples)
flatness(np.zeros(20), bounds=(0, 19))        # → nan (final level 0)
```

### `matlab` facade (`pulse_transitions.matlab`)

Every function except `statelevels` takes `(y, t=None, *, fs=None, ..., mid_method="median")` (`"median"`, `"first"` or `"fit"`) (`t` is the time vector), detects levels with `statelevels` unless `levels=` is given, and returns numpy float arrays. `reference` is MATLAB's `PercentReferenceLevels`, a pair `(lower, upper)`, default `(10, 90)`; the mid level is 50.

```python
def statelevels(y, nbins=100, method="mode") -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...  # (levels[2], histogram, bin_centers)
def midcross(y, t=None, *, fs=None, levels=None) -> np.ndarray: ...             # mid instants of all transitions
def risetime(y, t=None, *, fs=None, levels=None, reference=(10, 90)) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]: ...
    # (durations, lower instants, upper instants, lower level, upper level) of rising transitions
def falltime(...same...) -> ...: ...                                             # falling transitions; lower/upper as named
def slewrate(y, t=None, *, fs=None, levels=None, reference=(10, 90)) -> np.ndarray: ...   # signed, all transitions
def overshoot(y, t=None, *, fs=None, levels=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...   # post-overshoot (percent, value, instant)
def undershoot(...same...) -> ...: ...                                           # post-undershoot
def settlingtime(y, d, t=None, *, fs=None, levels=None) -> np.ndarray: ...       # d = tolerance percent
def pulsewidth(y, t=None, *, fs=None, levels=None, polarity="positive") -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...  # (widths, initcross, finalcross)
def pulseperiod(...same...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...  # (periods, initcross, finalcross, nextcross), complete periods only
def pulsesep(...same...) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...       # (separations, finalcross, nextcross), complete periods only
def dutycycle(...same...) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...    # (duty, initcross, finalcross, nextcross), complete periods only
```

```python
y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=float)
matlab.statelevels(y)[0]   # → array([0., 1.])
matlab.midcross(y)         # → array([1.5, 4.5, 6.5, 9.5])
matlab.risetime(y)         # → (array([0.8, 0.8]), array([1.1, 6.1]), array([1.9, 6.9]), 0.1, 0.9)
matlab.slewrate(y)         # → array([1., -1., 1., -1.])
matlab.settlingtime(y, 2)  # → array([0.48, 0.48, 0.48, 0.48])
matlab.pulsewidth(y)       # → (array([3., 3.]), array([1.5, 6.5]), array([4.5, 9.5]))
matlab.dutycycle(y)        # → (array([0.6]), array([1.5]), array([4.5]), array([6.5]))
matlab.pulsesep(y)         # → (array([2.]), array([4.5]), array([6.5]))
matlab.risetime(y, polarity="positive")  # raises TypeError  (§E14)
matlab.pulsewidth(y, polarity="up")      # raises InputError  (§E15)
```

(all arrays ≈ 1e-9)

## Internal interfaces

Each internal module has one entry point that its own test file calls directly, so it can be written and tested before `_api.py` exists. Internal functions receive validated float64 arrays and parameters; only `_validate` checks arguments.

```python
# _validate.py — raises InputError; the message names the argument
def as_waveform(y, x=None, fs=None, bounds=None) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) as 1-D float64 arrays: NaN pairs dropped (§C1), x = arange(n) / (fs or 1) when x is None (§C2),
    bounds applied inclusively (§C4). §E1, §E2, §E3, §E6."""
def check_levels(levels) -> tuple[float, float]: ...                    # §E4
def check_reference(reference, *, pair: bool = False) -> tuple[float, ...]: ...   # §E5; pair=True for the matlab (lower, upper) form
def check_choice(name: str, value, allowed) -> object: ...             # value if value in allowed, else InputError naming `name` (§E8, §E15)

# _reference.py
def reference_levels(levels: tuple[float, float], reference: tuple[float, float, float]) -> tuple[float, float, float]: ...  # no checks
def interpolate_instant(x0, y0, x1, y1, level): ...                    # as above

# _levels.py — raises LevelError for a constant record (§E7)
def detect_levels(y: np.ndarray, method: str = "mode", bins: int = 100,
                  split: tuple[float, float] = (0.5, 0.5), endpoint_samples: int = 100) -> StateLevels: ...

# _transitions.py
def find_transitions(x: np.ndarray, y: np.ndarray, ref_levels: tuple[float, float, float],
                     mid_method: str = "median",
                     state_levels: tuple[float, float] | None = None) -> tuple[Transition, ...]: ...
    # rev 3: _api and _multilevel always pass state_levels=(low, high); None narrows the windows to i .. j

# _aberrations.py
def metrics_for(x: np.ndarray, y: np.ndarray, transitions: tuple[Transition, ...], levels: tuple[float, float],
                aberration_factor: float = 3.0, state_tolerance: float = 2.0) -> tuple[TransitionMetrics, ...]: ...

# _pulses.py
def make_pulses(transitions: tuple[Transition, ...], polarity: Polarity = Polarity.RISING) -> tuple[Pulse, ...]: ...

# _multilevel.py — raises EdgeNotFoundError (§E11–§E13)
def locate_and_measure(x: np.ndarray, y: np.ndarray, around: float, max_distance: float,
                       reference: tuple[float, float, float], levels: tuple[float, float] | None,
                       mid_method: str = "median") -> EdgeMeasurement: ...

# _plateau.py — x, y are the whole record; bounds selects the window
def flatness_of(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float], normalize: bool = True) -> float: ...
```

The public `reference_levels` in `_api.py` validates with `check_levels`/`check_reference` and then calls `_reference.reference_levels`.

## Error contract

```
ValueError
└── PulseTransitionsError
    ├── InputError           # bad argument, message names it
    ├── LevelError           # levels cannot be determined
    └── EdgeNotFoundError    # a required edge is missing, message says what
```

`_validate.py` raises `InputError`; `_levels.py` may raise `LevelError` and `_multilevel.py` may raise `EdgeNotFoundError` directly, because each carries a reason only that module knows (an exception to "returns internally", accepted here). Every other internal module returns empty tuples or NaN.

| ID   | Raised by                                                                             | Exception         | Condition                                                                                                                            |
| ---- | ------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| §E1  | every public function                                                                 | InputError        | `y` not 1-D numeric, or fewer than 2 samples after NaN removal and bounds                                                            |
| §E2  | every function taking `x`/`fs`                                                        | InputError        | both `x` and `fs` given, or `fs` not finite and > 0                                                                                  |
| §E3  | every function taking `x`                                                             | InputError        | `len(x) != len(y)`, or `x` not strictly increasing after NaN removal                                                                 |
| §E4  | functions taking `levels`, `reference_levels`                                         | InputError        | `levels` not two finite numbers with `low < high`                                                                                    |
| §E5  | functions taking `reference`                                                          | InputError        | not three finite numbers with `0 <= lower < mid < upper <= 100` (matlab: two, `0 <= lower < upper <= 100`)                           |
| §E6  | functions taking `bounds`                                                             | InputError        | `bounds` not two finite numbers with `start < end`                                                                                   |
| §E7  | `state_levels`, and any function detecting levels                                     | LevelError        | `y` (within bounds) is constant                                                                                                      |
| §E8  | `state_levels`                                                                        | InputError        | unknown `method`, `bins < 2`, or `split` not `0 < f1 <= f2 < 1`                                                                      |
| §E9  | `state_levels`                                                                        | InputError        | `method="endpoints"` and `2 * endpoint_samples > n`, or `endpoint_samples < 1`                                                       |
| §E10 | `step_metrics` | EdgeNotFoundError | not exactly one transition in the analysed data (none, or more than one) |
| §E11 | `measure_edge`                                                                        | EdgeNotFoundError | no edge within `max_distance` of `around` (including traces with no edge or fewer than 16 samples)                                   |
| §E12 | `measure_edge`                                                                        | EdgeNotFoundError | fewer than 3 flat samples on one side of the edge                                                                                    |
| §E13 | `measure_edge`                                                                        | EdgeNotFoundError | the edge doesn't settle to a new level in its direction, or the reference levels aren't crossed                                      |
| §E14 | every public function                                                                 | TypeError         | unknown keyword argument (Python's own; no `**kwargs` anywhere)                                                                      |
| §E15 | `transition_metrics`, `step_metrics`, `pulses`, `transitions`, `measure_edge`, matlab functions | InputError | `aberration_factor <= 0`, `state_tolerance` not in `(0, 50)`, `polarity` not a `Polarity` (matlab: not `"positive"`/`"negative"`), or `mid_method` not `"median"`/`"first"`/`"fit"` (rev 5) |

## Constraints & edge cases

- **§C1** Samples where `y` or `x` is NaN are dropped (both arrays) before anything else. Indices in results refer to the cleaned, bounded arrays.
- **§C2** With neither `x` nor `fs`, `x = arange(n)` as float.
- **§C3** Non-uniform `x` is used as given; nothing is resampled. Instants on a non-uniform axis interpolate between the actual sample times.
- **§C4** `bounds` are inclusive. With `levels=None`, levels are detected from the bounded data only.
- **§C5** Transitions are returned in time order and strictly alternate polarity (before `polarity` filtering).
- **§C6** Hysteresis: a sample equal to the lower reference level is in the low state, equal to the upper is in the high state. An excursion that doesn't reach the opposite state's reference level makes no transition (runt). A record that starts or ends mid-transition yields no partial transition.
- **§C7** (rev 3) The lower and upper instants are the median crossings of their levels within the guarded windows of `transitions` step 3, so noise near a reference level doesn't bias them; on an edge that crosses each level once they are §5.3.3's crossings nearest the mid instant. The mid level is crossed an odd number of times there; the mid instant is the median crossing (`mid_method="median"`, default) or the first (`"first"`, §5.3.3's default), or (rev 5) the least-squares line's mid-level instant (`"fit"`, `transitions` step 5). Every result built on mid instants (pulse width, period, separation, duty cycle, settling time, `midcross`) uses the same method, and each `Transition` records the method actually used in `mid_method` (`"median"` where `"fit"` fell back).
- **§C8** `duration > 0`. `slew_rate` is positive for rising, negative for falling.
- **§C9** (rev 3) `start_index < end_index`, `start <= mid_instant <= end`, `start < x[start_index + 1]` and `end > x[end_index - 1]`. (A median start-side crossing may lie before `x[start_index]`.)
- **§C10** Scaling `y → a*y + b` with `a > 0` (and `levels` transformed the same way) leaves all instants unchanged; `a < 0` flips every polarity and leaves instants unchanged.
- **§C11** Shifting `x → x + c` shifts every instant by `c` and leaves durations unchanged.
- **§C12** Accuracy: a linear ramp gives exact instants (relative 1e-9) whatever its sub-sample position (offsets 0, 0.25, 0.5, 0.77 of a sample). An RC step `1 - exp(-t/τ)` gives a 10–90 duration of `τ·ln 9` within 1 % at 50 samples per τ and within 0.2 % at 200.
- **§C13** Histogram levels: `mode` is the mean of the samples in the modal bin of each sub-histogram (first bin on ties); `mean` is the mean of all samples in each sub-histogram. For a two-level signal with additive noise σ ≤ 2 % of the amplitude and each state ≥ 20 % of the record, `mode` levels are within 1 % of the amplitude of the true levels.
- **§C14** Aberration regions as defined under `transition_metrics`: inclusive, clipped at neighbouring transitions and the record ends. An empty region gives `Aberration(nan, nan, nan)`.
- **§C15** Overshoot is the excursion in the direction of travel, undershoot against it, both as positive percentages of `|A|`; `percent` is 0.0 unless the extreme passes `L ± tol`.
- **§C16** Settling time is measured from `mid_instant`, within the transition's own interval (up to the next transition); NaN if the last searched sample is still out of band.
- **§C17** Pulses: width, period, separation and duty cycle use mid instants. The last pulse (no following same-polarity transition) has NaN period, separation and duty cycle. A pulse never spans more than two transitions.
- **§C18** `measure_edge` finds an edge of any length from 1 sample to `n/10` samples, provided its height is at least 10 × the noise σ and at least 10 % of the largest edge in the record. Densely sampled slow edges are found: a 1e6-sample trace with one 2000-sample edge of height 1 and noise σ = 0.01 (seed 0) is measured with a 10–90 duration within 1 % of the noiseless value.
- **§C19** `measure_edge` levels are medians of the flat parts adjacent to the edge (step 4 of its algorithm), so a staircase with several levels measures each step between its own levels.
- **§C20** `measure_edge` returns the transition of the located run's polarity nearest the run centre; `levels` in the result is `(bottom, top)` for both directions.
- **§C21** `flatness` uses the population std-dev (`ddof=0`) and the last 10 samples of the whole (unbounded) record for normalisation.
- **§C22** Performance: `transitions` on a 1e6-sample, 20-transition pulse train with 1 % noise and given levels takes ≤ 100 ms (v0.2 `detect_edges`: 1009 ms). Test skipped unless `PT_PERF=1`.
- **§C23** Scaling: `transitions` time for 1e6 samples ≤ 15 × its time for 1e5 samples (same signal shape). Skipped unless `PT_PERF=1`.
- **§C24** `state_levels` on 1e6 samples ≤ 20 ms (v0.2: 14 ms). Skipped unless `PT_PERF=1`.
- Stage 3 of the pipeline runs `tests/test_performance.py` with `PT_PERF=1` on the machine the v0.2 figures were measured on and records the timings in `docs/spec/pipeline-log.md`; a miss there is a failure, not a skip.
- **§C25** No Python loop over samples anywhere in the package; loops over transitions, runs, scales or bins are fine. (Checked by §C22–§C24 and review.)
- **§C26** All result types are frozen dataclasses; array fields don't take part in equality.
- **§C27** The `matlab` facade adds no logic: each function's values equal those from the corresponding Pythonic call with the same levels and reference.
- **§C28** The library logs nothing above DEBUG and issues no warnings; "nothing found" is an empty tuple, NaN, or one of the documented exceptions.
- **§C29** `transitions`, `transition_metrics` and `pulses` return `()` when there is nothing to report; only `step_metrics` and `measure_edge` raise for a missing edge.
- **§C30** `benchmarks/benchmark.py` runs with `PYTHONPATH=src python benchmarks/benchmark.py`, uses the brief spec's pulse-train generator (10 periods, 20 edges, 1 % noise, seed 0, best of 3) at 1e4, 1e5, 1e6 samples, and prints v0.3 timings for `state_levels`, `transitions`, `step_metrics` and `measure_edge(around=0.2)` beside the v0.2 figures.
- **§C31** (rev 5) `mid_method="fit"` accuracy (`_transitions.py`). Exact (relative 1e-9) on a linear ramp at sub-sample offsets 0, 0.25, 0.5, 0.77, as §C12. On a noisy linear edge — `x = arange(4200)`, `y = clip((x - 1600) / 1000, 0, 1) + normal(0, 0.02)`, true mid instant 2100, levels `(0, 1)`, seeds 0–49 (`numpy.random.default_rng(seed)`) — every seed gives exactly one transition, and the RMS error of the fitted mid instant is ≤ 0.2 % of the 1000-sample rise and ≤ half the RMS error of the median crossing on the same records. Prototype: fit RMS 0.077 %, median 0.47 %, first 2.95 %, no fallbacks. Not a timing clause, but `transitions(..., mid_method="fit")` on the §C22 signal must also meet §C22's 100 ms (prototype: fit adds 0.5 ms to 57 ms).

## Calling interface

```python
import numpy as np
import pulse_transitions as pt
from pulse_transitions import matlab

t = np.linspace(0, 10e-6, 100_000)
y = (np.clip((t % 2e-6 - 0.4e-6) / 40e-9, 0, 1) - np.clip((t % 2e-6 - 1.2e-6) / 40e-9, 0, 1))

lv = pt.state_levels(y, x=t)                         # low ≈ 0.0, high ≈ 1.0 (modal bins hold a few ramp samples)
for tr in pt.transitions(y, x=t, levels=lv.levels):
    print(tr.polarity.name, tr.mid_instant, tr.duration, tr.slew_rate)

step = pt.step_metrics(y, x=t, bounds=(0, 1e-6))     # first rise only
print(step.post_overshoot.percent, step.settling_time)

for p in pt.pulses(y, x=t):
    print(p.width, p.period, p.duty_cycle)

edge = pt.measure_edge(y, x=t, around=0.42e-6, reference=(20, 50, 80))
print(edge.levels, edge.duration)

durations, lower, upper, ll, ul = matlab.risetime(y, t)
```

## Decisions (confirmed by the user, 2026-10-27)

- D1. Public functions take `y` first and everything else keyword-only (`x=`, `fs=`), so a v0.2-style `f(x, y)` call raises `TypeError` instead of silently swapping axes. The `matlab` facade keeps MATLAB's positional `(y, t)`.
- D2. The histogram `mode` level is the mean of the samples in the modal bin, not the bin centre (IEEE allows mode or mean; MATLAB uses the bin centre), so clean levels come out exact.
- D3. Overshoot is measured in the direction of travel and undershoot against it, for both regions and both polarities. The state boundary for the "else 0" rule is `L ± state_tolerance` (2 %), the band also used for settling.
- D4. `step_metrics` requires exactly one transition in the analysed data and raises `EdgeNotFoundError` otherwise (§E10); `bounds` selects the step.
- D5. scipy is dropped; numpy is the only runtime dependency. `CrossingDetectionSettings` and its pre-filter are gone; hysteresis between the reference levels rejects noise. The example shows filtering with scipy before measuring, for users who want it.
- D6. Exception hierarchy under `PulseTransitionsError(ValueError)`; `EdgeNotFoundError` is no longer an `IndexError`.
- D7. Before stage 2 the v0.2 modules, v0.2 tests and `examples/predicted_metrics.py` are deleted from the `pipeline-rewrite` branch (they remain on `develop`); `examples/second_order_system.py` is rewritten. The README is outside this run.
- D8. `measure_edge` uses the scale-adaptive detector described above, O(n log n). An edge smaller than 10 % of the largest edge in the record isn't found (as in v0.2).
- D9. `flatness` returns NaN instead of `None` and keeps v0.2's normalisation by the last 10 samples of the whole record.
- D10. Timing clauses §C22–§C24 are skipped unless `PT_PERF=1`; stage 3 must run them with it set and treats a miss as a failure.
- D11. The mid instant is the median of the mid-level crossings within the transition by default; `mid_method="first"` gives §5.3.3's stated default. Rev 5 adds the least-squares `"fit"` method (`transitions` step 5); the default stays `"median"`, because the fit is biased on curved edges.
- D12. `matlab.pulseperiod`, `pulsesep` and `dutycycle` return only pulses with a complete period, and the overshoot/undershoot functions report the post-transition region (MATLAB's defaults).
- D13. (Rev 5, confirmed 2026-10-27) `mid_method="fit"` fits samples `start_index .. end_index`, clamps the instant to `[start, end]`, falls back to the median crossing (recorded as `"median"`) when the slope is non-finite or has the wrong sign, and is not the default.

## Revisions

### Rev 2 — 2026-10-27
Reason: spec-to-tests needed entry points for the internal modules so each `[simple]` module can be tested on its own before `_api.py` exists.
Added: section "Internal interfaces" (no new or changed § IDs)   Changed: none   Withdrawn: none
Modules affected: _validate.py, _reference.py, _levels.py, _transitions.py, _aberrations.py, _pulses.py, _multilevel.py, _plateau.py (signatures only)

### Rev 3 — 2026-10-27
Reason: stage 3 showed §C18 unreachable as specified: measure_edge's smallest-scale rule locked onto noise fragments of slow edges (also failing the 1e4 benchmark pulse train), and last-≤10 % / first-≥90 % crossings shortened a noisy slow edge by 2.8–4.4 %. User chose median crossings for the lower/upper instants and a resolved-or-merged scale rule.
Added: none   Changed: §C7, §C9 (rev 3); transitions algorithm step 3; measure_edge algorithm steps 2–3; find_transitions gains `state_levels`   Withdrawn: none
Also (tests only): §C6 and §C15 need tests that kill the two surviving mutants (a sample equal to the lower level is low; an excursion within the tolerance reads 0).
Modules affected: _transitions.py, _multilevel.py, _api.py (passes state_levels)

### Rev 4 — 2026-10-27
Reason: stage 3 found Rev 3's merged/resolved stops firing on noise fragments of one slow edge (§C18 and the 1e4 pulse-train test failed). Loop guard hit on §C18; the user chose a noise-floor guard over peak finding or change-point detection.
Added: none   Changed: measure_edge algorithm step 3 (noise-floor guard); no § IDs   Withdrawn: none
Tests: unchanged (the existing §C18 and pulse-train tests cover it)
Modules affected: _multilevel.py

### Rev 5 — 2026-10-27
Reason: planned loop-back from D11: add `mid_method="fit"`, a least-squares mid instant for noisy near-linear edges.
Added: §C31; `transitions` algorithm step 5   Changed: §C7, §E15 (rev 5); `Transition.mid_method` values; `transitions` examples (`"fit"` no longer raises; `"slope"` does)   Withdrawn: none
Modules affected: _transitions.py (fit), _api.py and matlab.py (accept `"fit"`), _types.py (comment only)
