# Measured data for the examples

| File | Source | Contents |
| --- | --- | --- |
| `photoreceiver_pulses.csv` | Photoreceiver, 10 MΩ transimpedance, 850 nm pulsed source | 10 µs light pulses on two photodiodes, A and B, at small (0.5 V) and large (5 V for A, 10 V for B) output swing; 4 ns sampling. The first sample of every capture is a trigger glitch. |
| `tdr_terminations.csv` | TDR01 sampler, 10 ps spacing | One probe and line into open, short, 100 Ω, 50 Ω and waveguide terminations. The sampler's output is inverted: the incident step falls and an open steps down. |
| `rg58_50m_open.csv` | TDR01 sampler, 20 ps spacing | The open end of 50 m of RG58, seen through the cable: a skin-effect step with a long, slow tail. |

Voltages are as recorded; times are in the units the column names give.
