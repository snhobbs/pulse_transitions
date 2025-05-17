# pulse_transitions
This library implements a python version of the [Pulse and Transition Metrics function category](https://www.mathworks.com/help/signal/pulse-and-transition-metrics.html?s_tid=CRUX_lftnav) in Matlab.


| Function                                                                    | Description                                                       |                  |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------- |
| [dutycycle](https://www.mathworks.com/help/signal/ref/dutycycle.html)       | Duty cycle of pulse waveform                                      |                  |
| [midcross](https://www.mathworks.com/help/signal/ref/midcross.html)         | Mid-reference level crossing for bilevel waveform                 |                  |
| [pulseperiod](https://www.mathworks.com/help/signal/ref/pulseperiod.html)   | Period of bilevel pulse                                           |                  |
| [pulsesep](https://www.mathworks.com/help/signal/ref/pulsesep.html)         | Separation between bilevel waveform pulses                        |                  |
| [pulsewidth](https://www.mathworks.com/help/signal/ref/pulsewidth.html)     | Bilevel waveform pulse width                                      |                  |
| [statelevels](https://www.mathworks.com/help/signal/ref/statelevels.html)   | State-level estimation for bilevel waveform with histogram method |                  |
| [falltime](https://www.mathworks.com/help/signal/ref/falltime.html)         | Fall time of negative-going bilevel waveform transitions          |                  |
| [overshoot](https://www.mathworks.com/help/signal/ref/overshoot.html)       | Overshoot metrics of bilevel waveform transitions                 |                  |
| [risetime](https://www.mathworks.com/help/signal/ref/risetime.html)         | Rise time of positive-going bilevel waveform transitions          |                  |
| [settlingtime](https://www.mathworks.com/help/signal/ref/settlingtime.html) | Settling time for bilevel waveform                                |                  |
| [slewrate](https://www.mathworks.com/help/signal/ref/slewrate.html)         | Slew rate of bilevel waveform                                     |                  |
| [undershoot](https://www.mathworks.com/help/signal/ref/undershoot.html)     | Undershoot metrics of bilevel waveform transitions                | ([MathWorks][1]) |

[1]: https://www.mathworks.com/help/signal/ref/dutycycle.html?utm_source=chatgpt.com "dutycycle - Duty cycle of pulse waveform - MATLAB - MathWorks"



## Background & Resources
### HP Journal
There's an excellent description of the algorithms used by HP [here](https://hparchive.com/Journals/HPJ-1996-12.pdf).

### Control Library
+ [GitHub](https://github.com/python-control/python-control) 
+ [Docs](https://python-control.readthedocs.io)
