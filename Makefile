# Rebuild the example figures (examples/figures/<script>.svg) whenever an example script,
# the shared plotting helper, the measured data or the library changes.

PYTHON ?= python3
RUN := PYTHONPATH=src MPLBACKEND=Agg $(PYTHON)

EXAMPLES := $(filter-out examples/_plot.py examples/__init__.py,$(wildcard examples/*.py))
FIGURES := $(patsubst examples/%.py,examples/figures/%.svg,$(EXAMPLES))
LIBRARY := $(wildcard src/pulse_transitions/*.py)
DATA := $(wildcard examples/data/*.csv)

.PHONY: all examples test clean-examples

all: examples

examples: $(FIGURES)

examples/figures/%.svg: examples/%.py examples/_plot.py $(LIBRARY) $(DATA)
	$(RUN) $<

test:
	$(RUN) -m unittest discover -s tests -t .

clean-examples:
	rm -f $(FIGURES)
