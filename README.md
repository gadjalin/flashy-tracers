# flashy-tracers

Fast and flexible tracer code for multi-dimensional hydrodynamics simulations.

This is an extension of the [flashy](https://github.com/gadjalin/flashy)
package, originally built for FLASH simulations.

## Usage

First set the variables in `config.py` then run `./integrate_tracers.py`.
You can also set most variables in `config.py` through the command line
directly (try `./integrate_tracers.py --help`).

## Extending

The code can in principle be extended to work with simulation output from
different codes.

This can be done by extending the `Progenitor`, `Snapshot` and `SnapshotProxy` classes to
read and store the data from the simulation output and initial conditions
(progenitor model).
The snapshot class should provide implementations for the abstract methods `get_quantity` and `get_field`. The first should
return a linear interpolation of the quantities on the grid at a given position, while the second
should provide the entire grid (used for tracer placement).

Then set the `MODEL_CLS`, `SNAP_CLS` and `PROXY_CLS` in `config.py` to point to your new classes.

## Performances

The integration of 10,000 tracers over 1,500 FLASH snapshots takes about 2hrs on 32
cores.

