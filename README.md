# flashy-tracers

Fast and flexible tracer code for multi-dimensional hydrodynamics simulations.

This is an extension of the [flashy](https://github.com/gadjalin/flashy)
package, originally built for FLASH simulations.

It is initially designed for backward integration of tracer particles in
core-collapse supernova simulations. The placement of tracers is optimised to
sample the ejected material uniformly in space, and the mass of each tracer is
adjusted to the local density of material.

## Usage

First set the variables in `config.py` then run `./integrate_tracers.py`.
You can also set most variables in `config.py` through the command line
directly (try `./integrate_tracers.py --help`).

## Extending

The code can in principle be extended to work with simulation output from
different codes, with minimal to no changes to the integration code itself.

This can be done by extending the `Progenitor`, `Snapshot` and `SnapshotProxy` classes to
read and store the data from the simulation output and initial conditions
(progenitor model).
The snapshot class should provide implementations for the abstract methods `get_quantity` and `get_field`. The first should
return a linear interpolation of the quantities on the grid at a given position, while the second
should provide the entire grid (used for tracer placement).
It should also handle shared memory (see `snapshot_flash.py`) buffers for the
parallel integration.
In general, the `Snapshot` class should contain the actual data and main shared
memory handles while the `SnapshotProxy` is meant to reconstruct the data from
the shared memory descriptor and provide the interpolation operations to the
parallel workers.

Then set the `MODEL_CLS`, `SNAP_CLS` and `PROXY_CLS` in `config.py` to point to your new classes.

The method used to calculate the unbound material may need to be adjusted to
match the method used in the simulation code.

New tracer placement methods can be implemented in `placement.py` and
dispatched in the `place_tracers` function in the main `integrate_tracers.py`
script.

## Design

Taking example on similar codes made by other people before me
(credits to Moritz Reichert, Max Witt, Max Jacobi, as well as Benedikt Weinhold's master thesis)
I have come up with what seems to me like a much more optimised design. The
result is a code that is at least x10 faster.

One of the key optimisation is a loader thread (a producer) that reads and
prepares the simulation snapshots (as `Snapshot` objects) concurrently with the tracer integration
workers. When a snapshot is loaded, it is pushed to a queue on the main thread.
`SnapshotProxy` objects are then dispatched to the workers to continue integrating,
while the producer is signaled to prepare the next snapshot in parallel. This
is the fastest and most memory efficient method.

Next is the interpolation method. Especially for codes like FLASH that use AMR
and have irregular grids, interpolating quantities require knowledge on the
refinement level and can become approximate or even give incorrect results at block and refinement boundaries.
Interpolating the entire domain to a finer, uniform grid and injecting it to a
scipy RegularGridInterpolator is easy but very slow and memory intensive.
Interpolating only inside a block may give problems on the edges unless guard
cells are implemented and using np.interp are scipy RegularGridInterpolator
can produce large overheads. Furthermore, this method still assumes an AMR block
structure (which works for FLASH but may be different for another code and
leaks dependencies on the grid structure in the interpolation methods).
Instead `snapshot_flash.py` implements guard cells and a direct linear interpolation, so that the integration
does not need to be aware of the grid structure and can more easily adapt to a
different grid as long as the complexity is hidden in the snapshot class.
Moreover, the `get_quantity` method can batch the interpolation of multiple variables
together and return the result for each quantity in an array.

The trajectories calculated by the worker processes are buffered (see `StateBuffer` in `buffer.py`) to mitigate I/O operations.
The buffer is flushed to the disk when its contents increases above `MAX_TRACER_BUF_SIZE`
set in `config.py`.

## Performances

The integration of 10,000 tracers over 1,500 FLASH snapshots takes about 3hrs on 32
cores.

