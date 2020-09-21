# SankhyaSutra Labs HPC Benchmark Suite

Description
============
This benchmark runs simulates a system represented as a 3D grid with a 4x4
matrix at each point of the grid. The grid is domain decomposed across MPI
processes arranged in a 3D Cartesian MPI topology. The matrices on the grid
are initialized to certain values at the start of a simulation run and are
processed for a given number of steps before being written into binary files.
At each step of the run, the following kernels are used:
+ *update*: the matrix at each grid point is updated using the values of it's
  own elements
+ *sync*: the matrices on the layer of points at each face of the grid
  portion belonging to an MPI process are copied onto the layer of padding
  points on the grid portion adjacent to that face; since each MPI process
  contains one portion of the grid, this copy to neighboring grid portions
  residing on other MPI processes is done via MPI send and receive calls.
+ *scatter*: elements of the matrix at each grid point are moved to the
  corresponding element at a neighboring point on the same grid portion

Benchmark Timings
=================
Walltime statistics are printed at the end of the tests for these kernels:
1. Update
2. Sync
3. Scatter
4. Write
5. Read

+ The **simulation time** is the sum of times for Update+Sync+Scatter
+ The **checkpoint time** is the time for Write

MPI Processes
=============
The amount of data processed by the Update and Scatter kernels per second
(GB/s) on a node increases in proportion to the memory bandwidth that can
be utilized. If a node has **N cores** in total, the maximum sustained
memory bandwidth is expected to be realized when using **N MPI processes**
on that node. In some cases, the maximum bandwidth value can be saturated
using fewer than N MPI processes on that node.

Data Size
==========
1. **Memory**: This code allocates around **8 GB of memory** by default for
**each MPI process**. To modify the memory allocated, please modify the
`GRIDNX` parameter, whose default value is set to 400. Memory allocated
increases in proportion to (`GRIDNX`)^3. So, for `GRIDNX=800`, around 64 GB
memory will be allocated per MPI process.
2. **Storage**: This code writes the total data allocated in the MPI
processes to binary files during the Write kernel and reads them back
during the Read kernel.

Instructions
=============
1. Sample compilation and run using, for example, four MPI processes:
```
$ mpicc -O3 sslabs_hpc_benchmark.c -o benchmark
$ mpirun -np 4 ./benchmark
```
2. Using 800 points along each axis and 4 MPI processes:
```
$ mpicc -O3 sslabs_hpc_benchmark.c -o benchmark -DGRIDNX=800
$ mpirun -np 4 ./benchmark
```
3. For a quick run simulating a small problem size, use 100 points
along each axis and 10 simulation steps:
```
$ mpicc -O3 sslabs_hpc_benchmark.c -o benchmark -DGRIDNX=100 -DNSTEPS=10
$ mpirun -np 4 ./benchmark
```
4. To turn off IO test, use the compilation flag `-DNO_IO_TEST`:
```
$ mpicc -O3 sslabs_hpc_benchmark.c -o benchmark -DNO_IO_TEST
$ mpirun -np 4 ./benchmark
```

Sample Timings for a Quick Test
===============================

- using AMD AOCC 2.2 and MPICH 3.3
- on a node with 2S EPYC 7742 and 16 x 32 GB 3.2 GHz DDR4 RAM
- using a `200x200x200` grid block per process
- 1 simulation step per sample
- no IO test

Compilation
-----------

```sh
#!/bin/bash

cc_macros="-DGRIDNX=200 -DNSTEPS=1 -DNO_IO_TEST"
cc_flags="-std=c99 -DNDEBUG -O3 -mcmodel=large -mavx2 -march=znver2 -fnt-store=aggressive"
MPICH_CC=clang mpicc $cc_flags sslabs_hpc_benchmark.c -o benchmark $cc_macros
```

Run
---

```
mpirun -n 16 -bind-to core:8 ./benchmark
```

Output
------

```
--------------------------------------------------------------------------
SSLABS HPC Benchmark
Copyright (C) 2019 SankhyaSutra Labs Pvt. Ltd.
--------------------------------------------------------------------------
This test uses:
(a) 8 bytes per variable
(b) 4 x 4 = 16 variables per matrix
(c) 1 matrix per grid point
(d) 200 x 200 x 200 points per grid portion
(e) 128000000 variables per grid portion
(f) 1 extra layer of padding points beside each face of a grid portion
(g) 1.06 GB data allocated per padded grid portion
(h) 0.01 GB data allocated per buffer
(i) two send buffers and two recv buffers per grid portion
(j) one grid portion per MPI process
(k) 16 x 1 x 1 = 16 MPI processes in total
(l) 16.88 GB data allocated for padded grid
(m) 17.21 GB data allocated in total (padded grid + buffers)
(n) 1 simulation steps in a run
(o) 10 runs for generating walltime statistics
--------------------------------------------------------------------------
--------------------------------------------------------------------------
Kernel     Avg time     Min time     Max time
--------------------------------------------------------------------------
Update     0.257627     0.255894     0.260912
Sync       0.033473     0.032440     0.034372
Scatter    0.119065     0.118899     0.119535
Write      0.000028     0.000015     0.000045
Read       0.000024     0.000015     0.000037
--------------------------------------------------------------------------
```
