# SankhyaSutra Labs HPC Benchmark Suite

Description:
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
+ *scatter*: elements of the matrix at each grid point is moved to the same
  element at a neighboring point on the same grid

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

Data Size:
==========
1. **Memory**: This code allocates around **8GB of memory** by default for
**each MPI process**. To modify the memory allocated, please modify the
`GRIDNX` parameter, whose default values is set to 400. Memory allocated
increases in proportion to (`GRIDNX`)^3. So, for `GRIDNX=800`, around 32GB
memory will be allocated per MPI process.
2. **Storage**: This code writes the total data allocated in the MPI
processes to binary files during the Write kernel and reads them back
during the Read kernel.

Instructions:
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
