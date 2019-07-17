# SankhyaSutra Labs HPC Benchmark Suite

Description:
This benchmark runs simulates a system represented as a 3D grid with a 4x4
matrix at each point of the grid. The grid is domain decomposed across MPI
processes arranged in a 3D Cartesian MPI topology. The matrices on the grid
are initialized to certain values at the start of a simulation run and at
processed for a given number of steps before being written into binary files.
At each step of the run, the following kernels are used:
a) update : the matrix at each grid point is updated using the values of it's
   own elements
b) sync   : the matrices on the layer of points at each face of the grid
   portion belonging to an MPI process are copied onto the layer of padding
   points on the grid portion adjacent to that face; since each MPI process
   contains one portion of the grid, this copy to neighboring grid portions
   residing on other MPI processes is done via MPI send and receive calls.
c) scatter: elements of the matrix at each grid point is moved to the same
   element at a neighboring point on the same grid

Walltime statistics are printed at the end of the tests for these kernels:
1. Update
2. Sync
3. Scatter
4. Write
5. Read

The simulation time is the sum of times for Update+Sync+Scatter. The
checkpoint time is the time for Write.

NOTE: This code allocates around 4GB of memory by default for each MPI
process. To modify the memory allocated, please modify the GRIDNX
parameter, whose default values is set to 200. Memory allocated
increases in proportion to (GRIDNX)^3. So, for GRIDNX=400, 32GB memory
will be allocated per MPI process.
   
Instructions:
1. Compilation using gcc and default run using four MPI processes:
   $ mpicc -O3 sslabs_hpc_benchmark.c -o benchmark
   $ mpirun -np 4 ./benchmark
2. Using 400 points along each axis and 4 MPI processes:
   $ mpicc -O3 sslabs_hpc_benchmark.c -o benchmark -DGRIDNX=400
   $ mpirun -np 4 ./benchmark
