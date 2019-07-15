/*
===============================================================================
SSLABS HPC Benchmark : tests the performance of HPC systems
Copyright (C) 2019  SankhyaSutra Labs Pvt. Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact: sbhtta@sankhyasutralabs.com
===============================================================================
*/

#include <mpi.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/*
===============================================================================
Instructions:

1. Compilation using gcc and run:
   $ mpicc -O3 sslabs_hpc_benchmark.c -o benchmark
   $ mpirun -np 4 ./benchmark
===============================================================================
*/

// number of points along each axis
// in the grid for one MPI process
// should be two or more
#ifndef GRIDNX
  #define GRIDNX 200
#endif

#ifndef GRIDNY
  #define GRIDNY GRIDNX
#endif

#ifndef GRIDNZ
  #define GRIDNZ GRIDNX
#endif

// should be floating point
#ifndef VAR_TYPE
  #define VAR_TYPE double
#endif

#ifndef MPI_VAR_TYPE
  #define MPI_VAR_TYPE MPI_DOUBLE
#endif

// should be two or more
#ifndef NTIMES
  #define NTIMES 10
#endif

// should be two or more
#ifndef MEMALIGN
  #define MEMALIGN 4096
#endif

#ifndef NSTEPS
  #define NSTEPS 1
#endif

#define HLINE "-------------------------------------------------------------\n"

#ifndef MIN
  #define MIN(x,y) ((x)<(y)?(x):(y))
#endif

#ifndef MAX
  #define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#define DIM      3
#define GRIDNXY  (GRIDNX * GRIDNY)
#define GRIDNXZ  (GRIDNX * GRIDNZ)
#define GRIDNYZ  (GRIDNY * GRIDNZ)
#define W0       2./9.
#define WS       1./9.
#define WB       1./72.
#define ITHETA   3.
#define BETA2    1.
#define NROWS    4
#define NCOLS    4
#define NVARS    NROWS*NCOLS
#define NSTATS   4
#define NKERNELS 3

static char*  label[NKERNELS]   = {"Compute:   ",
                                   "Sync:      ",
                                   "Move:      "};
static double mem_ops[NKERNELS] = {3, 1, 2};
static double avgtime[NKERNELS] = {0};
static double maxtime[NKERNELS] = {0};
static double mintime[NKERNELS] = {FLT_MAX, FLT_MAX, FLT_MAX};
static size_t kernel_bytes[NKERNELS] =
  {
    sizeof(VAR_TYPE) * NVARS * GRIDNX * GRIDNY * GRIDNZ,
    sizeof(VAR_TYPE) * NVARS * (2 * (GRIDNXY + GRIDNXZ + GRIDNYZ) + 8 * (GRIDNX + GRIDNY + GRIDNZ) + 24),
    sizeof(VAR_TYPE) * NVARS * GRIDNX * GRIDNY * GRIDNZ,
  };

// computation coefficients
static VAR_TYPE w[NVARS]  = { W0, WS, WS, WS, WB, WB, WB, WB, 0., WS, WS, WS, WB, WB, WB, WB};
static VAR_TYPE c1[NVARS] = { 0.,+1., 0., 0.,-1.,+1.,-1.,+1., 0.,-1., 0., 0.,-1.,+1.,-1.,+1.};
static VAR_TYPE c2[NVARS] = { 0., 0.,+1., 0.,-1.,-1.,+1.,+1., 0., 0.,-1., 0.,-1.,-1.,+1.,+1.};
static VAR_TYPE c3[NVARS] = { 0., 0., 0.,+1.,-1.,-1.,-1.,-1., 0., 0., 0.,-1.,+1.,+1.,+1.,+1.};

void mpi_nprocs_from_args(int*, int, char**);
void initialize_grid_vars(const int, VAR_TYPE*);
void recompute_grid_vars(VAR_TYPE*);
void mpi_copy_faces_to_pads(MPI_Comm, VAR_TYPE*, VAR_TYPE*, VAR_TYPE*, VAR_TYPE*, VAR_TYPE*);
void stream_grid_vars(VAR_TYPE*);
void verify_results(VAR_TYPE*);

int
main(int argc, char** argv)
{
  // initialize MPI
  MPI_Init(&argc, &argv);
  int num_mpi_ranks, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_mpi_ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // get MPI process breakup from command line arguments
  int mpi_nprocs[DIM];
  mpi_nprocs_from_args(mpi_nprocs, argc, argv);

  // create MPI Cartesian topology
  MPI_Comm comm;
  int mpi_coords[DIM];
  int periodic[DIM] = {1, 1, 1};
  MPI_Cart_create(MPI_COMM_WORLD, DIM, mpi_nprocs, periodic, 1, &comm);
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Cart_coords(comm, mpi_rank, DIM, mpi_coords);

  // allocate aligned memory for grid of elements
  const int num_padded_vars = NVARS * (GRIDNX + 2) * (GRIDNY + 2) * (GRIDNZ + 2);
  const size_t alloc_grid_bytes = sizeof(VAR_TYPE) * num_padded_vars;
  const size_t memoffset = MEMALIGN - 1;
  VAR_TYPE* grid_unaligned = (VAR_TYPE*)malloc(alloc_grid_bytes + memoffset);
  VAR_TYPE* grid = (VAR_TYPE*)(((unsigned long)grid_unaligned + memoffset) & ~memoffset);

  // buffer sizes
  const int buffer_size_x = NVARS * (GRIDNY + 2) * (GRIDNZ + 2);
  const int buffer_size_y = NVARS * (GRIDNX + 2) * (GRIDNZ + 2);
  const int buffer_size_z = NVARS * (GRIDNX + 2) * (GRIDNY + 2);

  // allocate memory for send and receive buffers
  const int max_buffer_size = MAX(buffer_size_x, MAX(buffer_size_y, buffer_size_z));
  const size_t alloc_buffer_bytes = sizeof(VAR_TYPE) * max_buffer_size;
  VAR_TYPE* send_neg_buffer = (VAR_TYPE*)malloc(alloc_buffer_bytes);
  VAR_TYPE* send_pos_buffer = (VAR_TYPE*)malloc(alloc_buffer_bytes);
  VAR_TYPE* recv_neg_buffer = (VAR_TYPE*)malloc(alloc_buffer_bytes);
  VAR_TYPE* recv_pos_buffer = (VAR_TYPE*)malloc(alloc_buffer_bytes);

  // print pre-test summary
  if (mpi_rank == 0) {
    printf(HLINE);
    printf("SSLABS HPC Benchmark\n");
    printf("Copyright (C) 2019 SankhyaSutra Labs Pvt. Ltd.\n");
    printf(HLINE);
    int bytes_per_word = sizeof(VAR_TYPE);
    printf("This test uses:\n");
    printf("* %d bytes per variable\n", bytes_per_word);
    printf("* %d variables per grid point\n", NVARS);
    printf("* %d x %d x %d points per grid\n", GRIDNX, GRIDNY, GRIDNZ);
    printf("* %.2f GB data allocated per grid\n", 1.0E-9 * alloc_grid_bytes);
    printf("* %.2f GB data allocated per buffer\n", 1.0E-9 * alloc_buffer_bytes);
    printf("* two send buffers and two recv buffers per grid\n");
    printf("* one grid per MPI process\n");
    printf("* %d MPI processes in total\n", num_mpi_ranks);
    printf("* %.2f GB data allocated in total\n", num_mpi_ranks * 1.0E-9 *
                                                  (alloc_grid_bytes + 4 * alloc_buffer_bytes));
  }

  // main loop, run each kernel NTIMES
  double times[NKERNELS][NTIMES];
  for (size_t t = 0; t < NTIMES; t++) {

    // initialize elements at each grid point
    initialize_grid_vars(mpi_rank,grid);

    // local compute
    times[0][t] = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    recompute_grid_vars(grid);
    MPI_Barrier(MPI_COMM_WORLD);
    times[0][t] = MPI_Wtime() - times[0][t];

    // synchronize variables on pads using MPI communication
    times[1][t] = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    mpi_copy_faces_to_pads(comm, grid, send_neg_buffer, send_pos_buffer, recv_neg_buffer, recv_pos_buffer);
    MPI_Barrier(MPI_COMM_WORLD);
    times[1][t] = MPI_Wtime() - times[1][t];

    // data movement across neighbor points
    times[2][t] = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    stream_grid_vars(grid);
    MPI_Barrier(MPI_COMM_WORLD);
    times[2][t] = MPI_Wtime() - times[2][t];

  } // ends t loop

  // verify output
  verify_results(grid);

  // statistics for timings
  // ignore timing for the first iteration
  // timing for each kernel is assumed to be similar across each process
  // since MPI Barrier is called just before and after the kernels
  if (mpi_rank == 0) {
    for (int k = 0; k < NKERNELS; k++) {
      for (int t = 1; t < NTIMES; t++) {
        avgtime[k] = avgtime[k] + times[k][t];
        mintime[k] = MIN(mintime[k], times[k][t]);
        maxtime[k] = MAX(maxtime[k], times[k][t]);
      }
      avgtime[k] = avgtime[k]/(double)(NTIMES-1);
    }
  }

  // print summary of timing statistics
  if (mpi_rank == 0) {
    printf(HLINE);
    printf("Kernel      Best Rate MB/s  Avg time     Min time     Max time\n");
    printf(HLINE);
    for (int k = 0; k < NKERNELS; k++) {
      printf("%s%11.1f  %11.6f  %11.6f  %11.6f\n", label[k],
        1.0E-06 * mem_ops[k] * num_mpi_ranks * kernel_bytes[k] / mintime[k],
        avgtime[k],
        mintime[k],
        maxtime[k]);
    }
    printf(HLINE);
  }

  free(send_neg_buffer);
  free(send_pos_buffer);
  free(recv_neg_buffer);
  free(recv_pos_buffer);
  free(grid_unaligned);
  MPI_Finalize();
  return 0;
}

// parse arguments to get 3D distribution of MPI processors
// ========================================================
void
mpi_nprocs_from_args(int* mpi_nprocs, int argc, char** argv)
{
  int num_mpi_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_mpi_ranks);
  mpi_nprocs[0] = num_mpi_ranks;
  mpi_nprocs[1] = 1;
  mpi_nprocs[2] = 1;
  if (argc > 1) {
    if (argc < 4) {
      printf("Provide arguments: npx npy npz");
      MPI_Abort(MPI_COMM_WORLD, 2);
      exit(1);
    }
    mpi_nprocs[0] = atoi(argv[1]);
    mpi_nprocs[1] = atoi(argv[2]);
    mpi_nprocs[2] = atoi(argv[3]);
    if (mpi_nprocs[0] * mpi_nprocs[1] * mpi_nprocs[2] != num_mpi_ranks) {
      printf("Given MPI ranks don't add up to : npx * npy * npz");
      MPI_Abort(MPI_COMM_WORLD, 2);
      exit(1);
    }
  }
  return;
}

// flat index of element (r,c) of the matrix at grid point (x,y,z)
inline int
idx(const int r, const int c, const int x, const int y, const int z)
{ return c + NCOLS * (x + (GRIDNX + 2) * (y + (GRIDNY + 2) * (z + (GRIDNZ + 2) * r))); }

inline void
copy_vars_from_grid_point(const VAR_TYPE* grid, const int x, const int y, const int z, VAR_TYPE* var)
{
  for (int r = 0; r < NROWS; r++) {
    memcpy(&var[r*NCOLS], &grid[idx(r,0,x,y,z)], NCOLS * sizeof(VAR_TYPE));
  }
  return;
}

inline void
copy_vars_to_grid_point(VAR_TYPE* grid, const int x, const int y, const int z, VAR_TYPE* var)
{
  for (int r = 0; r < NROWS; r++) {
    memcpy(&grid[idx(r,0,x,y,z)], &var[r*NCOLS], NCOLS * sizeof(VAR_TYPE));
  }
  return;
}

static inline void
vars_to_stats(const VAR_TYPE* var, VAR_TYPE* s)
{
  s[0] = 0.;
  s[1] = 0.;
  s[2] = 0.;
  s[3] = 0.;
  for (int v = 0; v < NVARS; v++) {
    s[0] += var[v];
    s[1] += var[v] * c1[v];
    s[2] += var[v] * c2[v];
    s[3] += var[v] * c3[v];
  }
  const VAR_TYPE invs0 = 1. / s[0];
  s[1] *= invs0;
  s[2] *= invs0;
  s[3] *= invs0;
  return;
}

static inline void
stats_to_vars(const VAR_TYPE* s, VAR_TYPE* var)
{
  const VAR_TYPE sdots = (s[1]*s[1] + s[2]*s[2] + s[3]*s[3]) * ITHETA;
  for (int v = 0; v < NVARS; v++) {
    const VAR_TYPE cdots = (c1[v]*s[1] + c2[v]*s[2] + c3[v]*s[3]) * ITHETA;
    const VAR_TYPE coeff = 1 + cdots - 0.5 * sdots + 0.5 * cdots * cdots;
    var[v] = coeff * s[0] * w[v];
  }
  return;
}

static inline void
update_vars(VAR_TYPE* var)
{
  static VAR_TYPE tmpvar[NVARS];
  static VAR_TYPE s[NSTATS];
  vars_to_stats(var, s);
  stats_to_vars(s, tmpvar);
  for (int v = 0; v < NVARS; v++) {
    var[v] += BETA2 * (tmpvar[v] - var[v]);
  }
  return;
}

// initialize variables
// ====================
void
initialize_grid_vars(const int mpi_rank, VAR_TYPE* grid)
{
  VAR_TYPE tmpvar[NVARS];
  VAR_TYPE s[NSTATS] = {1.,0.,0.,0.};
  for (int z = 1; z <= GRIDNZ; z++)
  for (int y = 1; y <= GRIDNY; y++)
  for (int x = 1; x <= GRIDNX; x++)
  {
    stats_to_vars(s, tmpvar);
    copy_vars_to_grid_point(grid, x, y, z, tmpvar);
  }
  return;
}

// update variables
// ================
void
recompute_grid_vars(VAR_TYPE* grid)
{
  VAR_TYPE tmpvar[NVARS];
  for (int z = 1; z <= GRIDNZ; z++)
  for (int y = 1; y <= GRIDNY; y++)
  for (int x = 1; x <= GRIDNX; x++)
  {
    copy_vars_from_grid_point(grid, x, y, z, tmpvar);
    update_vars(tmpvar);
    copy_vars_to_grid_point(grid, x, y, z, tmpvar);
  }
  return;
}

// copy faces to pads
// ==================
void
pack_x_face_vars_to_buffer(const int side, const VAR_TYPE* grid, VAR_TYPE* buffer)
{
  int count = 0;
  for (int r = 0; r < NROWS; r++)
  for (int z = 0; z < GRIDNZ + 2; z++)
  for (int y = 0; y < GRIDNY + 2; y++)
  for (int c = 0; c < NCOLS; c++)
  {
    buffer[count++] = grid[idx(r, c, (side < 0 ? 1 : GRIDNX), y, z)];
  }
  return;
}

void
pack_y_face_vars_to_buffer(const int side, const VAR_TYPE* grid, VAR_TYPE* buffer)
{
  int count = 0;
  for (int r = 0; r < NROWS; r++)
  for (int z = 0; z < GRIDNZ + 2; z++)
  for (int x = 0; x < GRIDNX + 2; x++)
  for (int c = 0; c < NCOLS; c++)
  {
    buffer[count++] = grid[idx(r, c, x, (side < 0 ? 1 : GRIDNY), z)];
  }
  return;
}

void
pack_z_face_vars_to_buffer(const int side, const VAR_TYPE* grid, VAR_TYPE* buffer)
{
  int count = 0;
  for (int r = 0; r < NROWS; r++)
  for (int y = 0; y < GRIDNY + 2; y++)
  for (int x = 0; x < GRIDNX + 2; x++)
  for (int c = 0; c < NCOLS; c++)
  {
    buffer[count++] = grid[idx(r, c, x, y, (side < 0 ? 1 : GRIDNZ))];
  }
  return;
}

void
pack_face_vars_to_buffer(const int axis, const int side, const VAR_TYPE* grid, VAR_TYPE* buffer)
{
  switch (axis) {
  case 0 : return pack_x_face_vars_to_buffer(side, grid, buffer);
  case 1 : return pack_y_face_vars_to_buffer(side, grid, buffer);
  case 2 : return pack_z_face_vars_to_buffer(side, grid, buffer);
  }
}

void
unpack_buffer_to_x_pad_vars(const int side, VAR_TYPE* grid, const VAR_TYPE* buffer)
{
  int count = 0;
  for (int r = 0; r < NROWS; r++)
  for (int z = 0; z < GRIDNZ + 2; z++)
  for (int y = 0; y < GRIDNY + 2; y++)
  for (int c = 0; c < NCOLS; c++)
  {
    grid[idx(r, c, (side < 0 ? 0 : GRIDNX + 1), y, z)] = buffer[count++];
  }
  return;
}

void
unpack_buffer_to_y_pad_vars(const int side, VAR_TYPE* grid, const VAR_TYPE* buffer)
{
  int count = 0;
  for (int r = 0; r < NROWS; r++)
  for (int z = 0; z < GRIDNZ + 2; z++)
  for (int x = 0; x < GRIDNX + 2; x++)
  for (int c = 0; c < NCOLS; c++)
  {
    grid[idx(r, c, x, (side < 0 ? 0 : GRIDNY + 1), z)] = buffer[count++];
  }
  return;
}

void
unpack_buffer_to_z_pad_vars(const int side, VAR_TYPE* grid, const VAR_TYPE* buffer)
{
  int count = 0;
  for (int r = 0; r < NROWS; r++)
  for (int y = 0; y < GRIDNY + 2; y++)
  for (int x = 0; x < GRIDNX + 2; x++)
  for (int c = 0; c < NCOLS; c++)
  {
    grid[idx(r, c, x, y, (side < 0 ? 0 : GRIDNZ + 1))] = buffer[count++];
  }
  return;
}

void
unpack_buffer_to_pad_vars(const int axis, const int side, VAR_TYPE* grid, const VAR_TYPE* buffer)
{
  switch (axis) {
  case 0 : return unpack_buffer_to_x_pad_vars(side, grid, buffer);
  case 1 : return unpack_buffer_to_y_pad_vars(side, grid, buffer);
  case 2 : return unpack_buffer_to_z_pad_vars(side, grid, buffer);
  }
}

void
mpi_copy_faces_to_pads_along_axis(MPI_Comm comm,
                                  const int axis,
                                  VAR_TYPE* grid,
                                  const int buffer_size,
                                  VAR_TYPE* send_neg_buffer,
                                  VAR_TYPE* send_pos_buffer,
                                  VAR_TYPE* recv_neg_buffer,
                                  VAR_TYPE* recv_pos_buffer)
{
  int src_recv_neg, dest_send_neg;
  int src_recv_pos, dest_send_pos;
  int mpi_coords[DIM];
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Cart_coords(comm, mpi_rank, DIM, mpi_coords);
  MPI_Cart_shift(comm, 0, -1, &src_recv_pos, &dest_send_neg);
  MPI_Cart_shift(comm, 0, +1, &src_recv_neg, &dest_send_pos);
  MPI_Request request[4];
  MPI_Status  status[4];
  int tag = 0;

  MPI_Irecv(recv_neg_buffer, buffer_size, MPI_VAR_TYPE, src_recv_neg, tag, comm, &request[0]);
  MPI_Irecv(recv_pos_buffer, buffer_size, MPI_VAR_TYPE, src_recv_pos, tag, comm, &request[1]);

  pack_face_vars_to_buffer(axis, -1, grid, send_neg_buffer);
  pack_face_vars_to_buffer(axis, +1, grid, send_pos_buffer);

  MPI_Isend(send_neg_buffer, buffer_size, MPI_VAR_TYPE, dest_send_neg, tag, comm, &request[2]);
  MPI_Isend(send_pos_buffer, buffer_size, MPI_VAR_TYPE, dest_send_pos, tag, comm, &request[3]);

  MPI_Waitall(4, request, status);

  unpack_buffer_to_pad_vars(axis, -1, grid, recv_neg_buffer);
  unpack_buffer_to_pad_vars(axis, +1, grid, recv_pos_buffer);
  return;
}

void
mpi_copy_faces_to_pads(MPI_Comm comm,
                       VAR_TYPE* grid,
                       VAR_TYPE* send_neg_buffer,
                       VAR_TYPE* send_pos_buffer,
                       VAR_TYPE* recv_neg_buffer,
                       VAR_TYPE* recv_pos_buffer)
{
  const int buffer_size_x = NVARS * (GRIDNY + 2) * (GRIDNZ + 2);
  const int buffer_size_y = NVARS * (GRIDNX + 2) * (GRIDNZ + 2);
  const int buffer_size_z = NVARS * (GRIDNX + 2) * (GRIDNY + 2);
  mpi_copy_faces_to_pads_along_axis(comm, 0, grid, buffer_size_x, send_neg_buffer, send_pos_buffer, recv_neg_buffer, recv_pos_buffer);
  mpi_copy_faces_to_pads_along_axis(comm, 1, grid, buffer_size_y, send_neg_buffer, send_pos_buffer, recv_neg_buffer, recv_pos_buffer);
  mpi_copy_faces_to_pads_along_axis(comm, 2, grid, buffer_size_z, send_neg_buffer, send_pos_buffer, recv_neg_buffer, recv_pos_buffer);
  return;
}

// stream variables
// ================
void
stream_grid_vars(VAR_TYPE* grid)
{
  for (int z = 1; z <= GRIDNZ; z++)
  for (int y = 1; y <= GRIDNY; y++)
  for (int x = 1; x <= GRIDNX; x++)
  {
    grid[idx(0,0,x,y,z)] = grid[idx(0,0,x,y,z)];
    grid[idx(0,1,x,y,z)] = grid[idx(0,1,x+1,y,z)];
    grid[idx(0,2,x,y,z)] = grid[idx(0,2,x,y+1,z)];
    grid[idx(0,3,x,y,z)] = grid[idx(0,3,x,y,z+1)];
  }
  for (int z = 1; z <= GRIDNZ; z++)
  for (int y = 1; y <= GRIDNY; y++)
  for (int x = 1; x <= GRIDNX; x++)
  {
    grid[idx(1,0,x,y,z)] = grid[idx(1,0,x-1,y-1,z+1)];
    grid[idx(1,1,x,y,z)] = grid[idx(1,1,x+1,y-1,z+1)];
    grid[idx(1,2,x,y,z)] = grid[idx(1,2,x-1,y+1,z+1)];
    grid[idx(1,3,x,y,z)] = grid[idx(1,3,x+1,y+1,z+1)]; 
  }
  for (int z = GRIDNZ; z >= 1; z--)
  for (int y = GRIDNY; y >= 1; y--)
  for (int x = GRIDNX; x >= 1; x--)
  {
    grid[idx(2,0,x,y,z)] = grid[idx(2,0,x,y,z)];
    grid[idx(2,1,x,y,z)] = grid[idx(2,1,x-1,y,z)];
    grid[idx(2,2,x,y,z)] = grid[idx(2,2,x,y-1,z)];
    grid[idx(2,3,x,y,z)] = grid[idx(2,3,x,y,z-1)];
  }
  for (int z = GRIDNZ; z >= 1; z--)
  for (int y = GRIDNY; y >= 1; y--)
  for (int x = GRIDNX; x >= 1; x--)
  {
    grid[idx(3,0,x,y,z)] = grid[idx(3,0,x-1,y-1,z-1)];
    grid[idx(3,1,x,y,z)] = grid[idx(3,1,x+1,y-1,z-1)];
    grid[idx(3,2,x,y,z)] = grid[idx(3,2,x-1,y+1,z-1)];
    grid[idx(3,3,x,y,z)] = grid[idx(3,3,x+1,y+1,z-1)];
  }
  return;
}

void
verify_results(VAR_TYPE* grid)
{
  for (int z = 1; z <= GRIDNZ; z++)
  for (int y = 1; y <= GRIDNY; y++)
  for (int x = 1; x <= GRIDNX; x++)
  {
    VAR_TYPE tmpvar[NVARS] = {0};
    VAR_TYPE s[NSTATS] = {-99.,-99.,-99.,-99.};
    copy_vars_from_grid_point(grid,x,y,z,tmpvar);
    vars_to_stats(tmpvar, s);
    int found_error = 0;
    if (fabs(s[0] - 1.0) > 1.0E-6) found_error = 1;
    if (fabs(s[1] - 0.0) > 1.0E-6) found_error = 1;
    if (fabs(s[2] - 0.0) > 1.0E-6) found_error = 1;
    if (fabs(s[3] - 0.0) > 1.0E-6) found_error = 1;
    if (found_error == 1) {
      printf("Error found during verification of results.");
      MPI_Abort(MPI_COMM_WORLD, 2);
      exit(1);
    }
  }
  return;
}
