/*
===============================================================================
SSLABS HPC Saturate : saturate HPC system performance
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

#ifndef GRIDNX
  #define GRIDNX 300
#endif

#ifndef GRIDNY
  #define GRIDNY GRIDNX
#endif

#ifndef GRIDNZ
  #define GRIDNZ GRIDNX
#endif

// if you change this type, change MPI_VAR_TYPE as well
#ifndef VAR_TYPE
  #define VAR_TYPE double
#endif

// if you change this type, change VAR_TYPE as well
#ifndef MPI_VAR_TYPE
  #define MPI_VAR_TYPE MPI_DOUBLE
#endif

#ifndef NTIMES
  #define NTIMES 400
#endif

#ifndef MEMALIGN
  #define MEMALIGN 4096
#endif

#ifndef NSTEPS
  #define NSTEPS 100
#endif

#ifndef NCOMPUTE
  #define NCOMPUTE 1
#endif

#ifndef NFILES
  #define NFILES 45
#endif

// uncomment below to skip read-write IO tests
//#define NO_IO_TEST

#define HLINE "--------------------------------------------------------------------------\n"

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
#define U0       0.04
#define BETA2    1.
#define NROWS    4
#define NCOLS    4
#define NVARS    NROWS*NCOLS
#define NSTATS   4
#define NKERNELS 5

// info for kernels
static double times[NKERNELS][NTIMES];
static double avgtime[NKERNELS] = {0};
static double maxtime[NKERNELS] = {0};
static double mintime[NKERNELS] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
static char*  label[NKERNELS]   = {"Update  ", "Sync    ", "Scatter ", "Write   ", "Read    "};

// computation coefficients
static VAR_TYPE w[NVARS]  = { W0, WS, WS, WS, WB, WB, WB, WB, 0., WS, WS, WS, WB, WB, WB, WB};
static VAR_TYPE c1[NVARS] = { 0.,+1., 0., 0.,-1.,+1.,-1.,+1., 0.,-1., 0., 0.,-1.,+1.,-1.,+1.};
static VAR_TYPE c2[NVARS] = { 0., 0.,+1., 0.,-1.,-1.,+1.,+1., 0., 0.,-1., 0.,-1.,-1.,+1.,+1.};
static VAR_TYPE c3[NVARS] = { 0., 0., 0.,+1.,-1.,-1.,-1.,-1., 0., 0., 0.,-1.,+1.,+1.,+1.,+1.};

void check_grid_parameters(const int);
void print_summary(MPI_Comm, const size_t, const size_t);
void print_timings_oneline(const int rank, const int t);
void print_timings(const int);
void initialize(MPI_Comm, VAR_TYPE*);
void verify(MPI_Comm, VAR_TYPE*);
void update(VAR_TYPE*);
void sync(MPI_Comm, VAR_TYPE*, VAR_TYPE*, VAR_TYPE*, VAR_TYPE*, VAR_TYPE*);
void scatter(VAR_TYPE*);
void write_grid(const int, VAR_TYPE*, const int nfile, const int node_id);
void read_grid(const int, VAR_TYPE*, const int nfile, const int node_id);

int
main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  // from command line arguments, get the arrangement of
  // MPI processes and create 3D Cartesian MPI topology
  MPI_Comm comm;
  int rank;
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  int nprocs[DIM];
  int periodic[DIM] = {1, 1, 1};
  nprocs[0] = num_ranks;
  nprocs[1] = 1;
  nprocs[2] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, DIM, nprocs, periodic, 1, &comm);
  MPI_Comm_rank(comm, &rank);

  // get node ID from command line arguments
  if (argc != 2) {
    if (rank == 0) {
      printf("Error : Please provide a node ID (integer) as command line argument\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(1);
  }

  const int node_id = atoi(argv[1]);


  check_grid_parameters(rank);

  // allocate aligned memory for a padded grid of variables with
  // one extra padding point on either side of the grid along each axis
  const int num_padded_pts = (GRIDNX + 2) * (GRIDNY + 2) * (GRIDNZ + 2);
  const int num_padded_vars = NVARS * num_padded_pts;
  const size_t grid_bytes = sizeof(VAR_TYPE) * num_padded_vars;
  const size_t moffset = MEMALIGN - 1;
  VAR_TYPE* ua_grid = (VAR_TYPE*)malloc(grid_bytes + moffset);
  VAR_TYPE* grid = (VAR_TYPE*)(((unsigned long)ua_grid + moffset) & ~moffset);

  // calculate size of buffers for transfer data from each
  // face of the grid and allocate memory for each buffer
  const int size_x   = NVARS * (GRIDNY + 2) * (GRIDNZ + 2);
  const int size_y   = NVARS * (GRIDNX + 2) * (GRIDNZ + 2);
  const int size_z   = NVARS * (GRIDNX + 2) * (GRIDNY + 2);
  const int max_size = MAX(size_x, MAX(size_y, size_z));
  const size_t buffer_bytes = sizeof(VAR_TYPE) * max_size;
  VAR_TYPE* send_neg = (VAR_TYPE*)malloc(buffer_bytes);
  VAR_TYPE* send_pos = (VAR_TYPE*)malloc(buffer_bytes);
  VAR_TYPE* recv_neg = (VAR_TYPE*)malloc(buffer_bytes);
  VAR_TYPE* recv_pos = (VAR_TYPE*)malloc(buffer_bytes);

  int tmp_idx = 0;
  for (int r = 0; r < NROWS; r++)
  for (int z = 0; z <= GRIDNZ+1; z++)
  for (int y = 0; y <= GRIDNY+1; y++)
  for (int x = 0; x <= GRIDNX+1; x++)
  for (int c = 0; c < NCOLS; c++)
  {
    grid[tmp_idx++] = 0;
  }

  for (int k = 0; k < NKERNELS; k++) {
    for (int t = 0; t < NTIMES; t++) {
      times[k][t] = 0;
    }
  }

  print_summary(comm, grid_bytes, buffer_bytes);

  // main loop
  double seconds;
  for (size_t t = 0; t < NTIMES; t++) {

    initialize(comm, grid);

    for (int step = 0; step < NSTEPS; step++) {
      seconds = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD);
      update(grid);
      MPI_Barrier(MPI_COMM_WORLD);
      times[0][t] += MPI_Wtime() - seconds;

      seconds = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD);
      sync(comm, grid, send_neg, send_pos, recv_neg, recv_pos);
      MPI_Barrier(MPI_COMM_WORLD);
      times[1][t] += MPI_Wtime() - seconds;

      seconds = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD);
      scatter(grid);
      MPI_Barrier(MPI_COMM_WORLD);
      times[2][t] += MPI_Wtime() - seconds;
    }

    seconds = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    #ifndef NO_IO_TEST
    if (NFILES > 0) {
      write_grid(rank, grid, t % NFILES, node_id);
    }
    #endif
    MPI_Barrier(MPI_COMM_WORLD);
    times[3][t] += MPI_Wtime() - seconds;

    seconds = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    #ifndef NO_IO_TEST
    if (NFILES > 0) {
      read_grid(rank, grid, t % NFILES, node_id);
    }
    #endif
    MPI_Barrier(MPI_COMM_WORLD);
    times[4][t] += MPI_Wtime() - seconds;

    verify(comm, grid);

    print_timings_oneline(rank, t);
  }

  print_timings(rank);

  free(send_neg);
  free(send_pos);
  free(recv_neg);
  free(recv_pos);
  free(ua_grid);
  MPI_Finalize();
  return 0;
}

// ======================= Definitions of functions ===========================

void
check_grid_parameters(const int rank)
{
  if (GRIDNX < 2 || GRIDNY < 2 || GRIDNZ < 2) {
    if (rank == 0) {
      printf("Error : Please use two or more grid points along each axis\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(1);
  }
  if (NTIMES < 2) {
    if (rank == 0) {
      printf("Error : Please use NTIMES >= 2 for getting walltime statistics\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(1);
  }
  if (NSTEPS < 1) {
    if (rank == 0) {
      printf("Error : Please use NSTEPS >= 1 for a valid simulation run\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(1);
  }
  return;
}

// print parameters completely specifying the grid, buffers and MPI topology
void
print_summary(MPI_Comm comm, const size_t grid_bytes, const size_t buffer_bytes)
{
  int mpi_rank;
  int np;
  int n[DIM];
  int periodic[DIM];
  int mpi_coords[DIM];
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Cart_get(comm, DIM, n, periodic, mpi_coords);
  if (mpi_rank == 0) {
    printf(HLINE);
    printf("SSLABS HPC Saturate\n");
    printf("Copyright (C) 2019 SankhyaSutra Labs Pvt. Ltd.\n");
    printf(HLINE);
    printf("This test uses:\n");
    printf("(a) %d bytes per variable\n", (int)sizeof(VAR_TYPE));
    printf("(b) %d x %d = %d variables per matrix\n", NROWS, NCOLS, NVARS);
    printf("(c) 1 matrix per grid point\n");
    printf("(d) %d x %d x %d points per grid portion\n", GRIDNX, GRIDNY, GRIDNZ);
    printf("(e) %d variables per grid portion\n", (NVARS * GRIDNX * GRIDNY * GRIDNZ));
    printf("(f) 1 extra layer of padding points beside each face of a grid portion\n");
    printf("(g) %.2f GB data allocated per padded grid portion\n", 1.0E-9 * grid_bytes);
    printf("(h) %.2f GB data allocated per buffer\n", 1.0E-9 * buffer_bytes);
    printf("(i) two send buffers and two recv buffers per grid portion\n");
    printf("(j) one grid portion per MPI process\n");
    printf("(k) %d x %d x %d = %d MPI processes in total\n", n[0], n[1], n[2], np);
    printf("(l) %.2f GB data allocated for padded grid\n", 1.0E-9 * np * grid_bytes);
    printf("(m) %.2f GB data allocated in total (padded grid + buffers)\n", 1.0E-9 * np * (grid_bytes + 4 * buffer_bytes));
    printf("(n) %d simulation steps in a run\n", NSTEPS);
    printf("(o) %d runs for generating walltime statistics\n", NTIMES);
    printf("(p) %d iterations for compute intensive portion\n", NCOMPUTE);
    printf("(q) %d files generated per process\n", NFILES);
    printf(HLINE);
  }
  return;
}

void
print_timings_oneline(const int rank, const int t)
{
  if (rank == 0) {
    printf("(%d)  ", t);
    for (int k = 0; k < NKERNELS; k++) {
      printf("%11.6f  ", times[k][t]);
    }
    printf("\n");
  }
  return;
}

// print walltime statistics for each kernel
// ignore timing for the first run while calculating statistics
// timing for each kernel is assumed to be similar across each process
// since MPI Barrier is called just before and after each kernel
void
print_timings(const int rank)
{
  for (int k = 0; k < NKERNELS; k++) {
    for (int t = 1; t < NTIMES; t++) {
      avgtime[k] = avgtime[k] + times[k][t];
      mintime[k] = MIN(mintime[k], times[k][t]);
      maxtime[k] = MAX(maxtime[k], times[k][t]);
    }
    avgtime[k] = avgtime[k] / (NTIMES - 1.);
  }

  if (rank == 0) {
    printf(HLINE);
    printf("Kernel     Avg time     Min time     Max time\n");
    printf(HLINE);
    for (int k = 0; k < NKERNELS; k++) {
      printf("%s%11.6f  %11.6f  %11.6f\n",
        label[k], avgtime[k], mintime[k], maxtime[k]);
    }
    printf(HLINE);
  }
  return;
}

// flat index of variable (r,c) in the matrix at grid point (x,y,z)
static inline int
idx(const int r, const int c, const int x, const int y, const int z)
{ return c + NCOLS * (x + (GRIDNX + 2) * (y + (GRIDNY + 2) * (z + (GRIDNZ + 2) * r))); }

static inline void
copy_matrix_from_grid_point(const VAR_TYPE* grid, const int x, const int y, const int z, VAR_TYPE* m)
{
  for (int r = 0; r < NROWS; r++) {
    memcpy(&m[r*NCOLS], &grid[idx(r,0,x,y,z)], NCOLS * sizeof(VAR_TYPE));
  }
  return;
}

static inline void
copy_matrix_to_grid_point(VAR_TYPE* grid, const int x, const int y, const int z, VAR_TYPE* m)
{
  for (int r = 0; r < NROWS; r++) {
    memcpy(&grid[idx(r,0,x,y,z)], &m[r*NCOLS], NCOLS * sizeof(VAR_TYPE));
  }
  return;
}

static inline void
matrix_to_stats(const VAR_TYPE* m, VAR_TYPE* s)
{
  s[0] = 0.;
  s[1] = 0.;
  s[2] = 0.;
  s[3] = 0.;
  for (int v = 0; v < NVARS; v++) {
    s[0] += m[v];
    s[1] += m[v] * c1[v];
    s[2] += m[v] * c2[v];
    s[3] += m[v] * c3[v];
  }
  const VAR_TYPE invs0 = 1. / s[0];
  s[1] *= invs0;
  s[2] *= invs0;
  s[3] *= invs0;
  return;
}

static inline void
stats_to_matrix(const VAR_TYPE* s, VAR_TYPE* m)
{
  // assumes s[i] = 0 for i = 1,2,3
  const VAR_TYPE sdots = s[1]*s[1] + s[2]*s[2] + s[3]*s[3] * ITHETA;
  for (int v = 0; v < NVARS; v++) {
    const VAR_TYPE cdots = (cos(sdots) - exp(sin(c1[v]*s[1] + c2[v]*s[2] + c3[v]*s[3]))) * ITHETA;
    const VAR_TYPE coeff = 1 + cdots - 0.5 * sdots + 0.5 * cdots * cdots;
    m[v] = coeff * s[0] * w[v];
  }
  return;
}

static inline void
update_matrix(VAR_TYPE* m)
{
  static VAR_TYPE tmp_matrix[NVARS];
  static VAR_TYPE s[NSTATS];
  matrix_to_stats(m, s);
  stats_to_matrix(s, tmp_matrix);
  for (int v = 0; v < NVARS; v++) {
    m[v] += BETA2 * (tmp_matrix[v] - m[v]);
  }
  return;
}

void
initialize(MPI_Comm comm, VAR_TYPE* grid)
{
  int mpi_nprocs[DIM];
  int periodic[DIM];
  int mpi_coords[DIM];
  MPI_Cart_get(comm, DIM, mpi_nprocs, periodic, mpi_coords);

  VAR_TYPE m[NVARS];
  VAR_TYPE s[NSTATS] = {0.};
  for (int z = 1; z <= GRIDNZ; z++)
  for (int y = 1; y <= GRIDNY; y++)
  for (int x = 1; x <= GRIDNX; x++)
  {
    //const VAR_TYPE xc = 2. * M_PI * (x + mpi_coords[0] * GRIDNX) / (1. * mpi_nprocs[0] * GRIDNX);
    //const VAR_TYPE yc = 2. * M_PI * (y + mpi_coords[1] * GRIDNY) / (1. * mpi_nprocs[1] * GRIDNY);
    //const VAR_TYPE zc = 2. * M_PI * (z + mpi_coords[2] * GRIDNZ) / (1. * mpi_nprocs[2] * GRIDNZ);
    s[0] = 1.;
    s[1] = 0;//U0 * sin(xc) * (cos(3. * yc) * cos(zc) - cos(yc) * cos(3. * zc));
    s[2] = 0;//U0 * sin(yc) * (cos(3. * zc) * cos(xc) - cos(zc) * cos(3. * xc));
    s[3] = 0;//U0 * sin(zc) * (cos(3. * xc) * cos(yc) - cos(xc) * cos(3. * yc));
    stats_to_matrix(s, m);
    copy_matrix_to_grid_point(grid, x, y, z, m);
  }
  return;
}

void
update(VAR_TYPE* grid)
{
  VAR_TYPE m[NVARS];
  for (int z = 1; z <= GRIDNZ; z++)
  for (int y = 1; y <= GRIDNY; y++)
  for (int x = 1; x <= GRIDNX; x++)
  {
    copy_matrix_from_grid_point(grid, x, y, z, m);
    for (int n = 0; n < NCOMPUTE; n++) {
      update_matrix(m);
    }
    copy_matrix_to_grid_point(grid, x, y, z, m);
  }
  return;
}

// functions for the sync kernel
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
sync(MPI_Comm comm,
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

void
scatter(VAR_TYPE* grid)
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
write_grid(const int mpi_rank, VAR_TYPE* grid, const int nfile, const int node_id)
{
  char fname[256];
  sprintf(fname, "grid_%d_%d_%d.bin", node_id, nfile, mpi_rank);
  FILE *fptr = fopen(fname, "wb");
  if (fptr == NULL) {
    printf("Could not open file %s for write\n", fname);
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(1);
  }
  const int num = NVARS * (GRIDNX + 2) * (GRIDNY + 2) * (GRIDNZ + 2);
  size_t ret_code = fwrite(grid, sizeof(VAR_TYPE), num, fptr);
  if (ret_code != num) {
    printf("Error writing file %s\n", fname);
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(1);
  }
  fclose(fptr);
  return;
}

void
read_grid(const int mpi_rank, VAR_TYPE* grid, const int nfile, const int node_id)
{
  char fname[256];
  sprintf(fname, "grid_%d_%d_%d.bin", node_id, nfile, mpi_rank);
  FILE *fptr = fopen(fname, "rb");
  if (fptr == NULL) {
    printf("Could not open file %s for read\n", fname);
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(1);
  }
  const int num = NVARS * (GRIDNX + 2) * (GRIDNY + 2) * (GRIDNZ + 2);
  size_t ret_code = fread(grid, sizeof(VAR_TYPE), num, fptr);
  if (ret_code != num) {
    printf("Error reading file %s\n", fname);
    MPI_Abort(MPI_COMM_WORLD, 2);
    exit(1);
  }
  fclose(fptr);
  return;
}

void
verify(MPI_Comm comm, VAR_TYPE* grid)
{
  for (int z = 1; z <= GRIDNZ; z++)
  for (int y = 1; y <= GRIDNY; y++)
  for (int x = 1; x <= GRIDNX; x++)
  {
    VAR_TYPE m[NVARS] = {0};
    VAR_TYPE s[NSTATS] = {-99.,-99.,-99.,-99.};
    copy_matrix_from_grid_point(grid,x,y,z,m);
    matrix_to_stats(m, s);
    int found_error = 0;
    if (fabs(s[0] - 1.0) > 1.0E-6) found_error = 1;
    if (fabs(s[1] - 0.0) > 1.0E-6) found_error = 1;
    if (fabs(s[2] - 0.0) > 1.0E-6) found_error = 1;
    if (fabs(s[3] - 0.0) > 1.0E-6) found_error = 1;
    if (found_error == 1) {
      printf("Error found during verification of results.\n");
      MPI_Abort(MPI_COMM_WORLD, 2);
      exit(1);
    }
  }
  return;
}
