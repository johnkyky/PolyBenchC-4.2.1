/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* fdtd-2d.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "fdtd-2d.h"

/* Array initialization. */
static void init_array(int tmax, int nx, int ny,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, ex, NX, NY, nx, ny),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, ey, NX, NY, nx, ny),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, hz, NX, NY, nx, ny),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, _fict_, TMAX, tmax)) {
  for (int i = 0; i < tmax; i++)
    ARRAY_1D_ACCESS(_fict_, i) = (DATA_TYPE)i;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      ARRAY_2D_ACCESS(ex, i, j) = ((DATA_TYPE)i * (j + 1)) / nx;
      ARRAY_2D_ACCESS(ey, i, j) = ((DATA_TYPE)i * (j + 2)) / ny;
      ARRAY_2D_ACCESS(hz, i, j) = ((DATA_TYPE)i * (j + 3)) / nx;
    }
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nx, int ny,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, ex, NX, NY, nx, ny),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, ey, NX, NY, nx, ny),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, hz, NX, NY, nx, ny)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(ex, i, j));
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(ey, i, j));
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(hz, i, j));
    }
  POLYBENCH_DUMP_END("hz");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_fdtd_2d(size_t tmax, size_t nx, size_t ny,
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, ex, NX, NY, nx, ny),
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, ey, NX, NY, nx, ny),
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, hz, NX, NY, nx, ny),
                           ARRAY_1D_FUNC_PARAM(DATA_TYPE, _fict_, TMAX, tmax)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy_time = Kokkos::RangePolicy<Kokkos::OpenMP>(0, tmax);

  Kokkos::parallel_for<Kokkos::usePolyOpt,
                       "p0.l0 == 0, p0.u0 < 50000, ny < 100000, nx < 900000">(
      policy_time, KOKKOS_LAMBDA(const size_t t) {
        for (size_t j = 0; j < KOKKOS_LOOP_BOUND(ny); j++)
          ey(0, j) = _fict_(t);
        for (size_t i = 1; i < KOKKOS_LOOP_BOUND(nx); i++)
          for (size_t j = 0; j < KOKKOS_LOOP_BOUND(ny); j++)
            ey(i, j) = ey(i, j) - SCALAR_VAL(0.5) * (hz(i, j) - hz(i - 1, j));
        for (size_t i = 0; i < KOKKOS_LOOP_BOUND(nx); i++)
          for (size_t j = 1; j < KOKKOS_LOOP_BOUND(ny); j++)
            ex(i, j) = ex(i, j) - SCALAR_VAL(0.5) * (hz(i, j) - hz(i, j - 1));
        for (size_t i = 0; i < KOKKOS_LOOP_BOUND(nx) - 1; i++)
          for (size_t j = 0; j < KOKKOS_LOOP_BOUND(ny) - 1; j++)
            hz(i, j) = hz(i, j) - SCALAR_VAL(0.7) * (ex(i, j + 1) - ex(i, j) +
                                                     ey(i + 1, j) - ey(i, j));
      });

#elif defined(POLYBENCH_KOKKOS)
  const auto policy_1D_y = Kokkos::RangePolicy<Kokkos::OpenMP>(0, ny);
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({1, 0}, {nx, ny},
                                                             {32, 32});
  const auto policy_2D_2 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 1}, {nx, ny},
                                                             {32, 32});
  const auto policy_2D_3 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>(
          {0, 0}, {nx - 1, ny - 1}, {32, 32});

  for (size_t t = 0; t < tmax; t++) {
    Kokkos::parallel_for(
        policy_1D_y, KOKKOS_LAMBDA(const size_t j) { ey(0, j) = _fict_(t); });

    Kokkos::parallel_for(
        policy_2D_1, KOKKOS_LAMBDA(const size_t i, const size_t j) {
          ey(i, j) = ey(i, j) - SCALAR_VAL(0.5) * (hz(i, j) - hz(i - 1, j));
        });

    Kokkos::parallel_for(
        policy_2D_2, KOKKOS_LAMBDA(const size_t i, const size_t j) {
          ex(i, j) = ex(i, j) - SCALAR_VAL(0.5) * (hz(i, j) - hz(i, j - 1));
        });

    Kokkos::parallel_for(
        policy_2D_3, KOKKOS_LAMBDA(const size_t i, const size_t j) {
          hz(i, j) = hz(i, j) - SCALAR_VAL(0.7) * (ex(i, j + 1) - ex(i, j) +
                                                   ey(i + 1, j) - ey(i, j));
        });
  }
#else
#pragma scop
  for (size_t t = 0; t < tmax; t++) {
    for (size_t j = 0; j < ny; j++)
      ey[0][j] = _fict_[t];
    for (size_t i = 1; i < nx; i++)
      for (size_t j = 0; j < ny; j++)
        ey[i][j] = ey[i][j] - SCALAR_VAL(0.5) * (hz[i][j] - hz[i - 1][j]);
    for (size_t i = 0; i < nx; i++)
      for (size_t j = 1; j < ny; j++)
        ex[i][j] = ex[i][j] - SCALAR_VAL(0.5) * (hz[i][j] - hz[i][j - 1]);
    for (size_t i = 0; i < nx - 1; i++)
      for (size_t j = 0; j < ny - 1; j++)
        hz[i][j] = hz[i][j] - SCALAR_VAL(0.7) * (ex[i][j + 1] - ex[i][j] +
                                                 ey[i + 1][j] - ey[i][j]);
  }
#pragma endscop
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(ex, DATA_TYPE, NX, NY, nx, ny);
  POLYBENCH_2D_ARRAY_DECL(ey, DATA_TYPE, NX, NY, nx, ny);
  POLYBENCH_2D_ARRAY_DECL(hz, DATA_TYPE, NX, NY, nx, ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_, DATA_TYPE, TMAX, tmax);

  /* Initialize array(s). */
  init_array(tmax, nx, ny, POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey),
             POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(_fict_));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_fdtd_2d(tmax, nx, ny, POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey),
                 POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(_fict_));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(ex),
                                    POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);

  FINALIZE;

  return 0;
}
