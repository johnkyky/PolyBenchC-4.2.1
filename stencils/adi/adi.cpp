/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* adi.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"

/* Array initialization. */
static void init_array(size_t n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, u, N, N, n, n)) {
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) {
      ARRAY_2D_ACCESS(u, i, j) = (DATA_TYPE)(i + n - j) / n;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(size_t n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, u, N, N, n, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("u");
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(u, i, j));
    }
  POLYBENCH_DUMP_END("u");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel
 * Computers" by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
static void kernel_adi(size_t tsteps, size_t n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, u, N, N, n, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, v, N, N, n, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, p, N, N, n, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, q, N, N, n, n)) {
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;

  DX = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
  DY = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
  DT = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 / SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0) + mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0) + mul2;
  f = d;

#if defined(POLYBENCH_USE_POLLY)
  const auto policy_time = Kokkos::RangePolicy<Kokkos::OpenMP>(1, tsteps + 1);

  Kokkos::parallel_for<Kokkos::usePolyOpt,
                       "p0.l0 == 1, p0.u0 < 1000000, n < 900000">(
      policy_time, KOKKOS_LAMBDA(const size_t t) {
        for (size_t i = 1; i < KOKKOS_LOOP_BOUND(n) - 1; i++) {
          v(0, i) = SCALAR_VAL(1.0);
          p(i, 0) = SCALAR_VAL(0.0);
          q(i, 0) = v(0, i);
          for (size_t j = 1; j < KOKKOS_LOOP_BOUND(n) - 1; j++) {
            p(i, j) = -c / (a * p(i, j - 1) + b);
            q(i, j) = (-d * u(j, i - 1) +
                       (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u(j, i) -
                       f * u(j, i + 1) - a * q(i, j - 1)) /
                      (a * p(i, j - 1) + b);
          }

          v(n - 1, i) = SCALAR_VAL(1.0);
          for (size_t j = KOKKOS_LOOP_BOUND(n) - 2; j >= 1; j--) {
            v(j, i) = p(i, j) * v(j + 1, i) + q(i, j);
          }
        }
        // Row Sweep
        for (size_t i = 1; i < KOKKOS_LOOP_BOUND(n) - 1; i++) {
          u(i, 0) = SCALAR_VAL(1.0);
          p(i, 0) = SCALAR_VAL(0.0);
          q(i, 0) = u(i, 0);
          for (size_t j = 1; j < KOKKOS_LOOP_BOUND(n) - 1; j++) {
            p(i, j) = -f / (d * p(i, j - 1) + e);
            q(i, j) = (-a * v(i - 1, j) +
                       (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v(i, j) -
                       c * v(i + 1, j) - d * q(i, j - 1)) /
                      (d * p(i, j - 1) + e);
          }
          u(i, n - 1) = SCALAR_VAL(1.0);
          for (size_t j = KOKKOS_LOOP_BOUND(n) - 2; j >= 1; j--) {
            u(i, j) = p(i, j) * u(i, j + 1) + q(i, j);
          }
        }
      });

#elif defined(POLYBENCH_KOKKOS)
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(1, n - 1);

  for (size_t t = 1; t <= tsteps; t++) {
    // Column Sweep
    Kokkos::parallel_for(
        policy_1D, KOKKOS_LAMBDA(const size_t i) {
          v(0, i) = SCALAR_VAL(1.0);
          p(i, 0) = SCALAR_VAL(0.0);
          q(i, 0) = v(0, i);
          for (size_t j = 1; j < n - 1; j++) {
            p(i, j) = -c / (a * p(i, j - 1) + b);
            q(i, j) = (-d * u(j, i - 1) +
                       (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u(j, i) -
                       f * u(j, i + 1) - a * q(i, j - 1)) /
                      (a * p(i, j - 1) + b);
          }

          v(n - 1, i) = SCALAR_VAL(1.0);
          for (size_t j = n - 2; j >= 1; j--) {
            v(j, i) = p(i, j) * v(j + 1, i) + q(i, j);
          }
        });

    // Row Sweep
    Kokkos::parallel_for<usePolyOpt>(
        policy_1D, KOKKOS_LAMBDA(const size_t i) {
          u(i, 0) = SCALAR_VAL(1.0);
          p(i, 0) = SCALAR_VAL(0.0);
          q(i, 0) = u(i, 0);
          for (size_t j = 1; j < n - 1; j++) {
            p(i, j) = -f / (d * p(i, j - 1) + e);
            q(i, j) = (-a * v(i - 1, j) +
                       (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v(i, j) -
                       c * v(i + 1, j) - d * q(i, j - 1)) /
                      (d * p(i, j - 1) + e);
          }
          u(i, n - 1) = SCALAR_VAL(1.0);
          for (size_t j = n - 2; j >= 1; j--) {
            u(i, j) = p(i, j) * u(i, j + 1) + q(i, j);
          }
        });
  }

#else
#pragma scop
  for (size_t t = 1; t <= tsteps; t++) {
    // Column Sweep
    for (size_t i = 1; i < n - 1; i++) {
      v[0][i] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = v[0][i];
      for (size_t j = 1; j < n - 1; j++) {
        p[i][j] = -c / (a * p[i][j - 1] + b);
        q[i][j] = (-d * u[j][i - 1] +
                   (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u[j][i] -
                   f * u[j][i + 1] - a * q[i][j - 1]) /
                  (a * p[i][j - 1] + b);
      }

      v[n - 1][i] = SCALAR_VAL(1.0);
      for (size_t j = n - 2; j >= 1; j--) {
        v[j][i] = p[i][j] * v[j + 1][i] + q[i][j];
      }
    }
    // Row Sweep
    for (size_t i = 1; i < n - 1; i++) {
      u[i][0] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = u[i][0];
      for (size_t j = 1; j < n - 1; j++) {
        p[i][j] = -f / (d * p[i][j - 1] + e);
        q[i][j] = (-a * v[i - 1][j] +
                   (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v[i][j] -
                   c * v[i + 1][j] - d * q[i][j - 1]) /
                  (d * p[i][j - 1] + e);
      }
      u[i][n - 1] = SCALAR_VAL(1.0);
      for (size_t j = n - 2; j >= 1; j--) {
        u[i][j] = p[i][j] * u[i][j + 1] + q[i][j];
      }
    }
  }
#pragma endscop
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  size_t n = N;
  size_t tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(u));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_adi(tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v),
             POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(u);
  POLYBENCH_FREE_ARRAY(v);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(q);

  FINALIZE;

  return 0;
}
