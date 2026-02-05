/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* heat-3d.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "heat-3d.h"

/* Array initialization. */
static void init_array(size_t n,
                       ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, N, N, N, n, n, n),
                       ARRAY_3D_FUNC_PARAM(DATA_TYPE, B, N, N, N, n, n, n)) {
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++)
      for (size_t k = 0; k < n; k++)
        ARRAY_3D_ACCESS(A, i, j, k) = ARRAY_3D_ACCESS(B, i, j, k) =
            (DATA_TYPE)(i + j + (n - k)) * 10 / (n);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(size_t n,
                        ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, N, N, N, n, n, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++)
      for (size_t k = 0; k < n; k++) {
        if ((i * n * n + j * n + k) % 20 == 0)
          fprintf(POLYBENCH_DUMP_TARGET, "\n");
        fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
                ARRAY_3D_ACCESS(A, i, j, k));
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_heat_3d(size_t tsteps, size_t n,
                           ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, N, N, N, n, n, n),
                           ARRAY_3D_FUNC_PARAM(DATA_TYPE, B, N, N, N, n, n,
                                               n)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy_time = Kokkos::RangePolicy<Kokkos::OpenMP>(0, tsteps + 1);
  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0, p0.u0 > 10">(
      policy_time, KOKKOS_LAMBDA(const size_t t) {
        for (size_t i = 1; i < KOKKOS_LOOP_BOUND(n) - 1; i++) {
          for (size_t j = 1; j < KOKKOS_LOOP_BOUND(n) - 1; j++) {
            for (size_t k = 1; k < KOKKOS_LOOP_BOUND(n) - 1; k++) {
              B(i, j, k) = SCALAR_VAL(0.125) *
                               (A(i + 1, j, k) - SCALAR_VAL(2.0) * A(i, j, k) +
                                A(i - 1, j, k)) +
                           SCALAR_VAL(0.125) *
                               (A(i, j + 1, k) - SCALAR_VAL(2.0) * A(i, j, k) +
                                A(i, j - 1, k)) +
                           SCALAR_VAL(0.125) *
                               (A(i, j, k + 1) - SCALAR_VAL(2.0) * A(i, j, k) +
                                A(i, j, k - 1)) +
                           A(i, j, k);
            }
          }
        }
        for (size_t i = 1; i < KOKKOS_LOOP_BOUND(n) - 1; i++) {
          for (size_t j = 1; j < KOKKOS_LOOP_BOUND(n) - 1; j++) {
            for (size_t k = 1; k < KOKKOS_LOOP_BOUND(n) - 1; k++) {
              A(i, j, k) = SCALAR_VAL(0.125) *
                               (B(i + 1, j, k) - SCALAR_VAL(2.0) * B(i, j, k) +
                                B(i - 1, j, k)) +
                           SCALAR_VAL(0.125) *
                               (B(i, j + 1, k) - SCALAR_VAL(2.0) * B(i, j, k) +
                                B(i, j - 1, k)) +
                           SCALAR_VAL(0.125) *
                               (B(i, j, k + 1) - SCALAR_VAL(2.0) * B(i, j, k) +
                                B(i, j, k - 1)) +
                           B(i, j, k);
            }
          }
        }
      });
#elif defined(POLYBENCH_KOKKOS)
  const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
      {1, 1, 1}, {n - 1, n - 1, n - 1}, {32, 32, 32});
  for (size_t t = 1; t <= tsteps; t++) {
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
          B(i, j, k) = SCALAR_VAL(0.125) *
                           (A(i + 1, j, k) - SCALAR_VAL(2.0) * A(i, j, k) +
                            A(i - 1, j, k)) +
                       SCALAR_VAL(0.125) *
                           (A(i, j + 1, k) - SCALAR_VAL(2.0) * A(i, j, k) +
                            A(i, j - 1, k)) +
                       SCALAR_VAL(0.125) *
                           (A(i, j, k + 1) - SCALAR_VAL(2.0) * A(i, j, k) +
                            A(i, j, k - 1)) +
                       A(i, j, k);
        });
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k) {
          A(i, j, k) = SCALAR_VAL(0.125) *
                           (B(i + 1, j, k) - SCALAR_VAL(2.0) * B(i, j, k) +
                            B(i - 1, j, k)) +
                       SCALAR_VAL(0.125) *
                           (B(i, j + 1, k) - SCALAR_VAL(2.0) * B(i, j, k) +
                            B(i, j - 1, k)) +
                       SCALAR_VAL(0.125) *
                           (B(i, j, k + 1) - SCALAR_VAL(2.0) * B(i, j, k) +
                            B(i, j, k - 1)) +
                       B(i, j, k);
        });
  }
#else
#pragma scop
  for (size_t t = 1; t <= tsteps; t++) {
    for (size_t i = 1; i < n - 1; i++) {
      for (size_t j = 1; j < n - 1; j++) {
        for (size_t k = 1; k < n - 1; k++) {
          B[i][j][k] = SCALAR_VAL(0.125) *
                           (A[i + 1][j][k] - SCALAR_VAL(2.0) * A[i][j][k] +
                            A[i - 1][j][k]) +
                       SCALAR_VAL(0.125) *
                           (A[i][j + 1][k] - SCALAR_VAL(2.0) * A[i][j][k] +
                            A[i][j - 1][k]) +
                       SCALAR_VAL(0.125) *
                           (A[i][j][k + 1] - SCALAR_VAL(2.0) * A[i][j][k] +
                            A[i][j][k - 1]) +
                       A[i][j][k];
        }
      }
    }
    for (size_t i = 1; i < n - 1; i++) {
      for (size_t j = 1; j < n - 1; j++) {
        for (size_t k = 1; k < n - 1; k++) {
          A[i][j][k] = SCALAR_VAL(0.125) *
                           (B[i + 1][j][k] - SCALAR_VAL(2.0) * B[i][j][k] +
                            B[i - 1][j][k]) +
                       SCALAR_VAL(0.125) *
                           (B[i][j + 1][k] - SCALAR_VAL(2.0) * B[i][j][k] +
                            B[i][j - 1][k]) +
                       SCALAR_VAL(0.125) *
                           (B[i][j][k + 1] - SCALAR_VAL(2.0) * B[i][j][k] +
                            B[i][j][k - 1]) +
                       B[i][j][k];
        }
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
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, N, N, N, n, n, n);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, N, N, N, n, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_heat_3d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  FINALIZE;

  return 0;
}
