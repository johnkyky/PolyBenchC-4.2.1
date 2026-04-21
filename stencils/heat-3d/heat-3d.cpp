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
static void init_array(INT_TYPE n,
                       ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, N, N, N, n, n, n),
                       ARRAY_3D_FUNC_PARAM(DATA_TYPE, B, N, N, N, n, n, n)) {
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++)
      for (INT_TYPE k = 0; k < n; k++)
        ARRAY_3D_ACCESS(A, i, j, k) = ARRAY_3D_ACCESS(B, i, j, k) =
            (DATA_TYPE)(i + j + (n - k)) * 10 / (n);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE n,
                        ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, N, N, N, n, n, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++)
      for (INT_TYPE k = 0; k < n; k++) {
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
static void kernel_heat_3d(INT_TYPE tsteps, INT_TYPE n,
                           ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, N, N, N, n, n, n),
                           ARRAY_3D_FUNC_PARAM(DATA_TYPE, B, N, N, N, n, n,
                                               n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_time = Kokkos::RangePolicy<Kokkos::OpenMP>(0, tsteps + 1);
#else // GPU
  const auto policy_time =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0,
                                                                    tsteps + 1);
#endif
  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0, p0.u0 > 10">(
      policy_time, KOKKOS_LAMBDA(const INT_TYPE t) {
        for (INT_TYPE i = 1; i < KOKKOS_LOOP_BOUND(n) - 1; i++) {
          for (INT_TYPE j = 1; j < KOKKOS_LOOP_BOUND(n) - 1; j++) {
            for (INT_TYPE k = 1; k < KOKKOS_LOOP_BOUND(n) - 1; k++) {
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
        for (INT_TYPE i = 1; i < KOKKOS_LOOP_BOUND(n) - 1; i++) {
          for (INT_TYPE j = 1; j < KOKKOS_LOOP_BOUND(n) - 1; j++) {
            for (INT_TYPE k = 1; k < KOKKOS_LOOP_BOUND(n) - 1; k++) {
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
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy = Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<3>>(
      {1, 1, 1}, {n - 1, n - 1, n - 1}, {32, 32, 32});
  for (INT_TYPE t = 1; t <= tsteps; t++) {
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j, const INT_TYPE k) {
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
        policy,
        KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j, const INT_TYPE k) {
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
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_3D(A, n, n, n);
  polybench_GPU_array_3D(B, n, n, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(B);

  const auto policy =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<3>>({1, 1, 1}, {n - 1, n - 1, n - 1});
  for (INT_TYPE t = 1; t <= tsteps; t++) {
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j, const INT_TYPE k) {
          d_B(i, j, k) = SCALAR_VAL(0.125) * (d_A(i + 1, j, k) -
                                              SCALAR_VAL(2.0) * d_A(i, j, k) +
                                              d_A(i - 1, j, k)) +
                         SCALAR_VAL(0.125) * (d_A(i, j + 1, k) -
                                              SCALAR_VAL(2.0) * d_A(i, j, k) +
                                              d_A(i, j - 1, k)) +
                         SCALAR_VAL(0.125) * (d_A(i, j, k + 1) -
                                              SCALAR_VAL(2.0) * d_A(i, j, k) +
                                              d_A(i, j, k - 1)) +
                         d_A(i, j, k);
        });
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j, const INT_TYPE k) {
          d_A(i, j, k) = SCALAR_VAL(0.125) * (d_B(i + 1, j, k) -
                                              SCALAR_VAL(2.0) * d_B(i, j, k) +
                                              d_B(i - 1, j, k)) +
                         SCALAR_VAL(0.125) * (d_B(i, j + 1, k) -
                                              SCALAR_VAL(2.0) * d_B(i, j, k) +
                                              d_B(i, j - 1, k)) +
                         SCALAR_VAL(0.125) * (d_B(i, j, k + 1) -
                                              SCALAR_VAL(2.0) * d_B(i, j, k) +
                                              d_B(i, j, k - 1)) +
                         d_B(i, j, k);
        });
  }

  polybench_GPU_array_copy_to_host(A);
  polybench_GPU_array_copy_to_host(B);

  polybench_stop_instruments;

  polybench_GPU_array_sync_3D(A, n, n, n);
  polybench_GPU_array_sync_3D(B, n, n, n);

#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE t = 1; t <= tsteps; t++) {
    for (INT_TYPE i = 1; i < n - 1; i++) {
      for (INT_TYPE j = 1; j < n - 1; j++) {
        for (INT_TYPE k = 1; k < n - 1; k++) {
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
    for (INT_TYPE i = 1; i < n - 1; i++) {
      for (INT_TYPE j = 1; j < n - 1; j++) {
        for (INT_TYPE k = 1; k < n - 1; k++) {
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
  polybench_stop_instruments;
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  INT_TYPE n = N;
  INT_TYPE tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, N, N, N, n, n, n);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, N, N, N, n, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Run kernel. */
  kernel_heat_3d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  FINALIZE;

  return 0;
}
