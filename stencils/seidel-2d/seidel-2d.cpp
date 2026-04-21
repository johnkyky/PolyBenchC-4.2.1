/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* seidel-2d.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "seidel-2d.h"

/* Array initialization. */
static void init_array(INT_TYPE n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++)
      ARRAY_2D_ACCESS(A, i, j) = ((DATA_TYPE)i * (j + 2) + 2) / n;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(A, i, j));
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_seidel_2d(INT_TYPE tsteps, INT_TYPE n,
                             ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_time = Kokkos::RangePolicy<Kokkos::OpenMP>(0, tsteps);
#else // GPU
  const auto policy_time =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, tsteps);
#endif
  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0, n > 10">(
      policy_time, KOKKOS_LAMBDA(const INT_TYPE t) {
        for (INT_TYPE i = 1; i <= KOKKOS_LOOP_BOUND(n) - 2; i++)
          for (INT_TYPE j = 1; j <= KOKKOS_LOOP_BOUND(n) - 2; j++)
            A(i, j) = (A(i - 1, j - 1) + A(i - 1, j) + A(i - 1, j + 1) +
                       A(i, j - 1) + A(i, j) + A(i, j + 1) + A(i + 1, j - 1) +
                       A(i + 1, j) + A(i + 1, j + 1)) /
                      SCALAR_VAL(9.0);
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;

  const auto policy = Kokkos::RangePolicy<Kokkos::Serial>(0, tsteps);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE t) {
        for (INT_TYPE i = 1; i <= n - 2; i++)
          for (INT_TYPE j = 1; j <= n - 2; j++)
            A(i, j) = (A(i - 1, j - 1) + A(i - 1, j) + A(i - 1, j + 1) +
                       A(i, j - 1) + A(i, j) + A(i, j + 1) + A(i + 1, j - 1) +
                       A(i + 1, j) + A(i + 1, j + 1)) /
                      SCALAR_VAL(9.0);
      });
  polybench_stop_instruments;
#else
  polybench_GPU_array_2D(A, n, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);

  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, 1);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE thread_id) {
        for (INT_TYPE t = 0; t < tsteps; t++) {
          for (INT_TYPE i = 1; i <= n - 2; i++) {
            for (INT_TYPE j = 1; j <= n - 2; j++) {
              d_A(i, j) =
                  (d_A(i - 1, j - 1) + d_A(i - 1, j) + d_A(i - 1, j + 1) +
                   d_A(i, j - 1) + d_A(i, j) + d_A(i, j + 1) +
                   d_A(i + 1, j - 1) + d_A(i + 1, j) + d_A(i + 1, j + 1)) /
                  SCALAR_VAL(9.0);
            }
          }
        }
      });

  polybench_GPU_array_copy_to_host(A);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(A, n, n);
#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE t = 0; t <= tsteps - 1; t++) {
    for (INT_TYPE i = 1; i <= n - 2; i++)
      for (INT_TYPE j = 1; j <= n - 2; j++)
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] +
                   A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] +
                   A[i + 1][j] + A[i + 1][j + 1]) /
                  SCALAR_VAL(9.0);
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
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A));

  /* Run kernel. */
  kernel_seidel_2d(tsteps, n, POLYBENCH_ARRAY(A));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  FINALIZE;

  return 0;
}
