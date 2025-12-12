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
static void init_array(int n, ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      ARRAY_2D_ACCESS(A, i, j) = ((DATA_TYPE)i * (j + 2) + 2) / n;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
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
static void kernel_seidel_2d(size_t tsteps, size_t n,
                             ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy_time = Kokkos::RangePolicy<Kokkos::OpenMP>(0, tsteps);
  Kokkos::parallel_for<Kokkos::usePolyOpt,
                       "p0.l0 == 0, p0.u0 < 10000, p0.l0 < p0.u0">(
      policy_time, KOKKOS_LAMBDA(const size_t t) {
        for (size_t i = 1; i <= KOKKOS_LOOP_BOUND(n) - 2; i++)
          for (size_t j = 1; j <= KOKKOS_LOOP_BOUND(n) - 2; j++)
            A(i, j) = (A(i - 1, j - 1) + A(i - 1, j) + A(i - 1, j + 1) +
                       A(i, j - 1) + A(i, j) + A(i, j + 1) + A(i + 1, j - 1) +
                       A(i + 1, j) + A(i + 1, j + 1)) /
                      SCALAR_VAL(9.0);
      });
#elif defined(POLYBENCH_KOKKOS)
  const auto policy = Kokkos::RangePolicy<Kokkos::Serial>(0, tsteps);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const size_t t) {
        for (size_t i = 1; i <= n - 2; i++)
          for (size_t j = 1; j <= n - 2; j++)
            A(i, j) = (A(i - 1, j - 1) + A(i - 1, j) + A(i - 1, j + 1) +
                       A(i, j - 1) + A(i, j) + A(i, j + 1) + A(i + 1, j - 1) +
                       A(i + 1, j) + A(i + 1, j + 1)) /
                      SCALAR_VAL(9.0);
      });
#else
#pragma scop
  for (size_t t = 0; t <= tsteps - 1; t++) {
    for (size_t i = 1; i <= n - 2; i++)
      for (size_t j = 1; j <= n - 2; j++)
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] +
                   A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] +
                   A[i + 1][j] + A[i + 1][j + 1]) /
                  SCALAR_VAL(9.0);
#pragma endscop
  }
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_seidel_2d(tsteps, n, POLYBENCH_ARRAY(A));

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
