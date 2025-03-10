/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"

/* Array initialization. */
static void init_array(int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, A, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, B, N, n)) {
  for (int i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(A, i) = ((DATA_TYPE)i + 2) / n;
    ARRAY_1D_ACCESS(B, i) = ((DATA_TYPE)i + 3) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, A, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(A, i));
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_jacobi_1d(int tsteps, int n,
                             ARRAY_1D_FUNC_PARAM(DATA_TYPE, A, N, n),
                             ARRAY_1D_FUNC_PARAM(DATA_TYPE, B, N, n)) {
#if defined(POLYBENCH_KOKKOS)
  const auto policy = Kokkos::RangePolicy<>(1, n - 1);
  for (int t = 0; t < _PB_TSTEPS; t++) {
    Kokkos::parallel_for<usePolyOpt>(
        policy, KOKKOS_LAMBDA(const int i) {
          B(i) = 0.33333 * (A(i - 1) + A(i) + A(i + 1));
        });
    Kokkos::parallel_for<usePolyOpt>(
        policy, KOKKOS_LAMBDA(const int i) {
          A(i) = 0.33333 * (B(i - 1) + B(i) + B(i + 1));
        });
  }
#else
  for (int t = 0; t < _PB_TSTEPS; t++) {
#pragma scop
    for (int i = 1; i < _PB_N - 1; i++)
      B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
    for (int i = 1; i < _PB_N - 1; i++)
      A[i] = 0.33333 * (B[i - 1] + B[i] + B[i + 1]);
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
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_1d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  FINALIZE;

  return 0;
}
