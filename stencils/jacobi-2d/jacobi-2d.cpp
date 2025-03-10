/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-2d.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"

/* Array initialization. */
static void init_array(int n, ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, N, N, n, n)) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      ARRAY_2D_ACCESS(A, i, j) = ((DATA_TYPE)i * (j + 2) + 2) / n;
      ARRAY_2D_ACCESS(B, i, j) = ((DATA_TYPE)i * (j + 3) + 3) / n;
    }
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
static void kernel_jacobi_2d(int tsteps, int n,
                             ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n),
                             ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, N, N, n, n)) {
#if defined(POLYBENCH_KOKKOS)
  const auto policy =
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {n - 1, n - 1}, {32, 32});
  for (int t = 0; t < _PB_TSTEPS; t++) {
    Kokkos::parallel_for<usePolyOpt>(
        policy, KOKKOS_LAMBDA(const int i, const int j) {
          B(i, j) = SCALAR_VAL(0.2) * (A(i, j) + A(i, j - 1) + A(i, 1 + j) +
                                       A(1 + i, j) + A(i - 1, j));
        });
    Kokkos::parallel_for<usePolyOpt>(
        policy, KOKKOS_LAMBDA(const int i, const int j) {
          A(i, j) = SCALAR_VAL(0.2) * (B(i, j) + B(i, j - 1) + B(i, 1 + j) +
                                       B(1 + i, j) + B(i - 1, j));
        });
  }
#else
  for (int t = 0; t < _PB_TSTEPS; t++) {
#pragma scop
    for (int i = 1; i < _PB_N - 1; i++)
      for (int j = 1; j < _PB_N - 1; j++)
        B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j - 1] + A[i][1 + j] +
                                     A[1 + i][j] + A[i - 1][j]);
    for (int i = 1; i < _PB_N - 1; i++)
      for (int j = 1; j < _PB_N - 1; j++)
        A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j - 1] + B[i][1 + j] +
                                     B[1 + i][j] + B[i - 1][j]);
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
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_2d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

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
