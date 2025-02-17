/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* floyd-warshall.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "floyd-warshall.h"

/* Array initialization. */
static void init_array(int n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, path, N, N, n, n)) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      ARRAY_2D_ACCESS(path, i, j) = i * j % 7 + 1;
      if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
        ARRAY_2D_ACCESS(path, i, j) = 999;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, path, N, N, n, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("path");
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(path, i, j));
    }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_floyd_warshall(int n, ARRAY_2D_FUNC_PARAM(DATA_TYPE, path, N,
                                                             N, n, n)) {
#if defined(POLYBENCH_KOKKOS)
  const auto policy = Kokkos::MDRangePolicy<Kokkos::Serial, Kokkos::Rank<3>>(
      {0, 0, 0}, {n, n, n});
  Kokkos::parallel_for<usePolyOpt>(
      policy, KOKKOS_LAMBDA(const int k, const int i, const int j) {
        path(i, j) = path(i, j) < path(i, k) + path(k, j)
                         ? path(i, j)
                         : path(i, k) + path(k, j);
      });
#else
#pragma scop
  for (int k = 0; k < _PB_N; k++) {
    for (int i = 0; i < _PB_N; i++)
      for (int j = 0; j < _PB_N; j++)
        path[i][j] = path[i][j] < path[i][k] + path[k][j]
                         ? path[i][j]
                         : path[i][k] + path[k][j];
  }
#pragma endscop
#endif
}

int main(int argc, char **argv) {

  INITIALIZE;

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(path));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_floyd_warshall(n, POLYBENCH_ARRAY(path));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(path);

  FINALIZE;

  return 0;
}
