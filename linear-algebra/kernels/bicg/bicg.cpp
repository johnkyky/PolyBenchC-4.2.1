/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* bicg.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bicg.h"

/* Array initialization. */
static void init_array(int m, int n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, M, n, m),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, r, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, p, M, m)) {
  for (int i = 0; i < m; i++)
    ARRAY_1D_ACCESS(p, i) = (DATA_TYPE)(i % m) / m;
  for (int i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(r, i) = (DATA_TYPE)(i % n) / n;
    for (int j = 0; j < m; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)(i * (j + 1) % n) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m, int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, s, M, m),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, q, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (int i = 0; i < m; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(s, i));
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(q, i));
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_bicg(int m, int n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, M, n, m),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, s, M, m),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, q, N, n),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, p, M, m),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, r, N, n)) {
#if defined(POLYBENCH_KOKKOS)
  const auto policy_1D_1 = Kokkos::RangePolicy<Kokkos::Serial>(0, m);
  const auto policy_1D_2 = Kokkos::RangePolicy<Kokkos::Serial>(0, n);

  Kokkos::parallel_for<usePolyOpt>(
      policy_1D_1, KOKKOS_LAMBDA(const int i) { s(i) = 0; });
  Kokkos::parallel_for<usePolyOpt>(
      policy_1D_2, KOKKOS_LAMBDA(const int i) {
        q(i) = SCALAR_VAL(0.0);
        for (int j = 0; j < m; j++) {
          s(j) = s(j) + r(i) * A(i, j);
          q(i) = q(i) + A(i, j) * p(j);
        }
      });
#else
#pragma scop
  for (int i = 0; i < _PB_M; i++)
    ARRAY_1D_ACCESS(s, i) = 0;
  for (int i = 0; i < _PB_N; i++) {
    ARRAY_1D_ACCESS(q, i) = SCALAR_VAL(0.0);
    for (int j = 0; j < _PB_M; j++) {
      ARRAY_1D_ACCESS(s, j) = ARRAY_1D_ACCESS(s, j) +
                              ARRAY_1D_ACCESS(r, i) * ARRAY_2D_ACCESS(A, i, j);
      ARRAY_1D_ACCESS(q, i) = ARRAY_1D_ACCESS(q, i) +
                              ARRAY_2D_ACCESS(A, i, j) * ARRAY_1D_ACCESS(p, j);
    }
  }
#pragma endscop
#endif
}

int main(int argc, char **argv) {

  INITIALIZE;

  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(p));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_bicg(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q),
              POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(r));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(
      print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  FINALIZE;

  return 0;
}
