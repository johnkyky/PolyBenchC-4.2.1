/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trisolv.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"

/* Array initialization. */
static void init_array(int n, ARRAY_2D_FUNC_PARAM(DATA_TYPE, L, N, N, n, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, b, N, n)) {
  for (int i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(x, i) = -999;
    ARRAY_1D_ACCESS(b, i) = i;
    for (int j = 0; j <= i; j++)
      ARRAY_2D_ACCESS(L, i, j) = (DATA_TYPE)(i + n - j + 1) * 2 / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (int i = 0; i < n; i++) {
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(x, i));
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_trisolv(size_t n,
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, L, N, N, n, n),
                           ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                           ARRAY_1D_FUNC_PARAM(DATA_TYPE, b, N, n)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);

  Kokkos::parallel_for<usePolyOpt, "p0.l0 == 0">(
      policy, KOKKOS_LAMBDA(const size_t i) {
        x(i) = b(i);
        for (size_t j = 0; j < i; j++)
          x(i) -= L(i, j) * x(j);
        x(i) = x(i) / L(i, i);
      });
#elif defined(POLYBENCH_KOKKOS)
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
  for (size_t i = 0; i < n; i++) {
    DATA_TYPE sum = DATA_TYPE(0);
    Kokkos::parallel_reduce(
        policy,
        KOKKOS_LAMBDA(size_t j, DATA_TYPE &local_sum) {
          local_sum += L(i, j) * x(j);
        },
        sum);

    x(i) = b(i) - sum;
    x(i) = x(i) / L(i, i);
  }
#else
#pragma scop
  for (size_t i = 0; i < n; i++) {
    x[i] = b[i];
    for (size_t j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
#pragma endscop
#endif
}

int main(int argc, char **argv) {

  INITIALIZE;

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trisolv(n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(L);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(b);

  FINALIZE;

  return 0;
}
