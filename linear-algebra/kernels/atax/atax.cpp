/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"

/* Array initialization. */
static void init_array(int m, int n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, N, m, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n)) {
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (int i = 0; i < n; i++)
    ARRAY_1D_ACCESS(x, i) = 1 + (i / fn);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i + j) % n) / (5 * m);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(y, i));
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_atax(size_t m, size_t n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, N, m, n),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, tmp, M, m)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy_1D_1 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
  const auto policy_1D_2 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m);

  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0, p1.l0 == 0">(
      "kernel", policy_1D_1, KOKKOS_LAMBDA(const size_t i) { y(i) = 0; },
      policy_1D_2,
      KOKKOS_LAMBDA(const size_t i) {
        tmp(i) = SCALAR_VAL(0.0);
        for (size_t j = 0; j < n; j++)
          tmp(i) = tmp(i) + A(i, j) * x(j);
        for (size_t j = 0; j < n; j++)
          y(j) = y(j) + A(i, j) * tmp(i);
      });

#elif defined(POLYBENCH_KOKKOS)
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);

  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const size_t i) { y(i) = 0; });

  for (size_t i = 0; i < m; i++) {
    DATA_TYPE tmp = SCALAR_VAL(0.0);
    Kokkos::parallel_reduce(
        policy,
        KOKKOS_LAMBDA(const size_t i, DATA_TYPE &local_tmp) {
          local_tmp += A(i, j) * x(j);
        },
        tmp);
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const size_t i) { y(j) = y(j) + A(i, j) * tmp(i); });
  });
#else
#pragma scop
  for (size_t i = 0; i < n; i++)
    y[i] = 0;
  for (size_t i = 0; i < m; i++) {
    tmp[i] = SCALAR_VAL(0.0);
    for (size_t j = 0; j < n; j++)
      tmp[i] = tmp[i] + A[i][j] * x[j];
    for (size_t j = 0; j < n; j++)
      y[j] = y[j] + A[i][j] * tmp[i];
  }
#pragma endscop
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_atax(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y),
              POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  FINALIZE;

  return 0;
}
