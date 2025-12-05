/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syrk.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syrk.h"

/* Array initialization. */
static void init_array(int n, int m, DATA_TYPE *alpha, DATA_TYPE *beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, N, N, n, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, M, n, m)) {
  *alpha = 1.5;
  *beta = 1.2;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i * j + 1) % n) / n;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      ARRAY_2D_ACCESS(C, i, j) = (DATA_TYPE)((i * j + 2) % m) / m;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, N, N, n, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(C, i, j));
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_syrk(size_t n, size_t m, DATA_TYPE alpha, DATA_TYPE beta,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, N, N, n, n),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, M, n, m)) {
// BLAS PARAMS
// TRANS = 'N'
// UPLO  = 'L'
//  =>  Form  C := alpha*A*A**T + beta*C.
// A is NxM
// C is NxN
#if defined(POLYBENCH_USE_POLLY)
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);

  Kokkos::parallel_for<usePolyOpt, "p0.l0 == 0">(
      policy, KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j <= i; j++)
          C(i, j) *= beta;
        for (size_t k = 0; k < m; k++) {
          for (size_t j = 0; j <= i; j++)
            C(i, j) += alpha * A(i, k) * A(j, k);
        }
      });

#elif defined(POLYBENCH_KOKKOS)
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j <= i; j++)
          C(i, j) *= beta;
        for (size_t k = 0; k < m; k++) {
          for (size_t j = 0; j <= i; j++)
            C(i, j) += alpha * A(i, k) * A(j, k);
        }
      });
#else
#pragma scop
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (size_t k = 0; k < m; k++) {
      for (size_t j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
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
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);

  /* Initialize array(s). */
  init_array(n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_syrk(n, m, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);

  FINALIZE;

  return 0;
}
