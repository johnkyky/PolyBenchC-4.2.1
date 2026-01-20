/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trmm.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trmm.h"

/* Array initialization. */
static void init_array(int m, int n, DATA_TYPE *alpha,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, M, m, m),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, M, N, m, n)) {
  *alpha = 1.5;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < i; j++) {
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i + j) % m) / m;
    }
    ARRAY_2D_ACCESS(A, i, i) = 1.0;
    for (int j = 0; j < n; j++) {
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)((n + (i - j)) % n) / n;
    }
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m, int n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, M, N, m, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("B");
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(B, i, j));
    }
  POLYBENCH_DUMP_END("B");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_trmm(size_t m, size_t n, DATA_TYPE alpha,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, M, m, m),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, M, N, m, n)) {
// BLAS parameters
// SIDE   = 'L'
// UPLO   = 'L'
// TRANSA = 'T'
// DIAG   = 'U'
//  => Form  B := alpha*A**T*B.
//  A is MxM
//  B is MxN
#if defined(POLYBENCH_USE_POLLY)
  const auto policy =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {m, n});

  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l == 0, p0.u0 == m">(
      policy, KOKKOS_LAMBDA(const size_t i, const size_t j) {
        for (size_t k = i + 1; k < KOKKOS_LOOP_BOUND(m); k++)
          B(i, j) += A(k, i) * B(k, j);
        B(i, j) = alpha * B(i, j);
      });
#elif defined(POLYBENCH_KOKKOS)
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
  for (size_t i = 0; i < m; i++) {
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const size_t j) {
          for (size_t k = i + 1; k < m; k++) {
            B[i][j] += A[k][i] * B[k][j];
          }
          B[i][j] = alpha * B[i][j];
        });
  }
#else
#pragma scop
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      for (size_t k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
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
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, M, m, m);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, M, N, m, n);

  /* Initialize array(s). */
  init_array(m, n, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_trmm(m, n, alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(B)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  FINALIZE;

  return 0;
}
