/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* symm.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "symm.h"

/* Array initialization. */
static void init_array(int m, int n, DATA_TYPE *alpha, DATA_TYPE *beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, M, N, m, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, M, m, m),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, M, N, m, n)) {
  *alpha = 1.5;
  *beta = 1.2;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      ARRAY_2D_ACCESS(C, i, j) = (DATA_TYPE)((i + j) % 100) / m;
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)((n + i - j) % 100) / m;
    }
  for (int i = 0; i < m; i++) {
    for (int j = 0; j <= i; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i + j) % 100) / m;
    for (int j = i + 1; j < m; j++)
      ARRAY_2D_ACCESS(A, i, j) =
          -999; // regions of arrays that should not be used
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m, int n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, M, N, m, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      if ((i * m + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(C, i, j));
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_symm(int m, int n, DATA_TYPE alpha, DATA_TYPE beta,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, M, N, m, n),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, M, m, m),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, M, N, m, n)) {
  DATA_TYPE temp2;

  // BLAS PARAMS
  // SIDE = 'L'
  // UPLO = 'L'
  //  =>  Form  C := alpha*A*B + beta*C
  //  A is MxM
  //  B is MxN
  //  C is MxN
  // note that due to Fortran array layout, the code below more closely
  // resembles upper triangular case in BLAS

#if defined(POLYBENCH_USE_POLLY)
  const auto policy =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {m, n});

  Kokkos::parallel_for<usePolyOpt>(
      policy, KOKKOS_LAMBDA(const size_t i, const size_t j) {
        DATA_TYPE temp = 0;
        for (size_t k = 0; k < i; k++) {
          C(k, j) += alpha * B(i, j) * A(i, k);
          temp += B(k, j) * A(i, k);
        }
        C(i, j) = beta * C(i, j) + alpha * B(i, j) * A(i, i) + alpha * temp;
      });
#elif defined(POLYBENCH_KOKKOS)
#else
#pragma scop
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
      temp2 = 0;
      for (size_t k = 0; k < i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
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
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, M, N, m, n);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, M, m, m);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, M, N, m, n);

  /* Initialize array(s). */
  init_array(m, n, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_symm(m, n, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  FINALIZE;

  return 0;
}
