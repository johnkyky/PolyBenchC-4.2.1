/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemm.h"

/* Array initialization. */
static void init_array(int ni, int nj, int nk, DATA_TYPE *alpha,
                       DATA_TYPE *beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NI, NJ, ni, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj)) {
  *alpha = 1.5;
  *beta = 1.2;
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++)
      ARRAY_2D_ACCESS(C, i, j) = (DATA_TYPE)((i * j + 1) % ni) / ni;
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nk; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)(i * (j + 1) % nk) / nk;
  for (int i = 0; i < nk; i++)
    for (int j = 0; j < nj; j++)
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)(i * (j + 2) % nj) / nj;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NI, NJ, ni, nj)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      if ((i * ni + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(C, i, j));
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_gemm(size_t ni, size_t nj, int nk, DATA_TYPE alpha,
                        DATA_TYPE beta,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NI, NJ, ni, nj),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj)) {
// BLAS PARAMS
// TRANSA = 'N'
// TRANSB = 'N'
//  => Form C := alpha*A*B + beta*C,
// A is NIxNK
// B is NKxNJ
// C is NIxNJ
#if defined(POLYBENCH_USE_POLLY)
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, ni);

  Kokkos::parallel_for<usePolyOpt, "p0.l0 == 0">(
      policy, KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < KOKKOS_LOOP_BOUND(nj); j++)
          C(i, j) *= beta;
        for (size_t k = 0; k < KOKKOS_LOOP_BOUND(nk); k++) {
          for (size_t j = 0; j < KOKKOS_LOOP_BOUND(nj); j++)
            C(i, j) += alpha * A(i, k) * B(k, j);
        }
      });
#elif defined(POLYBENCH_KOKKOS)
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, ni);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < nj; j++)
          C(i, j) *= beta;
        for (size_t k = 0; k < nk; k++) {
          for (size_t j = 0; j < nj; j++)
            C(i, j) += alpha * A(i, k) * B(k, j);
        }
      });
#else
#pragma scop
  for (size_t i = 0; i < ni; i++) {
    for (size_t j = 0; j < nj; j++)
      C[i][j] *= beta;
    for (size_t k = 0; k < nk; k++) {
      for (size_t j = 0; j < nj; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);

  /* Initialize array(s). */
  init_array(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  FINALIZE;

  return 0;
}
