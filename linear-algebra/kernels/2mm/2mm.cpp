/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 2mm.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "2mm.h"

/* Array initialization. */
static void init_array(int ni, int nj, int nk, int nl, DATA_TYPE *alpha,
                       DATA_TYPE *beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NJ, NL, nj, nl),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, D, NI, NL, ni, nl)) {
  *alpha = 1.5;
  *beta = 1.2;
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nk; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i * j + 1) % ni) / ni;
  for (int i = 0; i < nk; i++)
    for (int j = 0; j < nj; j++)
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)(i * (j + 1) % nj) / nj;
  for (int i = 0; i < nj; i++)
    for (int j = 0; j < nl; j++)
      ARRAY_2D_ACCESS(C, i, j) = (DATA_TYPE)((i * (j + 3) + 1) % nl) / nl;
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++)
      ARRAY_2D_ACCESS(D, i, j) = (DATA_TYPE)(i * (j + 2) % nk) / nk;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nl,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, D, NI, NL, ni, nl)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(D, i, j));
    }
  POLYBENCH_DUMP_END("D");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_2mm(size_t ni, size_t nj, size_t nk, size_t nl,
                       DATA_TYPE alpha, DATA_TYPE beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, tmp, NI, NJ, ni, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NJ, NL, nj, nl),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, D, NI, NL, ni, nl)) {

#if defined(POLYBENCH_USE_POLLY)
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, ni);

  Kokkos::parallel_for<usePolyOpt, "p0.l0 == 0, p1.l0 == 0, p0.u0 == p1.u0">(
      "kernel", policy,
      KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < nj; j++) {
          tmp(i, j) = SCALAR_VAL(0.0);
          for (size_t k = 0; k < nk; ++k)
            tmp(i, j) += alpha * A(i, k) * B(k, j);
        }
      },
      policy,
      KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < nl; j++) {
          D(i, j) *= beta;
          for (size_t k = 0; k < nj; ++k)
            D(i, j) += tmp(i, k) * C(k, j);
        }
      });
#elif defined(POLYBENCH_KOKKOS)
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ni, nj});
  const auto policy_2D_2 =
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ni, nl});

  /* D := alpha*A*B*C + beta*D */
  Kokkos::parallel_for(
      policy_2D_1, KOKKOS_LAMBDA(const size_t i, const size_t j) {
        ARRAY_2D_ACCESS(tmp, i, j) = SCALAR_VAL(0.0);
        for (size_t k = 0; k < _PB_NK; ++k)
          ARRAY_2D_ACCESS(tmp, i, j) +=
              alpha * ARRAY_2D_ACCESS(A, i, k) * ARRAY_2D_ACCESS(B, k, j);
      });

  Kokkos::parallel_for(
      policy_2D_2, KOKKOS_LAMBDA(const size_t i, const size_t j) {
        ARRAY_2D_ACCESS(D, i, j) *= beta;
        for (size_t k = 0; k < _PB_NJ; ++k)
          ARRAY_2D_ACCESS(D, i, j) +=
              ARRAY_2D_ACCESS(tmp, i, k) * ARRAY_2D_ACCESS(C, k, j);
      });
#else
#pragma scop
  /* D := alpha*A*B*C + beta*D */
  for (size_t i = 0; i < ni; i++)
    for (size_t j = 0; j < nj; j++) {
      tmp[i][j] = SCALAR_VAL(0.0);
      for (size_t k = 0; k < nk; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
    }
  for (size_t i = 0; i < ni; i++)
    for (size_t j = 0; j < nl; j++) {
      D[i][j] *= beta;
      for (size_t k = 0; k < nj; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
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
  int nl = NL;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(tmp, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NI, NL, ni, nl);

  /* Initialize array(s). */
  init_array(ni, nj, nk, nl, &alpha, &beta, POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_2mm(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp),
             POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C),
             POLYBENCH_ARRAY(D));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(D)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);

  FINALIZE;

  return 0;
}
