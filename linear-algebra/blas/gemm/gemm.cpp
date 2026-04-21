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
static void init_array(INT_TYPE ni, INT_TYPE nj, INT_TYPE nk, DATA_TYPE *alpha,
                       DATA_TYPE *beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NI, NJ, ni, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj)) {
  *alpha = 1.5;
  *beta = 1.2;
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nj; j++)
      ARRAY_2D_ACCESS(C, i, j) = (DATA_TYPE)((i * j + 1) % ni) / ni;
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nk; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)(i * (j + 1) % nk) / nk;
  for (INT_TYPE i = 0; i < nk; i++)
    for (INT_TYPE j = 0; j < nj; j++)
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)(i * (j + 2) % nj) / nj;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE ni, INT_TYPE nj,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NI, NJ, ni, nj)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nj; j++) {
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
static void kernel_gemm(INT_TYPE ni, INT_TYPE nj, INT_TYPE nk, DATA_TYPE alpha,
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
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, ni);
#else // GPU
  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, ni);
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0, p0.u0 > 10">(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < KOKKOS_LOOP_BOUND(nj); j++)
          C(i, j) *= beta;
        for (INT_TYPE k = 0; k < KOKKOS_LOOP_BOUND(nk); k++) {
          for (INT_TYPE j = 0; j < KOKKOS_LOOP_BOUND(nj); j++)
            C(i, j) += alpha * A(i, k) * B(k, j);
        }
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, ni);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < nj; j++)
          C(i, j) *= beta;
        for (INT_TYPE k = 0; k < nk; k++) {
          for (INT_TYPE j = 0; j < nj; j++)
            C(i, j) += alpha * A(i, k) * B(k, j);
        }
      });
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(A, ni, nk);
  polybench_GPU_array_2D(B, nk, nj);
  polybench_GPU_array_2D(C, ni, nj);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(B);
  polybench_GPU_array_copy_to_device(C);

  Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                        Kokkos::Rank<2>>
      policy({0, 0}, {ni, nj});
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        d_C(i, j) *= beta;
        for (INT_TYPE k = 0; k < nk; k++) {
          d_C(i, j) += alpha * d_A(i, k) * d_B(k, j);
        }
      });

  polybench_GPU_array_copy_to_host(C);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(C, ni, nj);

#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < ni; i++) {
    for (INT_TYPE j = 0; j < nj; j++)
      C[i][j] *= beta;
    for (INT_TYPE k = 0; k < nk; k++) {
      for (INT_TYPE j = 0; j < nj; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop
  polybench_stop_instruments;
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  INT_TYPE ni = NI;
  INT_TYPE nj = NJ;
  INT_TYPE nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);

  /* Initialize array(s). */
  init_array(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B));

  /* Run kernel. */
  kernel_gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(B));

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
