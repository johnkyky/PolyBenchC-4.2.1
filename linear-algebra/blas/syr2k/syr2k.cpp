/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* syr2k.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "syr2k.h"

/* Array initialization. */
static void init_array(INT_TYPE n, INT_TYPE m, DATA_TYPE *alpha,
                       DATA_TYPE *beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, N, N, n, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, M, n, m),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, N, M, n, m)) {
  *alpha = 1.5;
  *beta = 1.2;
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < m; j++) {
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i * j + 1) % n) / n;
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)((i * j + 2) % m) / m;
    }
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++) {
      ARRAY_2D_ACCESS(C, i, j) = (DATA_TYPE)((i * j + 3) % n) / m;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, N, N, n, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++) {
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
static void kernel_syr2k(INT_TYPE n, INT_TYPE m, DATA_TYPE alpha,
                         DATA_TYPE beta,
                         ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, N, N, n, n),
                         ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, M, n, m),
                         ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, N, M, n, m)) {
// BLAS PARAMS
// UPLO  = 'L'
// TRANS = 'N'
// A is NxM
// B is NxM
// C is NxN
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
#else // GPU
  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0">(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j <= i; j++)
          C(i, j) *= beta;
        for (INT_TYPE k = 0; k < m; k++)
          for (INT_TYPE j = 0; j <= i; j++) {
            C(i, j) += A(j, k) * alpha * B(i, k) + B(j, k) * alpha * A(i, k);
          }
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;

  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j <= i; j++)
          C(i, j) *= beta;
        for (INT_TYPE k = 0; k < m; k++)
          for (INT_TYPE j = 0; j <= i; j++) {
            C(i, j) += A(j, k) * alpha * B(i, k) + B(j, k) * alpha * A(i, k);
          }
      });
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(A, n, m);
  polybench_GPU_array_2D(B, n, m);
  polybench_GPU_array_2D(C, n, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(B);
  polybench_GPU_array_copy_to_device(C);

  const auto policy_gpu =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);

  Kokkos::parallel_for(
      policy_gpu, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j <= i; j++)
          d_C(i, j) *= beta;
        for (INT_TYPE k = 0; k < m; k++) {
          for (INT_TYPE j = 0; j <= i; j++) {
            d_C(i, j) +=
                d_A(j, k) * alpha * d_B(i, k) + d_B(j, k) * alpha * d_A(i, k);
          }
        }
      });

  polybench_GPU_array_copy_to_host(C);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(C, n, n);
#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < n; i++) {
    for (INT_TYPE j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (INT_TYPE k = 0; k < m; k++)
      for (INT_TYPE j = 0; j <= i; j++) {
        C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k];
      }
  }
#pragma endscop
  polybench_stop_instruments;
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  INT_TYPE n = N;
  INT_TYPE m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, M, n, m);

  /* Initialize array(s). */
  init_array(n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B));

  /* Run kernel. */
  kernel_syr2k(n, m, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A),
               POLYBENCH_ARRAY(B));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  FINALIZE;

  return 0;
}
