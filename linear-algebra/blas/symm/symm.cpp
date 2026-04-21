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
static void init_array(INT_TYPE m, INT_TYPE n, DATA_TYPE *alpha,
                       DATA_TYPE *beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, M, N, m, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, M, m, m),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, M, N, m, n)) {
  *alpha = 1.5;
  *beta = 1.2;
  for (INT_TYPE i = 0; i < m; i++)
    for (INT_TYPE j = 0; j < n; j++) {
      ARRAY_2D_ACCESS(C, i, j) = (DATA_TYPE)((i + j) % 100) / m;
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)((n + i - j) % 100) / m;
    }
  for (INT_TYPE i = 0; i < m; i++) {
    for (INT_TYPE j = 0; j <= i; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i + j) % 100) / m;
    for (INT_TYPE j = i + 1; j < m; j++)
      ARRAY_2D_ACCESS(A, i, j) =
          -999; // regions of arrays that should not be used
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE m, INT_TYPE n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, M, N, m, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (INT_TYPE i = 0; i < m; i++)
    for (INT_TYPE j = 0; j < n; j++) {
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
static void kernel_symm(INT_TYPE m, INT_TYPE n, DATA_TYPE alpha, DATA_TYPE beta,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, M, N, m, n),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, M, m, m),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, M, N, m, n),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, tmp, M, N, m, n)) {
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
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {m, n});
#else // GPU
  const auto policy =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {m, n});
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l==0, p0.u0>10, p0.u1>15">(
      policy, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        tmp(i, j) = 0;
        for (INT_TYPE k = 0; k < i; k++) {
          C(k, j) += alpha * B(i, j) * A(i, k);
          tmp(i, j) += B(k, j) * A(i, k);
        }
        C(i, j) =
            beta * C(i, j) + alpha * B(i, j) * A(i, i) + alpha * tmp(i, j);
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);

  for (INT_TYPE i = 0; i < m; i++) {
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const INT_TYPE j) {
          tmp(i, j) = 0;
          for (INT_TYPE k = 0; k < i; k++) {
            C(k, j) += alpha * B(i, j) * A(i, k);
            tmp(i, j) += B(k, j) * A(i, k);
          }
          C(i, j) =
              beta * C(i, j) + alpha * B(i, j) * A(i, i) + alpha * tmp(i, j);
        });
  }
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(A, m, m);
  polybench_GPU_array_2D(B, m, n);
  polybench_GPU_array_2D(C, m, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(B);
  polybench_GPU_array_copy_to_device(C);

  const auto policy_1d =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);

  for (INT_TYPE i = 0; i < m; i++) {
    Kokkos::parallel_for(
        policy_1d, KOKKOS_LAMBDA(const INT_TYPE j) {
          d_tmp(i, j) = SCALAR_VAL(0.0);
          for (INT_TYPE k = 0; k < i; k++) {
            d_C(k, j) += alpha * d_B(i, j) * d_A(i, k);
            d_tmp(i, j) += d_B(k, j) * d_A(i, k);
          }
          d_C(i, j) = beta * d_C(i, j) + alpha * d_B(i, j) * d_A(i, i) +
                      alpha * d_tmp(i, j);
        });
  }

  polybench_GPU_array_copy_to_host(C);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(C, m, n);
#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < m; i++) {
    for (INT_TYPE j = 0; j < n; j++) {
      tmp[i][j] = 0;
      for (INT_TYPE k = 0; k < i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        tmp[i][j] += B[k][j] * A[i][k];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * tmp[i][j];
    }
  }
#pragma endscop
  polybench_stop_instruments;
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  INT_TYPE m = M;
  INT_TYPE n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, M, N, m, n);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, M, m, m);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, M, N, m, n);
  POLYBENCH_2D_ARRAY_DECL(tmp, DATA_TYPE, M, N, m, n);

  /* Initialize array(s). */
  init_array(m, n, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B));

  /* Run kernel. */
  kernel_symm(m, n, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A),
              POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp));

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
