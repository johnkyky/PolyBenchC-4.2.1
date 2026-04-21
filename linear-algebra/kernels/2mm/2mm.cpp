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
static void init_array(INT_TYPE ni, INT_TYPE nj, INT_TYPE nk, INT_TYPE nl,
                       DATA_TYPE *alpha, DATA_TYPE *beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NJ, NL, nj, nl),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, D, NI, NL, ni, nl)) {
  *alpha = 1.5;
  *beta = 1.2;
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nk; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i * j + 1) % ni) / ni;
  for (INT_TYPE i = 0; i < nk; i++)
    for (INT_TYPE j = 0; j < nj; j++)
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)(i * (j + 1) % nj) / nj;
  for (INT_TYPE i = 0; i < nj; i++)
    for (INT_TYPE j = 0; j < nl; j++)
      ARRAY_2D_ACCESS(C, i, j) = (DATA_TYPE)((i * (j + 3) + 1) % nl) / nl;
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nl; j++)
      ARRAY_2D_ACCESS(D, i, j) = (DATA_TYPE)(i * (j + 2) % nk) / nk;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE ni, INT_TYPE nl,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, D, NI, NL, ni, nl)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("D");
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nl; j++) {
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
static void kernel_2mm(INT_TYPE ni, INT_TYPE nj, INT_TYPE nk, INT_TYPE nl,
                       DATA_TYPE alpha, DATA_TYPE beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, tmp, NI, NJ, ni, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NJ, NL, nj, nl),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, D, NI, NL, ni, nl)) {

#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, ni);
#else // GPU
  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, ni);
#endif

  Kokkos::parallel_for<
      Kokkos::usePolyOpt,
      "p0.l0 == 0, p1.l0 == 0, p0.u0 == p1.u0, p0.u0 > 10, 12 < nj">(
      "kernel", policy,
      KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < KOKKOS_LOOP_BOUND(nj); j++) {
          tmp(i, j) = SCALAR_VAL(0.0);
          for (INT_TYPE k = 0; k < KOKKOS_LOOP_BOUND(nk); ++k)
            tmp(i, j) += alpha * A(i, k) * B(k, j);
        }
      },
      policy,
      KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < KOKKOS_LOOP_BOUND(nl); j++) {
          D(i, j) *= beta;
          for (INT_TYPE k = 0; k < KOKKOS_LOOP_BOUND(nj); ++k)
            D(i, j) += tmp(i, k) * C(k, j);
        }
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {ni, nj},
                                                             {32, 32});
  const auto policy_2D_2 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {ni, nl},
                                                             {32, 32});

  /* D := alpha*A*B*C + beta*D */
  Kokkos::parallel_for(
      policy_2D_1, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        tmp(i, j) = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < nk; ++k)
          tmp(i, j) += alpha * A(i, k) * B(k, j);
      });

  Kokkos::parallel_for(
      policy_2D_2, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        D(i, j) *= beta;
        for (INT_TYPE k = 0; k < nj; ++k)
          D(i, j) += tmp(i, k) * C(k, j);
      });
  polybench_stop_instruments;
#else
  polybench_GPU_array_2D(A, ni, nk);
  polybench_GPU_array_2D(B, nk, nj);
  polybench_GPU_array_2D(C, nj, nl);
  polybench_GPU_array_2D(D, ni, nl);
  polybench_GPU_array_2D(tmp, ni, nj);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(B);
  polybench_GPU_array_copy_to_device(C);
  polybench_GPU_array_copy_to_device(D);

  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {ni, nj});
  const auto policy_2D_2 =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {ni, nl});

  // Premier kernel : tmp = alpha * A * B
  Kokkos::parallel_for(
      policy_2D_1, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        DATA_TYPE local_tmp = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < nk; ++k) {
          local_tmp += alpha * d_A(i, k) * d_B(k, j);
        }
        d_tmp(i, j) = local_tmp;
      });

  // Deuxième kernel : D = beta * D + tmp * C
  Kokkos::parallel_for(
      policy_2D_2, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        DATA_TYPE local_sum = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < nj; ++k) {
          local_sum += d_tmp(i, k) * d_C(k, j);
        }
        d_D(i, j) = beta * d_D(i, j) + local_sum;
      });

  polybench_GPU_array_copy_to_host(tmp);
  polybench_GPU_array_copy_to_host(D);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(tmp, ni, nj);
  polybench_GPU_array_sync_2D(D, ni, nl);

#endif
#else
  polybench_start_instruments;
#pragma scop
  /* D := alpha*A*B*C + beta*D */
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nj; j++) {
      tmp[i][j] = SCALAR_VAL(0.0);
      for (INT_TYPE k = 0; k < nk; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
    }
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nl; j++) {
      D[i][j] *= beta;
      for (INT_TYPE k = 0; k < nj; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
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
  INT_TYPE nl = NL;

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

  /* Run kernel. */
  kernel_2mm(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp),
             POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C),
             POLYBENCH_ARRAY(D));

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
