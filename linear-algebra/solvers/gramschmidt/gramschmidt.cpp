/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gramschmidt.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gramschmidt.h"

/* Array initialization. */
static void init_array(int m, int n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, N, m, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, R, N, N, n, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, Q, M, N, m, n)) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      ARRAY_2D_ACCESS(A, i, j) = (((DATA_TYPE)((i * j) % m) / m) * 100) + 10;
      ARRAY_2D_ACCESS(Q, i, j) = 0.0;
    }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      ARRAY_2D_ACCESS(R, i, j) = 0.0;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m, int n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, N, m, n),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, R, N, N, n, n),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, Q, M, N, m, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(R, i, j));
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(Q, i, j));
    }
  POLYBENCH_DUMP_END("Q");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* QR Decomposition with Modified Gram Schmidt:
 http://www.inf.ethz.ch/personal/gander/ */
static void kernel_gramschmidt(int m, int n,
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, N, m, n),
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, R, N, N, n, n),
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, Q, M, N, m, n)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::Serial>(0, n);

  Kokkos::parallel_for<usePolyOpt>(
      policy_1D, KOKKOS_LAMBDA(const int k) {
        DATA_TYPE nrm = SCALAR_VAL(0.0);
        for (int i = 0; i < m; i++)
          nrm += A(i, k) * A(i, k);
        R(k, k) = SQRT_FUN(nrm);
        for (int i = 0; i < m; i++)
          Q(i, k) = A(i, k) / R(k, k);
        for (int j = k + 1; j < n; j++) {
          R(k, j) = SCALAR_VAL(0.0);
          for (int i = 0; i < m; i++)
            R(k, j) += Q(i, k) * A(i, j);
          for (int i = 0; i < m; i++)
            A(i, j) = A(i, j) - Q(i, k) * R(k, j);
        }
      });
#elif defined(POLYBENCH_KOKKOS)
  for (int k = 0; k < N; k++) {
    DATA_TYPE nrm = DATA_TYPE(0);
    Kokkos::parallel_reduce(
        "Compute_Norm", Kokkos::RangePolicy<>(0, M),
        KOKKOS_LAMBDA(int i, DATA_TYPE &local_nrm) {
          local_nrm += A(i, k) * A(i, k);
        },
        nrm);
    R(k, k) = SQRT_FUN(nrm);

    Kokkos::parallel_for(
        "Normalize_Column", Kokkos::RangePolicy<>(0, M),
        KOKKOS_LAMBDA(int i) { Q(i, k) = A(i, k) / R(k, k); });

    Kokkos::parallel_for(
        "Update_A_R", Kokkos::RangePolicy<>(k + 1, N), KOKKOS_LAMBDA(int j) {
          for (int i = 0; i < _PB_M; i++)
            R(k, j) += Q(i, k) * A(i, j);
          for (int i = 0; i < _PB_M; i++)
            A(i, j) = A(i, j) - Q(i, k) * R(k, j);
        });
  }
#else
  DATA_TYPE nrm;

#pragma scop
  for (int k = 0; k < _PB_N; k++) {
    nrm = SCALAR_VAL(0.0);
    for (int i = 0; i < _PB_M; i++)
      nrm += A[i][k] * A[i][k];
    R[k][k] = SQRT_FUN(nrm);
    for (int i = 0; i < _PB_M; i++)
      Q[i][k] = A[i][k] / R[k][k];
    for (int j = k + 1; j < _PB_N; j++) {
      R[k][j] = SCALAR_VAL(0.0);
      for (int i = 0; i < _PB_M; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (int i = 0; i < _PB_M; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
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
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_2D_ARRAY_DECL(R, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(Q, DATA_TYPE, M, N, m, n);

  /* Initialize array(s). */
  init_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gramschmidt(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R),
                     POLYBENCH_ARRAY(Q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(A),
                                    POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  FINALIZE;

  return 0;
}
