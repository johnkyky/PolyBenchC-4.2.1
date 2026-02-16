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
static void kernel_gramschmidt(size_t m, size_t n,
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, N, m, n),
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, R, N, N, n, n),
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, Q, M, N, m, n)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::Serial>(0, n);

  Kokkos::parallel_for<Kokkos::usePolyOpt,
                       "p0.l0 == 0, p0.u0 == n, p0.u0 > 10, n > 15, m > 5">(
      policy_1D, KOKKOS_LAMBDA(const size_t k) {
        DATA_TYPE nrm = SCALAR_VAL(0.0);
        for (size_t i = 0; i < KOKKOS_LOOP_BOUND(m); i++)
          nrm += A(i, k) * A(i, k);
        R(k, k) = SQRT_FUN(nrm);
        for (size_t i = 0; i < KOKKOS_LOOP_BOUND(m); i++)
          Q(i, k) = A(i, k) / R(k, k);
        for (size_t j = k + 1; j < KOKKOS_LOOP_BOUND(n); j++) {
          R(k, j) = SCALAR_VAL(0.0);
          for (size_t i = 0; i < KOKKOS_LOOP_BOUND(m); i++)
            R(k, j) += Q(i, k) * A(i, j);
          for (size_t i = 0; i < KOKKOS_LOOP_BOUND(m); i++)
            A(i, j) = A(i, j) - Q(i, k) * R(k, j);
        }
      });
#elif defined(POLYBENCH_KOKKOS)
  auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m);
  for (size_t k = 0; k < n; k++) {
    DATA_TYPE nrm = DATA_TYPE(0);
    Kokkos::parallel_reduce(
        policy_1D,
        KOKKOS_LAMBDA(size_t i, DATA_TYPE &local_nrm) {
          local_nrm += A(i, k) * A(i, k);
        },
        nrm);

    R(k, k) = SQRT_FUN(nrm);

    Kokkos::parallel_for(
        policy_1D, KOKKOS_LAMBDA(size_t i) { Q(i, k) = A(i, k) / R(k, k); });

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(k + 1, n), KOKKOS_LAMBDA(size_t j) {
          R(k, j) = SCALAR_VAL(0.0);
          for (size_t i = 0; i < m; i++)
            R(k, j) += Q(i, k) * A(i, j);
          for (size_t i = 0; i < m; i++)
            A(i, j) = A(i, j) - Q(i, k) * R(k, j);
        });
  }
#else
  DATA_TYPE nrm;

#pragma scop
  for (size_t k = 0; k < n; k++) {
    nrm = SCALAR_VAL(0.0);

    for (size_t i = 0; i < m; i++)
      nrm += A[i][k] * A[i][k];

    R[k][k] = SQRT_FUN(nrm);

    for (size_t i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];

    for (size_t j = k + 1; j < n; j++) {
      R[k][j] = SCALAR_VAL(0.0);
      for (size_t i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (size_t i = 0; i < m; i++)
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
