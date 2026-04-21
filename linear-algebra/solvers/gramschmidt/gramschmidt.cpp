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
static void init_array(INT_TYPE m, INT_TYPE n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, N, m, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, R, N, N, n, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, Q, M, N, m, n)) {
  for (INT_TYPE i = 0; i < m; i++)
    for (INT_TYPE j = 0; j < n; j++) {
      ARRAY_2D_ACCESS(A, i, j) = (((DATA_TYPE)((i * j) % m) / m) * 100) + 10;
      ARRAY_2D_ACCESS(Q, i, j) = 0.0;
    }
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++)
      ARRAY_2D_ACCESS(R, i, j) = 0.0;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE m, INT_TYPE n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, N, m, n),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, R, N, N, n, n),
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, Q, M, N, m, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(R, i, j));
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (INT_TYPE i = 0; i < m; i++)
    for (INT_TYPE j = 0; j < n; j++) {
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
static void kernel_gramschmidt(INT_TYPE m, INT_TYPE n,
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, M, N, m, n),
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, R, N, N, n, n),
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, Q, M, N, m, n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
#else // GPU
  const auto policy_1D =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt,
                       "p0.l0 == 0, p0.u0 == n, p0.u0 > 10, n > 15, m > 5">(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE k) {
        DATA_TYPE nrm = SCALAR_VAL(0.0);
        for (INT_TYPE i = 0; i < KOKKOS_LOOP_BOUND(m); i++)
          nrm += A(i, k) * A(i, k);
        R(k, k) = SQRT_FUN(nrm);
        for (INT_TYPE i = 0; i < KOKKOS_LOOP_BOUND(m); i++)
          Q(i, k) = A(i, k) / R(k, k);
        for (INT_TYPE j = k + 1; j < KOKKOS_LOOP_BOUND(n); j++) {
          R(k, j) = SCALAR_VAL(0.0);
          for (INT_TYPE i = 0; i < KOKKOS_LOOP_BOUND(m); i++)
            R(k, j) += Q(i, k) * A(i, j);
          for (INT_TYPE i = 0; i < KOKKOS_LOOP_BOUND(m); i++)
            A(i, j) = A(i, j) - Q(i, k) * R(k, j);
        }
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m);
  for (INT_TYPE k = 0; k < n; k++) {
    DATA_TYPE nrm = DATA_TYPE(0);
    Kokkos::parallel_reduce(
        policy_1D,
        KOKKOS_LAMBDA(INT_TYPE i, DATA_TYPE & local_nrm) {
          local_nrm += A(i, k) * A(i, k);
        },
        nrm);

    R(k, k) = SQRT_FUN(nrm);

    Kokkos::parallel_for(
        policy_1D, KOKKOS_LAMBDA(INT_TYPE i) { Q(i, k) = A(i, k) / R(k, k); });

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(k + 1, n),
        KOKKOS_LAMBDA(INT_TYPE j) {
          R(k, j) = SCALAR_VAL(0.0);
          for (INT_TYPE i = 0; i < m; i++)
            R(k, j) += Q(i, k) * A(i, j);
          for (INT_TYPE i = 0; i < m; i++)
            A(i, j) = A(i, j) - Q(i, k) * R(k, j);
        });
  }
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(A, m, n);
  polybench_GPU_array_2D(R, n, n);
  polybench_GPU_array_2D(Q, m, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(R);
  polybench_GPU_array_copy_to_device(Q);

  auto policy_1D_m =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, m);

  for (INT_TYPE k = 0; k < n; k++) {
    DATA_TYPE nrm = SCALAR_VAL(0.0);

    Kokkos::parallel_reduce(
        policy_1D_m,
        KOKKOS_LAMBDA(const INT_TYPE i, DATA_TYPE &local_nrm) {
          local_nrm += d_A(i, k) * d_A(i, k);
        },
        nrm);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, 1),
        KOKKOS_LAMBDA(const INT_TYPE dummy) { d_R(k, k) = SQRT_FUN(nrm); });

    Kokkos::parallel_for(
        policy_1D_m,
        KOKKOS_LAMBDA(const INT_TYPE i) { d_Q(i, k) = d_A(i, k) / d_R(k, k); });

    if (k + 1 < n) {
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(k + 1,
                                                                        n),
          KOKKOS_LAMBDA(const INT_TYPE j) {
            DATA_TYPE local_r = SCALAR_VAL(0.0);

            for (INT_TYPE i = 0; i < m; i++) {
              local_r += d_Q(i, k) * d_A(i, j);
            }
            d_R(k, j) = local_r;

            for (INT_TYPE i = 0; i < m; i++) {
              d_A(i, j) -= d_Q(i, k) * local_r;
            }
          });
    }
  }

  polybench_GPU_array_copy_to_host(A);
  polybench_GPU_array_copy_to_host(R);
  polybench_GPU_array_copy_to_host(Q);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(A, m, n);
  polybench_GPU_array_sync_2D(R, n, n);
  polybench_GPU_array_sync_2D(Q, m, n);

#endif
#else
  DATA_TYPE nrm;
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE k = 0; k < n; k++) {
    nrm = SCALAR_VAL(0.0);

    for (INT_TYPE i = 0; i < m; i++)
      nrm += A[i][k] * A[i][k];

    R[k][k] = SQRT_FUN(nrm);

    for (INT_TYPE i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];

    for (INT_TYPE j = k + 1; j < n; j++) {
      R[k][j] = SCALAR_VAL(0.0);
      for (INT_TYPE i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (INT_TYPE i = 0; i < m; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
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
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_2D_ARRAY_DECL(R, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(Q, DATA_TYPE, M, N, m, n);

  /* Initialize array(s). */
  init_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

  /* Run kernel. */
  kernel_gramschmidt(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R),
                     POLYBENCH_ARRAY(Q));

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
