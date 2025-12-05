/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* covariance.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"

/* Array initialization. */
static void init_array(int m, int n, DATA_TYPE *float_n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, data, N, M, n, m)) {
  *float_n = (DATA_TYPE)n;

  for (int i = 0; i < N; i++)
    for (int j = 0; j < M; j++)
      ARRAY_2D_ACCESS(data, i, j) = ((DATA_TYPE)i * j) / M;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, cov, M, M, m, m)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(cov, i, j));
    }
  POLYBENCH_DUMP_END("cov");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_covariance(size_t m, size_t n, DATA_TYPE float_n,
                              ARRAY_2D_FUNC_PARAM(DATA_TYPE, data, N, M, n, m),
                              ARRAY_2D_FUNC_PARAM(DATA_TYPE, cov, M, M, m, m),
                              ARRAY_1D_FUNC_PARAM(DATA_TYPE, mean, M, m)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m);
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {n, m});

  Kokkos::parallel_for<usePolyOpt,
                       "p0.l0 == 0, p0.l0 == p2.l0, p0.u0 == p2.u0, p1.l0 == "
                       "0, p1.l1 == 0, p1.u1 == p0.u0, p1.u0 == n, n > 10">(
      "kernel", policy_1D,
      KOKKOS_LAMBDA(const size_t j) {
        mean(j) = SCALAR_VAL(0.0);
        for (size_t i = 0; i < KOKKOS_LOOP_BOUND(n); i++)
          mean(j) += data(i, j);
        mean(j) /= float_n;
      },
      policy_2D,
      KOKKOS_LAMBDA(const size_t i, const size_t j) { data(i, j) -= mean(j); },
      policy_1D,
      KOKKOS_LAMBDA(const size_t i) {
        for (long j = i; j < m; j++) {
          cov(i, j) = SCALAR_VAL(0.0);
          for (long k = 0; k < KOKKOS_LOOP_BOUND(n); k++)
            cov(i, j) += data(k, i) * data(k, j);
          cov(i, j) /= (float_n - SCALAR_VAL(1.0));
          cov(j, i) = cov(i, j);
        }
      });
#elif defined(POLYBENCH_KOKKOS)
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m);
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {n, m});

  Kokkos::parallel_for<usePolyOpt>(
      policy_1D, KOKKOS_LAMBDA(const size_t j) {
        mean(j) = SCALAR_VAL(0.0);
        for (size_t i = 0; i < n; i++)
          mean(j) += data(i, j);
        mean(j) /= float_n;
      });

  Kokkos::parallel_for<usePolyOpt>(
      policy_2D_1,
      KOKKOS_LAMBDA(const size_t i, const size_t j) { data(i, j) -= mean(j); });

  Kokkos::parallel_for<usePolyOpt>(
      policy_1D, KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = i; j < m; j++) {
          cov(i, j) = SCALAR_VAL(0.0);
          for (size_t k = 0; k < n; k++)
            cov(i, j) += data(k, i) * data(k, j);
          cov(i, j) /= (float_n - SCALAR_VAL(1.0));
          cov(j, i) = cov(i, j);
        }
      });
#else
#pragma scop
  for (size_t j = 0; j < m; j++) {
    mean[j] = SCALAR_VAL(0.0);
    for (size_t i = 0; i < n; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }

  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < m; j++)
      data[i][j] -= mean[j];

  for (size_t i = 0; i < m; i++)
    for (size_t j = i; j < m; j++) {
      cov[i][j] = SCALAR_VAL(0.0);
      for (size_t k = 0; k < n; k++)
        cov[i][j] += data[k][i] * data[k][j];
      cov[i][j] /= (float_n - SCALAR_VAL(1.0));
      cov[j][i] = cov[i][j];
    }
#pragma endscop
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data, DATA_TYPE, N, M, n, m);
  POLYBENCH_2D_ARRAY_DECL(cov, DATA_TYPE, M, M, m, m);
  POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array(m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance(m, n, float_n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(cov),
                    POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(cov)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(cov);
  POLYBENCH_FREE_ARRAY(mean);

  FINALIZE;

  return 0;
}
