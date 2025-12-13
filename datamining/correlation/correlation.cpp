/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* correlation.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "correlation.h"

/* Array initialization. */
static void init_array(int m, int n, DATA_TYPE *float_n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, data, N, M, n, m)) {
  *float_n = (DATA_TYPE)N;

  for (int i = 0; i < N; i++)
    for (int j = 0; j < M; j++)
      ARRAY_2D_ACCESS(data, i, j) = (DATA_TYPE)(i * j) / M + i;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, corr, M, M, m, m)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(corr, i, j));
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_correlation(size_t m, size_t n, DATA_TYPE float_n,
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, data, N, M, n, m),
                               ARRAY_2D_FUNC_PARAM(DATA_TYPE, corr, M, M, m, m),
                               ARRAY_1D_FUNC_PARAM(DATA_TYPE, mean, M, m),
                               ARRAY_1D_FUNC_PARAM(DATA_TYPE, stddev, M, m)) {
  DATA_TYPE eps = SCALAR_VAL(0.1);

#if defined(POLYBENCH_USE_POLLY)
  const auto policy_1D_1 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m);
  const auto policy_1D_2 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m - 1);
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {n, m});

  Kokkos::parallel_for<Kokkos::usePolyOpt>(
      "kernel", policy_1D_1,
      KOKKOS_LAMBDA(const size_t j) {
        mean(j) = SCALAR_VAL(0.0);
        for (size_t i = 0; i < KOKKOS_LOOP_BOUND(n); i++)
          mean(j) += data(i, j);
        mean(j) /= float_n;
      },
      policy_1D_1,
      KOKKOS_LAMBDA(const size_t j) {
        stddev(j) = SCALAR_VAL(0.0);
        for (size_t i = 0; i < KOKKOS_LOOP_BOUND(n); i++)
          stddev(j) += (data(i, j) - mean(j)) * (data(i, j) - mean(j));
        stddev(j) /= float_n;
        stddev(j) = SQRT_FUN(stddev(j));
        /* The following in an inelegant but usual way to handle
           near-zero std. dev. values, which below would cause a zero-
           divide. */
        stddev(j) = stddev(j) <= eps ? SCALAR_VAL(1.0) : stddev(j);
      },
      policy_2D,
      KOKKOS_LAMBDA(const size_t i, const size_t j) {
        data(i, j) -= mean(j);
        data(i, j) /= SQRT_FUN(float_n) * stddev(j);
      },
      policy_1D_2,
      KOKKOS_LAMBDA(const size_t i) {
        corr(i, i) = SCALAR_VAL(1.0);
        for (size_t j = i + 1; j < KOKKOS_LOOP_BOUND(m); j++) {
          corr(i, j) = SCALAR_VAL(0.0);
          for (size_t k = 0; k < KOKKOS_LOOP_BOUND(n); k++)
            corr(i, j) += (data(k, i) * data(k, j));
          corr(j, i) = corr(i, j);
        }
      });
  corr(m - 1, m - 1) = SCALAR_VAL(1.0);
#elif defined(POLYBENCH_KOKKOS)

  const auto policy_1D_1 = Kokkos::RangePolicy<>(0, m);
  const auto policy_1D_2 = Kokkos::RangePolicy<>(0, m - 1);
  const auto policy_2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, m});

  Kokkos::parallel_for(
      policy_1D_1, KOKKOS_LAMBDA(const size_t j) {
        mean(j) = SCALAR_VAL(0.0);
        for (size_t i = 0; i < n; i++)
          mean(j) += data(i, j);
        mean(j) /= float_n;
      });

  Kokkos::parallel_for(
      policy_1D_1, KOKKOS_LAMBDA(const size_t j) {
        stddev(j) = SCALAR_VAL(0.0);
        for (size_t i = 0; i < n; i++)
          stddev(j) += (data(i, j) - mean(j)) * (data(i, j) - mean(j));
        stddev(j) /= float_n;
        stddev(j) = SQRT_FUN(stddev(j));
        /* The following in an inelegant but usual way to handle
           near-zero std. dev. values, which below would cause a zero-
           divide. */
        stddev(j) = stddev(j) <= eps ? SCALAR_VAL(1.0) : stddev(j);
      });

  /* Center and reduce the column vectors. */
  Kokkos::parallel_for(
      policy_2D, KOKKOS_LAMBDA(const size_t i, const size_t j) {
        data(i, j) -= mean(j);
        data(i, j) /= SQRT_FUN(float_n) * stddev(j);
      });

  /* Calculate the m * m correlation matrix. */
  Kokkos::parallel_for(
      policy_1D_2, KOKKOS_LAMBDA(const size_t i) {
        corr(i, i) = SCALAR_VAL(1.0);
        for (size_t j = i + 1; j < m; j++) {
          corr(i, j) = SCALAR_VAL(0.0);
          for (size_t k = 0; k < n; k++)
            corr(i, j) += (data(k, i) * data(k, j));
          corr(j, i) = corr(i, j);
        }
      });
  corr(m - 1, m - 1) = SCALAR_VAL(1.0);
#else
#pragma scop
  for (size_t j = 0; j < m; j++) {
    mean[j] = SCALAR_VAL(0.0);
    for (size_t i = 0; i < n; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }

  for (size_t j = 0; j < m; j++) {
    stddev[j] = SCALAR_VAL(0.0);
    for (size_t i = 0; i < n; i++)
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    stddev[j] /= float_n;
    stddev[j] = SQRT_FUN(stddev[j]);
    /* The following in an inelegant but usual way to handle
       near-zero std. dev. values, which below would cause a zero-
       divide. */
    stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
  }

  /* Center and reduce the column vectors. */
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < m; j++) {
      data[i][j] -= mean[j];
      data[i][j] /= SQRT_FUN(float_n) * stddev[j];
    }

  /* Calculate the m * m correlation matrix. */
  for (size_t i = 0; i < m - 1; i++) {
    corr[i][i] = SCALAR_VAL(1.0);
    for (size_t j = i + 1; j < m; j++) {
      corr[i][j] = SCALAR_VAL(0.0);
      for (size_t k = 0; k < n; k++)
        corr[i][j] += (data[k][i] * data[k][j]);
      corr[j][i] = corr[i][j];
    }
  }
  corr[m - 1][m - 1] = SCALAR_VAL(1.0);
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
  POLYBENCH_2D_ARRAY_DECL(corr, DATA_TYPE, M, M, m, m);
  POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(stddev, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array(m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_correlation(m, n, float_n, POLYBENCH_ARRAY(data),
                     POLYBENCH_ARRAY(corr), POLYBENCH_ARRAY(mean),
                     POLYBENCH_ARRAY(stddev));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  FINALIZE;

  return 0;
}
