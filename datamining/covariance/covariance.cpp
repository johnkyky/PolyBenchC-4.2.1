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
static void init_array(INT_TYPE m, INT_TYPE n, DATA_TYPE *float_n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, data, N, M, n, m)) {
  *float_n = (DATA_TYPE)n;

  for (INT_TYPE i = 0; i < N; i++)
    for (INT_TYPE j = 0; j < M; j++)
      ARRAY_2D_ACCESS(data, i, j) = ((DATA_TYPE)i * j) / M;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE m,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, cov, M, M, m, m)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (INT_TYPE i = 0; i < m; i++)
    for (INT_TYPE j = 0; j < m; j++) {
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
static void kernel_covariance(INT_TYPE m, INT_TYPE n, DATA_TYPE float_n,
                              ARRAY_2D_FUNC_PARAM(DATA_TYPE, data, N, M, n, m),
                              ARRAY_2D_FUNC_PARAM(DATA_TYPE, cov, M, M, m, m),
                              ARRAY_1D_FUNC_PARAM(DATA_TYPE, mean, M, m)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m);
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {n, m});
#else // GPU
  const auto policy_1D =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, m);
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {n, m});
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0. == p2., p0.l0 == 0, p0.u0 == m,"
                                           "p1.l == 0, p1.u0 == n, p1.u1 == m,"
                                           "n > 10">(
      "kernel", policy_1D,
      KOKKOS_LAMBDA(const INT_TYPE j) {
        mean(j) = SCALAR_VAL(0.0);
        for (INT_TYPE i = 0; i < KOKKOS_LOOP_BOUND(n); i++)
          mean(j) += data(i, j);
        mean(j) /= float_n;
      },
      policy_2D,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        data(i, j) -= mean(j);
      },
      policy_1D,
      KOKKOS_LAMBDA(const INT_TYPE i) {
        for (long j = i; j < KOKKOS_LOOP_BOUND(m); j++) {
          cov(i, j) = SCALAR_VAL(0.0);
          for (long k = 0; k < KOKKOS_LOOP_BOUND(n); k++)
            cov(i, j) += data(k, i) * data(k, j);
          cov(i, j) /= (float_n - SCALAR_VAL(1.0));
          cov(j, i) = cov(i, j);
        }
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m);
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {n, m});

  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE j) {
        mean(j) = SCALAR_VAL(0.0);
        for (INT_TYPE i = 0; i < n; i++)
          mean(j) += data(i, j);
        mean(j) /= float_n;
      });

  Kokkos::parallel_for(
      policy_2D_1, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        data(i, j) -= mean(j);
      });

  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = i; j < m; j++) {
          cov(i, j) = SCALAR_VAL(0.0);
          for (INT_TYPE k = 0; k < n; k++)
            cov(i, j) += data(k, i) * data(k, j);
          cov(i, j) /= (float_n - SCALAR_VAL(1.0));
          cov(j, i) = cov(i, j);
        }
      });
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(data, n, m);
  polybench_GPU_array_2D(cov, m, m);
  polybench_GPU_array_1D(mean, m);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(data);
  polybench_GPU_array_copy_to_device(cov);
  polybench_GPU_array_copy_to_device(mean);

  const auto policy_1D =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, m);
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {n, m});

  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE j) {
        d_mean(j) = SCALAR_VAL(0.0);
        for (INT_TYPE i = 0; i < n; i++)
          d_mean(j) += d_data(i, j);
        d_mean(j) /= float_n;
      });

  Kokkos::parallel_for(
      policy_2D_1, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        d_data(i, j) -= d_mean(j);
      });

  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = i; j < m; j++) {
          d_cov(i, j) = SCALAR_VAL(0.0);
          for (INT_TYPE k = 0; k < n; k++)
            d_cov(i, j) += d_data(k, i) * d_data(k, j);
          d_cov(i, j) /= (float_n - SCALAR_VAL(1.0));
          d_cov(j, i) = d_cov(i, j);
        }
      });

  polybench_GPU_array_copy_to_host(data);
  polybench_GPU_array_copy_to_host(cov);
  polybench_GPU_array_copy_to_host(mean);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(data, n, m);
  polybench_GPU_array_sync_2D(cov, m, m);
  polybench_GPU_array_sync_1D(mean, m);

#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE j = 0; j < m; j++) {
    mean[j] = SCALAR_VAL(0.0);
    for (INT_TYPE i = 0; i < n; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }

  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < m; j++)
      data[i][j] -= mean[j];

  for (INT_TYPE i = 0; i < m; i++)
    for (INT_TYPE j = i; j < m; j++) {
      cov[i][j] = SCALAR_VAL(0.0);
      for (INT_TYPE k = 0; k < n; k++)
        cov[i][j] += data[k][i] * data[k][j];
      cov[i][j] /= (float_n - SCALAR_VAL(1.0));
      cov[j][i] = cov[i][j];
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
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data, DATA_TYPE, N, M, n, m);
  POLYBENCH_2D_ARRAY_DECL(cov, DATA_TYPE, M, M, m, m);
  POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array(m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Run kernel. */
  kernel_covariance(m, n, float_n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(cov),
                    POLYBENCH_ARRAY(mean));

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
