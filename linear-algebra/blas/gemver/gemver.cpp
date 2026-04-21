/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"

/* Array initialization. */
static void init_array(INT_TYPE n, DATA_TYPE *alpha, DATA_TYPE *beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, u1, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, v1, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, u2, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, v2, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, w, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, z, N, n)) {
  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (INT_TYPE i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(u1, i) = i;
    ARRAY_1D_ACCESS(u2, i) = ((i + 1) / fn) / 2.0;
    ARRAY_1D_ACCESS(v1, i) = ((i + 1) / fn) / 4.0;
    ARRAY_1D_ACCESS(v2, i) = ((i + 1) / fn) / 6.0;
    ARRAY_1D_ACCESS(y, i) = ((i + 1) / fn) / 8.0;
    ARRAY_1D_ACCESS(z, i) = ((i + 1) / fn) / 9.0;
    ARRAY_1D_ACCESS(x, i) = 0.0;
    ARRAY_1D_ACCESS(w, i) = 0.0;
    for (INT_TYPE j = 0; j < n; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)(i * j % n) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, w, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (INT_TYPE i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(w, i));
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_gemver(INT_TYPE n, DATA_TYPE alpha, DATA_TYPE beta,
                          ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, u1, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, v1, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, u2, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, v2, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, w, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, z, N, n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {n, n});
#else // GPU
  const auto policy_1D =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {n, n});
#endif

  // Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l==0, p0.u0==p0.u1,"
  //                                          "p0==p1,p0==p3,"
  //                                          "p2.l==0, p2.u0==p0.u0">(
  Kokkos::parallel_for<Kokkos::usePolyOpt,
                       "p0.l==0, p0.u0==p0.u1, p0.u0>10,"
                       "p1.l==0, p1.u0==p1.u1, p0.u0==p1.u0,"
                       "p3.l==0, p3.u0==p3.u1, p0.u0==p3.u0,"
                       "p2.l==0, p2.u0==p0.u0">(
      "kernel", policy_2D,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        A(i, j) += u1(i) * v1(j) + u2(i) * v2(j);
      },
      policy_2D,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        x(i) += beta * A(j, i) * y(j);
      },
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) { x(i) += z(i); }, policy_2D,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        w(i) += alpha * A(i, j) * x(j);
      });
  polybench_stop_instruments;

#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
  const auto policy_2D = Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>(
      {0, 0}, {n, n}, {32, 32});

  Kokkos::parallel_for(
      policy_2D, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        A(i, j) += u1(i) * v1(j) + u2(i) * v2(j);
      });
  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < n; j++)
          x(i) += beta * A(j, i) * y(j);
      });
  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) { x(i) += z(i); });
  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < n; j++)
          w(i) += alpha * A(i, j) * x(j);
      });
  polybench_stop_instruments;
#else
  polybench_GPU_array_2D(A, n, n);
  polybench_GPU_array_1D(u1, n);
  polybench_GPU_array_1D(v1, n);
  polybench_GPU_array_1D(u2, n);
  polybench_GPU_array_1D(v2, n);
  polybench_GPU_array_1D(w, n);
  polybench_GPU_array_1D(x, n);
  polybench_GPU_array_1D(y, n);
  polybench_GPU_array_1D(z, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(u1);
  polybench_GPU_array_copy_to_device(v1);
  polybench_GPU_array_copy_to_device(u2);
  polybench_GPU_array_copy_to_device(v2);
  polybench_GPU_array_copy_to_device(w);
  polybench_GPU_array_copy_to_device(x);
  polybench_GPU_array_copy_to_device(y);
  polybench_GPU_array_copy_to_device(z);

  const auto policy_1D =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {n, n});

  Kokkos::parallel_for(
      policy_2D, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        d_A(i, j) += d_u1(i) * d_v1(j) + d_u2(i) * d_v2(j);
      });
  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < n; j++)
          d_x(i) += beta * d_A(j, i) * d_y(j);
      });
  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) { d_x(i) += d_z(i); });
  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < n; j++)
          d_w(i) += alpha * d_A(i, j) * d_x(j);
      });

  polybench_GPU_array_copy_to_host(A);
  polybench_GPU_array_copy_to_host(x);
  polybench_GPU_array_copy_to_host(w);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(A, n, n);
  polybench_GPU_array_sync_1D(x, n);
  polybench_GPU_array_sync_1D(w, n);
#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++)
      A[i][j] += u1[i] * v1[j] + u2[i] * v2[j];

  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++)
      x[i] += beta * A[j][i] * y[j];

  for (INT_TYPE i = 0; i < n; i++)
    x[i] += z[i];

  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++)
      w[i] += alpha * A[i][j] * x[j];
#pragma endscop
  polybench_stop_instruments;
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  INT_TYPE n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1),
             POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2),
             POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y),
             POLYBENCH_ARRAY(z));

  /* Run kernel. */
  kernel_gemver(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1),
                POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2),
                POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y),
                POLYBENCH_ARRAY(z));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(u1);
  POLYBENCH_FREE_ARRAY(v1);
  POLYBENCH_FREE_ARRAY(u2);
  POLYBENCH_FREE_ARRAY(v2);
  POLYBENCH_FREE_ARRAY(w);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(z);

  FINALIZE;

  return 0;
}
