/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gesummv.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gesummv.h"

/* Array initialization. */
static void init_array(INT_TYPE n, DATA_TYPE *alpha, DATA_TYPE *beta,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, N, N, n, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n)) {
  *alpha = 1.5;
  *beta = 1.2;
  for (INT_TYPE i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(x, i) = (DATA_TYPE)(i % n) / n;
    for (INT_TYPE j = 0; j < n; j++) {
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i * j + 1) % n) / n;
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)((i * j + 2) % n) / n;
    }
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (INT_TYPE i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(y, i));
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_gesummv(INT_TYPE n, DATA_TYPE alpha, DATA_TYPE beta,
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n),
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, N, N, n, n),
                           ARRAY_1D_FUNC_PARAM(DATA_TYPE, tmp, N, n),
                           ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                           ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
#else // GPU
  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0, p0.u0 == n">(
      "kernel", policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        tmp(i) = SCALAR_VAL(0.0);
        y(i) = SCALAR_VAL(0.0);
        for (INT_TYPE j = 0; j < KOKKOS_LOOP_BOUND(n); j++) {
          tmp(i) = A(i, j) * x(j) + tmp(i);
          y(i) = B(i, j) * x(j) + y(i);
        }
        y(i) = alpha * tmp(i) + beta * y(i);
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        tmp(i) = SCALAR_VAL(0.0);
        y(i) = SCALAR_VAL(0.0);
        for (INT_TYPE j = 0; j < n; j++) {
          tmp(i) = A(i, j) * x(j) + tmp(i);
          y(i) = B(i, j) * x(j) + y(i);
        }
        y(i) = alpha * tmp(i) + beta * y(i);
      });
  polybench_stop_instruments;
#else
  polybench_GPU_array_2D(A, n, n);
  polybench_GPU_array_2D(B, n, n);
  polybench_GPU_array_1D(x, n);
  polybench_GPU_array_1D(y, n);
  polybench_GPU_array_1D(tmp, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(B);
  polybench_GPU_array_copy_to_device(x);

  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        d_tmp(i) = SCALAR_VAL(0.0);
        d_y(i) = SCALAR_VAL(0.0);
        for (INT_TYPE j = 0; j < n; j++) {
          d_tmp(i) += d_A(i, j) * d_x(j);
          d_y(i) += d_B(i, j) * d_x(j);
        }
        d_y(i) = alpha * d_tmp(i) + beta * d_y(i);
      });

  polybench_GPU_array_copy_to_host(y);
  polybench_GPU_array_copy_to_host(tmp);

  polybench_stop_instruments;

  polybench_GPU_array_sync_1D(y, n);
  polybench_GPU_array_sync_1D(tmp, n);

#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < n; i++) {
    tmp[i] = SCALAR_VAL(0.0);
    y[i] = SCALAR_VAL(0.0);
    for (INT_TYPE j = 0; j < n; j++) {
      tmp[i] = A[i][j] * x[j] + tmp[i];
      y[i] = B[i][j] * x[j] + y[i];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
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
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
             POLYBENCH_ARRAY(x));

  /* Run kernel. */
  kernel_gesummv(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
                 POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  FINALIZE;

  return 0;
}
