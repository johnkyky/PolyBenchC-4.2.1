/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* durbin.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "durbin.h"

/* Array initialization. */
static void init_array(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, r, N, n)) {
  for (INT_TYPE i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(r, i) = (n + 1 - i);
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
static void kernel_durbin(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, r, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n)) {
#if defined(POLYBENCH_USE_POLLY)
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);
  DATA_TYPE alpha;
  DATA_TYPE beta;

  DATA_TYPE *alpha_ptr = &alpha;
  DATA_TYPE *beta_ptr = &beta;

  y(0) = -r(0);
  *beta_ptr = SCALAR_VAL(1.0);
  *alpha_ptr = -r[0];

  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(1, n);
#else // GPU
  const auto policy_1D =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(1, n);
#endif
  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 1">(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE k) {
        *beta_ptr = (1 - *alpha_ptr * *alpha_ptr) * *beta_ptr;
        DATA_TYPE sum = SCALAR_VAL(0.0);
        for (INT_TYPE i = 0; i < KOKKOS_LOOP_BOUND(k); i++) {
          sum += r(k - i - 1) * y(i);
        }
        *alpha_ptr = -(r[k] + sum) / *beta_ptr;

        for (INT_TYPE i = 0; i < KOKKOS_LOOP_BOUND(k); i++) {
          z[i] = y[i] + *alpha_ptr * y[k - i - 1];
        }
        for (INT_TYPE i = 0; i < KOKKOS_LOOP_BOUND(k); i++) {
          y[i] = z[i];
        }
        y[k] = *alpha_ptr;
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);
  DATA_TYPE alpha;
  DATA_TYPE beta;

  y(0) = -r(0);
  beta = SCALAR_VAL(1.0);
  alpha = -r(0);

#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  for (INT_TYPE k = 1; k < n; k++) {
    beta = (1 - alpha * alpha) * beta;

    DATA_TYPE sum = SCALAR_VAL(0.0);
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::OpenMP>(0, k),
        KOKKOS_LAMBDA(INT_TYPE i, DATA_TYPE & partial_sum) {
          partial_sum += r(k - i - 1) * y(i);
        },
        sum);

    alpha = -(r(k) + sum) / beta;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(0, k),
        KOKKOS_LAMBDA(INT_TYPE i) { z(i) = y(i) + alpha * y(k - i - 1); });

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(0, k),
        KOKKOS_LAMBDA(INT_TYPE i) { y(i) = z(i); });

    y(k) = alpha;
  }
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_1D(r, n);
  polybench_GPU_array_1D(y, n);
  polybench_GPU_array_1D(z, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(r);
  polybench_GPU_array_copy_to_device(y);

  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, 1);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE thread_id) {
        DATA_TYPE alpha = -d_r(0);
        DATA_TYPE beta = SCALAR_VAL(1.0);
        d_y(0) = -d_r(0);

        for (INT_TYPE k = 1; k < n; k++) {
          beta = (SCALAR_VAL(1.0) - alpha * alpha) * beta;

          DATA_TYPE sum = SCALAR_VAL(0.0);
          for (INT_TYPE i = 0; i < k; i++) {
            sum += d_r(k - i - 1) * d_y(i);
          }

          alpha = -(d_r(k) + sum) / beta;

          for (INT_TYPE i = 0; i < k; i++) {
            d_z(i) = d_y(i) + alpha * d_y(k - i - 1);
          }

          for (INT_TYPE i = 0; i < k; i++) {
            d_y(i) = d_z(i);
          }

          d_y(k) = alpha;
        }
      });

  polybench_GPU_array_copy_to_host(y);
  polybench_GPU_array_copy_to_host(z);

  polybench_stop_instruments;

  polybench_GPU_array_sync_1D(y, n);
  polybench_GPU_array_sync_1D(z, n);

#endif
#else
  DATA_TYPE z[N];
  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE sum;

  y[0] = -r[0];
  beta = SCALAR_VAL(1.0);
  alpha = -r[0];

  polybench_start_instruments;
#pragma scop
  for (INT_TYPE k = 1; k < n; k++) {
    beta = (1 - alpha * alpha) * beta;
    sum = SCALAR_VAL(0.0);
    for (INT_TYPE i = 0; i < k; i++) {
      sum += r[k - i - 1] * y[i];
    }
    alpha = -(r[k] + sum) / beta;

    for (INT_TYPE i = 0; i < k; i++) {
      z[i] = y[i] + alpha * y[k - i - 1];
    }
    for (INT_TYPE i = 0; i < k; i++) {
      y[i] = z[i];
    }
    y[k] = alpha;
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
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(r));

  /* Run kernel. */
  kernel_durbin(n, POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(y));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(y);

  FINALIZE;

  return 0;
}
