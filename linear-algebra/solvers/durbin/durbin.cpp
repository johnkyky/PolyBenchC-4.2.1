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
static void init_array(int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, r, N, n)) {
  for (int i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(r, i) = (n + 1 - i);
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(y, i));
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_durbin(size_t n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, r, N, n),
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

  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(1, n);
  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 1">(
      policy_1D, KOKKOS_LAMBDA(const size_t k) {
        *beta_ptr = (1 - *alpha_ptr * *alpha_ptr) * *beta_ptr;
        DATA_TYPE sum = SCALAR_VAL(0.0);
        for (size_t i = 0; i < KOKKOS_LOOP_BOUND(k); i++) {
          sum += r(k - i - 1) * y(i);
        }
        *alpha_ptr = -(r[k] + sum) / *beta_ptr;

        for (size_t i = 0; i < KOKKOS_LOOP_BOUND(k); i++) {
          z[i] = y[i] + *alpha_ptr * y[k - i - 1];
        }
        for (size_t i = 0; i < KOKKOS_LOOP_BOUND(k); i++) {
          y[i] = z[i];
        }
        y[k] = *alpha_ptr;
      });
#elif defined(POLYBENCH_KOKKOS)
  POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);
  DATA_TYPE alpha;
  DATA_TYPE beta;

  y(0) = -r(0);
  beta = SCALAR_VAL(1.0);
  alpha = -r[0];
  for (size_t k = 1; k < n; k++) {
    beta = (1 - alpha * alpha) * beta;

    DATA_TYPE sum = SCALAR_VAL(0.0);
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::OpenMP>(0, k),
        KOKKOS_LAMBDA(size_t i, DATA_TYPE &partial_sum) {
          partial_sum += r(k - i - 1) * y(i);
        },
        sum);

    alpha = -(r(k) + sum) / beta;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(0, k),
        KOKKOS_LAMBDA(size_t i) { z(i) = y(i) + alpha * y(k - i - 1); });

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(0, k),
        KOKKOS_LAMBDA(size_t i) { y(i) = z(i); });

    y(k) = alpha;
  }
#else
  DATA_TYPE z[N];
  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE sum;

  y[0] = -r[0];
  beta = SCALAR_VAL(1.0);
  alpha = -r[0];

#pragma scop
  for (size_t k = 1; k < n; k++) {
    beta = (1 - alpha * alpha) * beta;
    sum = SCALAR_VAL(0.0);
    for (size_t i = 0; i < k; i++) {
      sum += r[k - i - 1] * y[i];
    }
    alpha = -(r[k] + sum) / beta;

    for (size_t i = 0; i < k; i++) {
      z[i] = y[i] + alpha * y[k - i - 1];
    }
    for (size_t i = 0; i < k; i++) {
      y[i] = z[i];
    }
    y[k] = alpha;
  }
#pragma endscop
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(r));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_durbin(n, POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
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
