/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* trisolv.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "trisolv.h"

/* Array initialization. */
static void init_array(INT_TYPE n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, L, N, N, n, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, b, N, n)) {
  for (INT_TYPE i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(x, i) = -999;
    ARRAY_1D_ACCESS(b, i) = i;
    for (INT_TYPE j = 0; j <= i; j++)
      ARRAY_2D_ACCESS(L, i, j) = (DATA_TYPE)(i + n - j + 1) * 2 / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (INT_TYPE i = 0; i < n; i++) {
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(x, i));
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_trisolv(INT_TYPE n,
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, L, N, N, n, n),
                           ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                           ARRAY_1D_FUNC_PARAM(DATA_TYPE, b, N, n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
#else // GPU
  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0">(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        x(i) = b(i);
        for (INT_TYPE j = 0; j < i; j++)
          x(i) -= L(i, j) * x(j);
        x(i) = x(i) / L(i, i);
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  for (INT_TYPE i = 0; i < n; i++) {
    x(i) = b(i);
    DATA_TYPE sum = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::OpenMP>(0, i),
        KOKKOS_LAMBDA(INT_TYPE j, DATA_TYPE & local_sum) {
          local_sum += L(i, j) * x(j);
        },
        sum);
    x(i) -= sum;
    x(i) = x(i) / L(i, i);
  }
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(L, n, n);
  polybench_GPU_array_1D(x, n);
  polybench_GPU_array_1D(b, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(L);
  polybench_GPU_array_copy_to_device(x);
  polybench_GPU_array_copy_to_device(b);

  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, 1);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE thread_id) {
        for (INT_TYPE i = 0; i < n; i++) {

          d_x(i) = d_b(i);
          DATA_TYPE sum = SCALAR_VAL(0.0);

          for (INT_TYPE j = 0; j < i; j++) {
            sum += d_L(i, j) * d_x(j);
          }

          d_x(i) -= sum;
          d_x(i) /= d_L(i, i);
        }
      });

  polybench_GPU_array_copy_to_host(x);

  polybench_stop_instruments;

  polybench_GPU_array_sync_1D(x, n);

#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < n; i++) {
    x[i] = b[i];
    for (INT_TYPE j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
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
  POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  /* Run kernel. */
  kernel_trisolv(n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(L);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(b);

  FINALIZE;

  return 0;
}
