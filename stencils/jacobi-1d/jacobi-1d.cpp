/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"

/* Array initialization. */
static void init_array(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, A, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, B, N, n)) {
  for (INT_TYPE i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(A, i) = ((DATA_TYPE)i + 2) / n;
    ARRAY_1D_ACCESS(B, i) = ((DATA_TYPE)i + 3) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, A, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (INT_TYPE i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(A, i));
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_jacobi_1d(INT_TYPE tsteps, INT_TYPE n,
                             ARRAY_1D_FUNC_PARAM(DATA_TYPE, A, N, n),
                             ARRAY_1D_FUNC_PARAM(DATA_TYPE, B, N, n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_time = Kokkos::RangePolicy<Kokkos::OpenMP>(0, tsteps);
#else // GPU
  const auto policy_time =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, tsteps);
#endif
  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0">(
      policy_time, KOKKOS_LAMBDA(const INT_TYPE t) {
        for (INT_TYPE i = 1; i < KOKKOS_LOOP_BOUND(n) - 1; i++)
          B(i) = SCALAR_VAL(0.33333) * (A(i - 1) + A(i) + A(i + 1));
        for (INT_TYPE i = 1; i < KOKKOS_LOOP_BOUND(n) - 1; i++)
          A(i) = SCALAR_VAL(0.33333) * (B(i - 1) + B(i) + B(i + 1));
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(1, n - 1);
  for (INT_TYPE t = 0; t < tsteps; t++) {
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const INT_TYPE i) {
          B(i) = SCALAR_VAL(0.33333) * (A(i - 1) + A(i) + A(i + 1));
        });
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const INT_TYPE i) {
          A(i) = SCALAR_VAL(0.33333) * (B(i - 1) + B(i) + B(i + 1));
        });
  }
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_1D(A, n);
  polybench_GPU_array_1D(B, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(B);

  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(1, n - 1);
  for (INT_TYPE t = 0; t < tsteps; t++) {
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const INT_TYPE i) {
          d_B(i) = SCALAR_VAL(0.33333) * (d_A(i - 1) + d_A(i) + d_A(i + 1));
        });
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const INT_TYPE i) {
          d_A(i) = SCALAR_VAL(0.33333) * (d_B(i - 1) + d_B(i) + d_B(i + 1));
        });
  }

  polybench_GPU_array_copy_to_host(A);
  polybench_GPU_array_copy_to_host(B);

  polybench_stop_instruments;

  polybench_GPU_array_sync_1D(A, n);
  polybench_GPU_array_sync_1D(B, n);

#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE t = 0; t < tsteps; t++) {
    for (INT_TYPE i = 1; i < n - 1; i++)
      B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
    for (INT_TYPE i = 1; i < n - 1; i++)
      A[i] = 0.33333 * (B[i - 1] + B[i] + B[i + 1]);
  }
#pragma endscop
  polybench_stop_instruments;
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  INT_TYPE n = N;
  INT_TYPE tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Run kernel. */
  kernel_jacobi_1d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  FINALIZE;

  return 0;
}
