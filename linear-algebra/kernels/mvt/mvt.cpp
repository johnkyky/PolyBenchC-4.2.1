/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* mvt.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "mvt.h"

/* Array initialization. */
static void init_array(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, x1, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x2, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y_1, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y_2, N, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
  for (INT_TYPE i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(x1, i) = (DATA_TYPE)(i % n) / n;
    ARRAY_1D_ACCESS(x2, i) = (DATA_TYPE)((i + 1) % n) / n;
    ARRAY_1D_ACCESS(y_1, i) = (DATA_TYPE)((i + 3) % n) / n;
    ARRAY_1D_ACCESS(y_2, i) = (DATA_TYPE)((i + 4) % n) / n;
    for (INT_TYPE j = 0; j < n; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)(i * j % n) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, x1, N, n),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, x2, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (INT_TYPE i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
            ARRAY_1D_ACCESS(x1, i));
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (INT_TYPE i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
            ARRAY_1D_ACCESS(x2, i));
  }
  POLYBENCH_DUMP_END("x2");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_mvt(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, x1, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x2, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y_1, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y_2, N, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {n, n});
#else // GPU
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {n, n});
#endif
  Kokkos::parallel_for<Kokkos::usePolyOpt,
                       "p0.l0 == 0, p0.l1 == 0, p0.u0 == p0.u1, p1.l0 == 0, "
                       "p1.l1 == 0, p1.u0 == p1.u1, p0.u0 == p1.u0">(
      "kernel", policy_2D,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        x1(i) = x1(i) + A(i, j) * y_1(j);
      },
      policy_2D,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        x2(i) = x2(i) + A(j, i) * y_2(j);
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;

  auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < n; j++) {
          x1(i) = x1(i) + A(i, j) * y_1(j);
        }
      });

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < n; j++) {
          x2(i) = x2(i) + A(j, i) * y_2(j);
        }
      });
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_1D(x1, n);
  polybench_GPU_array_1D(x2, n);
  polybench_GPU_array_1D(y_1, n);
  polybench_GPU_array_1D(y_2, n);
  polybench_GPU_array_2D(A, n, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(x1);
  polybench_GPU_array_copy_to_device(x2);
  polybench_GPU_array_copy_to_device(y_1);
  polybench_GPU_array_copy_to_device(y_2);
  polybench_GPU_array_copy_to_device(A);

  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE local_sum = SCALAR_VAL(0.0);
        for (INT_TYPE j = 0; j < n; j++) {
          local_sum += d_A(i, j) * d_y_1(j);
        }
        d_x1(i) += local_sum;
      });

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE local_sum = SCALAR_VAL(0.0);
        for (INT_TYPE j = 0; j < n; j++) {
          local_sum += d_A(j, i) * d_y_2(j);
        }
        d_x2(i) += local_sum;
      });

  polybench_GPU_array_copy_to_host(x1);
  polybench_GPU_array_copy_to_host(x2);

  polybench_stop_instruments;

  polybench_GPU_array_sync_1D(x1, n);
  polybench_GPU_array_sync_1D(x2, n);
#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
#pragma endscop
  polybench_stop_instruments;
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  INT_TYPE n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1),
             POLYBENCH_ARRAY(y_2), POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_mvt(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1),
             POLYBENCH_ARRAY(y_2), POLYBENCH_ARRAY(A));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(
      print_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);

  FINALIZE;

  return 0;
}
