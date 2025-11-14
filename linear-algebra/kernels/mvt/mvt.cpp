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
static void init_array(int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, x1, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x2, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y_1, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y_2, N, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
  for (int i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(x1, i) = (DATA_TYPE)(i % n) / n;
    ARRAY_1D_ACCESS(x2, i) = (DATA_TYPE)((i + 1) % n) / n;
    ARRAY_1D_ACCESS(y_1, i) = (DATA_TYPE)((i + 3) % n) / n;
    ARRAY_1D_ACCESS(y_2, i) = (DATA_TYPE)((i + 4) % n) / n;
    for (int j = 0; j < n; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)(i * j % n) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, x1, N, n),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, x2, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
            ARRAY_1D_ACCESS(x1, i));
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (int i = 0; i < n; i++) {
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
static void kernel_mvt(int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, x1, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x2, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y_1, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y_2, N, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {n, n});

  Kokkos::parallel_for<usePolyOpt,
                       "p0.l0 == 0, p0.l1 == 0, p0.u0 == p0.u1, p0.l0 == "
                       "p1.l0, p0.l1 == p1.l1, p1.u0 == p1.u1, p0.u0 == p1.u0">(
      "kernel", policy_2D,
      KOKKOS_LAMBDA(const size_t i, const size_t j) {
        x1(i) = x1(i) + A(i, j) * y_1(j);
      },
      policy_2D,
      KOKKOS_LAMBDA(const size_t i, const size_t j) {
        x2(i) = x2(i) + A(j, i) * y_2(j);
      });
#elif defined(POLYBENCH_KOKKOS)
  const auto policy_2D = Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>(
      {0, 0}, {n, n}, {32, 32});

  Kokkos::parallel_for<usePolyOpt>(
      policy_2D, KOKKOS_LAMBDA(const size_t i, const size_t j) {
        x1(i) = x1(i) + A(i, j) * y_1(j);
      });
  Kokkos::parallel_for<usePolyOpt>(
      policy_2D, KOKKOS_LAMBDA(const size_t i, const size_t j) {
        x2(i) = x2(i) + A(j, i) * y_2(j);
      });
#else
#pragma scop
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++)
      ARRAY_1D_ACCESS(x1, i) =
          ARRAY_1D_ACCESS(x1, i) +
          ARRAY_2D_ACCESS(A, i, j) * ARRAY_1D_ACCESS(y_1, j);
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++)
      ARRAY_1D_ACCESS(x2, i) =
          ARRAY_1D_ACCESS(x2, i) +
          ARRAY_2D_ACCESS(A, j, i) * ARRAY_1D_ACCESS(y_2, j);
#pragma endscop
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  int n = N;

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

  /* Stop and print timer. */
  polybench_stop_instruments;
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
