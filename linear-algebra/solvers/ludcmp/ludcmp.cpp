/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* ludcmp.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"

/* Array initialization. */
static void init_array(int n, ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, b, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n)) {
  DATA_TYPE fn = (DATA_TYPE)n;

  for (int i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(x, i) = 0;
    ARRAY_1D_ACCESS(y, i) = 0;
    ARRAY_1D_ACCESS(b, i) = (i + 1) / fn / 2.0 + 4;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)(-j % n) / n + 1;
    for (int j = i + 1; j < n; j++) {
      ARRAY_2D_ACCESS(A, i, j) = 0;
    }
    ARRAY_2D_ACCESS(A, i, i) = 1;
  }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  std::vector<std::vector<DATA_TYPE>> B(n, std::vector<DATA_TYPE>(n));
  for (int r = 0; r < n; ++r)
    for (int s = 0; s < n; ++s)
      B[r][s] = 0;
  for (int t = 0; t < n; ++t)
    for (int r = 0; r < n; ++r)
      for (int s = 0; s < n; ++s)
        B[r][s] += ARRAY_2D_ACCESS(A, r, t) * ARRAY_2D_ACCESS(A, s, t);
  for (int r = 0; r < n; ++r)
    for (int s = 0; s < n; ++s)
      ARRAY_2D_ACCESS(A, r, s) = B[r][s];
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (int i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(x, i));
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_ludcmp(size_t n,
                          ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, b, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);

  // "p0.l0 == 0, p0.u0 == n, p0.u0 > 10, p0. == p1., p0. == p2."
  Kokkos::parallel_for<
      Kokkos::usePolyOpt,
      "p0.l0 == 0, p0.u0 == n, p0.u0 > 10, p0. == p1., p0. == p2., n < 10000">(
      "kernel", policy_1D,
      KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < i; j++) {
          DATA_TYPE w = A(i, j);
          for (size_t k = 0; k < j; k++) {
            w -= A(i, k) * A(k, j);
          }
          A(i, j) = w / A(j, j);
        }
        for (size_t j = i; j < KOKKOS_LOOP_BOUND(n); j++) {
          DATA_TYPE w = A(i, j);
          for (size_t k = 0; k < i; k++) {
            w -= A(i, k) * A(k, j);
          }
          A(i, j) = w;
        }
      },
      policy_1D,
      KOKKOS_LAMBDA(const size_t i) {
        DATA_TYPE w = b[i];
        for (size_t j = 0; j < i; j++)
          w -= A(i, j) * y(j);
        y(i) = w;
      },
      policy_1D,
      KOKKOS_LAMBDA(const size_t i) {
        DATA_TYPE w = y(n - 1 - i);
        for (size_t j = n - 1 - i + 1; j < KOKKOS_LOOP_BOUND(n); j++)
          w -= A(n - 1 - i, j) * x(j);
        x(n - 1 - i) = w / A(n - 1 - i, n - 1 - i);
      });
#elif defined(POLYBENCH_KOKKOS)
  auto policy_1D = Kokkos::RangePolicy<Kokkos::Serial>(0, n);

  for (size_t i = 0; i < n; i++) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::Serial>(0, i), KOKKOS_LAMBDA(size_t j) {
          DATA_TYPE w = A(i, j);
          for (size_t k = 0; k < j; k++) { // 1 empty iteration
            w -= A(i, k) * A(k, j);
          }
          A(i, j) = w / A(j, j);
        });

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(i, n), KOKKOS_LAMBDA(size_t j) {
          DATA_TYPE w = A(i, j);
          for (size_t k = 0; k < i; k++) { // 1 empty iteration
            w -= A(i, k) * A(k, j);
          }
          A(i, j) = w;
        });
  }

  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const size_t i) {
        DATA_TYPE w = b[i];
        for (size_t j = 0; j < i; j++)
          w -= A(i, j) * y(j);
        y(i) = w;
      });

  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const size_t i) {
        DATA_TYPE w = y(n - 1 - i);
        for (size_t j = n - 1 - i + 1; j < n; j++)
          w -= A(n - 1 - i, j) * x(j);
        x(n - 1 - i) = w / A(n - 1 - i, n - 1 - i);
      });
#else
#pragma scop
  for (size_t i = 0; i < n; i++) {   // 0 empty iteration
    for (size_t j = 0; j < i; j++) { // 1 empty iteration
      DATA_TYPE w = A[i][j];
      for (size_t k = 0; k < j; k++) { // 1 empty iteration
        w -= A[i][k] * A[k][j];
      }
      A[i][j] = w / A[j][j];
    }
    for (size_t j = i; j < n; j++) { // 0 empty iteration
      DATA_TYPE w = A[i][j];
      for (size_t k = 0; k < i; k++) { // 1 empty iteration
        w -= A[i][k] * A[k][j];
      }
      A[i][j] = w;
    }
  }

  for (size_t i = 0; i < n; i++) { // 0 empty iteration
    DATA_TYPE w = b[i];
    for (size_t j = 0; j < i; j++) // 1 empty iteration
      w -= A[i][j] * y[j];
    y[i] = w;
  }

  for (ssize_t i = n - 1; i >= 0; i--) { // 0 empty iteration
    DATA_TYPE w = y[i];
    for (size_t j = i + 1; j < n; j++) // 1 empty iteration
      w -= A[i][j] * x[j];
    x[i] = w / A[i][i];
  }
#pragma endscop
#endif
}

int main(int argc, char **argv) {

  INITIALIZE;

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(x),
             POLYBENCH_ARRAY(y));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_ludcmp(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(x),
                POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  FINALIZE;

  return 0;
}
