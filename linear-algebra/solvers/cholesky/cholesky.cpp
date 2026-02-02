/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* cholesky.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "cholesky.h"

/* Array initialization. */
static void init_array(int n, ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)(-j % n) / n + 1;
    for (int j = i + 1; j < n; j++) {
      ARRAY_2D_ACCESS(A, i, j) = 0;
    }
    ARRAY_2D_ACCESS(A, i, i) = 1;
  }

  /* Make the matrix positive semi-definite. */
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
static void print_array(int n, ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (int i = 0; i < n; i++)
    for (int j = 0; j <= i; j++) {
      if ((i * n + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(A, i, j));
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_cholesky(size_t n,
                            ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {

#if defined(POLYBENCH_USE_POLLY)
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0, p0.u0 > 10">(
      policy, KOKKOS_LAMBDA(const size_t i) {
        for (size_t j = 0; j < i; j++) {
          for (size_t k = 0; k < j; k++) {
            A(i, j) -= A(i, k) * A(j, k);
          }
          A(i, j) /= A(j, j);
        }
        // i==j case
        for (size_t k = 0; k < i; k++) {
          A(i, i) -= A(i, k) * A(i, k);
        }
        A(i, i) = SQRT_FUN(A(i, i));
        // A(i, i) += A(i, i);
      });
#elif defined(POLYBENCH_KOKKOS)
  auto policy = Kokkos::RangePolicy<Kokkos::Serial>(0, n);
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const size_t i) {
        // j<i
        for (size_t j = 0; j < i; j++) {
          for (size_t k = 0; k < j; k++) {
            A(i, j) -= A(i, k) * A(j, k);
          }
          A(i, j) /= A(j, j);
        }
        // i==j case
        for (size_t k = 0; k < i; k++) {
          A(i, i) -= A(i, k) * A(i, k);
        }
        A(i, i) = SQRT_FUN(A(i, i));
      });
#else
#pragma scop
  for (size_t i = 0; i < n; i++) {
    // j<i
    for (size_t j = 0; j < i; j++) {
      for (size_t k = 0; k < j; k++) {
        A[i][j] -= A[i][k] * A[j][k];
      }
      A[i][j] /= A[j][j];
    }
    // i==j case
    for (size_t k = 0; k < i; k++) {
      A[i][i] -= A[i][k] * A[i][k];
    }
    A[i][i] = SQRT_FUN(A[i][i]);
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

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky(n, POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  FINALIZE;

  return 0;
}
