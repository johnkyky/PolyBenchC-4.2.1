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
static void init_array(INT_TYPE n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, b, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n)) {
  DATA_TYPE fn = (DATA_TYPE)n;

  for (INT_TYPE i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(x, i) = 0;
    ARRAY_1D_ACCESS(y, i) = 0;
    ARRAY_1D_ACCESS(b, i) = (i + 1) / fn / 2.0 + 4;
  }

  for (INT_TYPE i = 0; i < n; i++) {
    for (INT_TYPE j = 0; j <= i; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)(-j % n) / n + 1;
    for (INT_TYPE j = i + 1; j < n; j++) {
      ARRAY_2D_ACCESS(A, i, j) = 0;
    }
    ARRAY_2D_ACCESS(A, i, i) = 1;
  }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  std::vector<std::vector<DATA_TYPE>> B(n, std::vector<DATA_TYPE>(n));
  for (INT_TYPE r = 0; r < n; ++r)
    for (INT_TYPE s = 0; s < n; ++s)
      B[r][s] = 0;
  for (INT_TYPE t = 0; t < n; ++t)
    for (INT_TYPE r = 0; r < n; ++r)
      for (INT_TYPE s = 0; s < n; ++s)
        B[r][s] += ARRAY_2D_ACCESS(A, r, t) * ARRAY_2D_ACCESS(A, s, t);
  for (INT_TYPE r = 0; r < n; ++r)
    for (INT_TYPE s = 0; s < n; ++s)
      ARRAY_2D_ACCESS(A, r, s) = B[r][s];
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE n, ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (INT_TYPE i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(x, i));
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_ludcmp(INT_TYPE n,
                          ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, b, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, x, N, n),
                          ARRAY_1D_FUNC_PARAM(DATA_TYPE, y, N, n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
#else // GPU
  const auto policy_1D =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);
#endif

  Kokkos::parallel_for<
      Kokkos::usePolyOpt,
      "p0.l0 == 0, p0.u0 == n, p0.u0 > 10, p0. == p1., p0. == p2., n < 1000000">(
      "kernel", policy_1D,
      KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < i; j++) {
          DATA_TYPE w = A(i, j);
          for (INT_TYPE k = 0; k < j; k++) {
            w -= A(i, k) * A(k, j);
          }
          A(i, j) = w / A(j, j);
        }
        for (INT_TYPE j = i; j < KOKKOS_LOOP_BOUND(n); j++) {
          DATA_TYPE w = A(i, j);
          for (INT_TYPE k = 0; k < i; k++) {
            w -= A(i, k) * A(k, j);
          }
          A(i, j) = w;
        }
      },
      policy_1D,
      KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE w = b(i);
        for (INT_TYPE j = 0; j < i; j++)
          w -= A(i, j) * y(j);
        y(i) = w;
      },
      policy_1D,
      KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE w = y(n - 1 - i);
        for (INT_TYPE j = n - 1 - i + 1; j < KOKKOS_LOOP_BOUND(n); j++)
          w -= A(n - 1 - i, j) * x(j);
        x(n - 1 - i) = w / A(n - 1 - i, n - 1 - i);
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  auto policy_1D = Kokkos::RangePolicy<Kokkos::Serial>(0, n);

  for (INT_TYPE i = 0; i < n; i++) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::Serial>(0, i), KOKKOS_LAMBDA(INT_TYPE j) {
          DATA_TYPE w = A(i, j);
          for (INT_TYPE k = 0; k < j; k++) { // 1 empty iteration
            w -= A(i, k) * A(k, j);
          }
          A(i, j) = w / A(j, j);
        });

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(i, n), KOKKOS_LAMBDA(INT_TYPE j) {
          DATA_TYPE w = A(i, j);
          for (INT_TYPE k = 0; k < i; k++) { // 1 empty iteration
            w -= A(i, k) * A(k, j);
          }
          A(i, j) = w;
        });
  }

  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE w = b[i];
        for (INT_TYPE j = 0; j < i; j++)
          w -= A(i, j) * y(j);
        y(i) = w;
      });

  Kokkos::parallel_for(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE w = y(n - 1 - i);
        for (INT_TYPE j = n - 1 - i + 1; j < n; j++)
          w -= A(n - 1 - i, j) * x(j);
        x(n - 1 - i) = w / A(n - 1 - i, n - 1 - i);
      });
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(A, n, n);
  polybench_GPU_array_1D(b, n);
  polybench_GPU_array_1D(x, n);
  polybench_GPU_array_1D(y, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(b);
  polybench_GPU_array_copy_to_device(x);
  polybench_GPU_array_copy_to_device(y);

  // Un seul thread pour gouverner tout l'algorithme
  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, 1);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE thread_id) {
        for (INT_TYPE i = 0; i < n; i++) {
          for (INT_TYPE j = 0; j < i; j++) {
            DATA_TYPE w = d_A(i, j);
            for (INT_TYPE k = 0; k < j; k++) {
              w -= d_A(i, k) * d_A(k, j);
            }
            d_A(i, j) = w / d_A(j, j);
          }

          for (INT_TYPE j = i; j < n; j++) {
            DATA_TYPE w = d_A(i, j);
            for (INT_TYPE k = 0; k < i; k++) {
              w -= d_A(i, k) * d_A(k, j);
            }
            d_A(i, j) = w;
          }
        }

        for (INT_TYPE i = 0; i < n; i++) {
          DATA_TYPE w = d_b(i);
          for (INT_TYPE j = 0; j < i; j++) {
            w -= d_A(i, j) * d_y(j);
          }
          d_y(i) = w;
        }

        for (INT_TYPE i = 0; i < n; i++) {
          INT_TYPE row = n - 1 - i;
          DATA_TYPE w = d_y(row);
          for (INT_TYPE j = row + 1; j < n; j++) {
            w -= d_A(row, j) * d_x(j);
          }
          d_x(row) = w / d_A(row, row);
        }
      });

  polybench_GPU_array_copy_to_host(A);
  polybench_GPU_array_copy_to_host(x);
  polybench_GPU_array_copy_to_host(y);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(A, n, n);
  polybench_GPU_array_sync_1D(x, n);
  polybench_GPU_array_sync_1D(y, n);
#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < n; i++) {   // 0 empty iteration
    for (INT_TYPE j = 0; j < i; j++) { // 1 empty iteration
      DATA_TYPE w = A[i][j];
      for (INT_TYPE k = 0; k < j; k++) { // 1 empty iteration
        w -= A[i][k] * A[k][j];
      }
      A[i][j] = w / A[j][j];
    }
    for (INT_TYPE j = i; j < n; j++) { // 0 empty iteration
      DATA_TYPE w = A[i][j];
      for (INT_TYPE k = 0; k < i; k++) { // 1 empty iteration
        w -= A[i][k] * A[k][j];
      }
      A[i][j] = w;
    }
  }

  for (INT_TYPE i = 0; i < n; i++) { // 0 empty iteration
    DATA_TYPE w = b[i];
    for (INT_TYPE j = 0; j < i; j++) // 1 empty iteration
      w -= A[i][j] * y[j];
    y[i] = w;
  }

  for (SINT_TYPE i = n - 1; i >= 0; i--) { // 0 empty iteration
    DATA_TYPE w = y[i];
    for (INT_TYPE j = i + 1; j < n; j++) // 1 empty iteration
      w -= A[i][j] * x[j];
    x[i] = w / A[i][i];
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
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(x),
             POLYBENCH_ARRAY(y));

  /* Run kernel. */
  kernel_ludcmp(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(x),
                POLYBENCH_ARRAY(y));

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
