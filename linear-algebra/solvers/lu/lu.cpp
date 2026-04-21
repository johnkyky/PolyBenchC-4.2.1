/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* lu.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "lu.h"

/* Array initialization. */
static void init_array(INT_TYPE n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
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
static void print_array(INT_TYPE n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (INT_TYPE i = 0; i < n; i++)
    for (INT_TYPE j = 0; j < n; j++) {
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
static void kernel_lu(INT_TYPE n,
                      ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, N, n, n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_1D = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
#else // GPU
  const auto policy_1D =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0, p0.u0 == n, n > 10">(
      policy_1D, KOKKOS_LAMBDA(const INT_TYPE i) {
        for (INT_TYPE j = 0; j < i; j++) {
          for (INT_TYPE k = 0; k < j; k++) {
            A(i, j) -= A(i, k) * A(k, j);
          }
          A(i, j) /= A(j, j);
        }
        for (INT_TYPE j = i; j < KOKKOS_LOOP_BOUND(n); j++) {
          for (INT_TYPE k = 0; k < i; k++) {
            A(i, j) -= A(i, k) * A(k, j);
          }
        }
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  for (INT_TYPE i = 0; i < n; i++) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::Serial>(0, i), KOKKOS_LAMBDA(INT_TYPE j) {
          for (INT_TYPE k = 0; k < j; k++) {
            A(i, j) -= A(i, k) * A(k, j);
          }
          A(i, j) /= A(j, j);
        });

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::OpenMP>(i, n), KOKKOS_LAMBDA(INT_TYPE j) {
          for (INT_TYPE k = 0; k < i; k++) {
            A(i, j) -= A(i, k) * A(k, j);
          }
        });
  }
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(A, n, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);

  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, 1);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const INT_TYPE thread_id) {
        for (INT_TYPE i = 0; i < n; i++) {

          for (INT_TYPE j = 0; j < i; j++) {
            for (INT_TYPE k = 0; k < j; k++) {
              d_A(i, j) -= d_A(i, k) * d_A(k, j);
            }
            d_A(i, j) /= d_A(j, j);
          }

          for (INT_TYPE j = i; j < n; j++) {
            for (INT_TYPE k = 0; k < i; k++) {
              d_A(i, j) -= d_A(i, k) * d_A(k, j);
            }
          }
        }
      });

  polybench_GPU_array_copy_to_host(A);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(A, n, n);

#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < n; i++) {
    for (INT_TYPE j = 0; j < i; j++) {
      for (INT_TYPE k = 0; k < j; k++) {
        A[i][j] -= A[i][k] * A[k][j];
      }
      A[i][j] /= A[j][j];
    }
    for (INT_TYPE j = i; j < n; j++) {
      for (INT_TYPE k = 0; k < i; k++) {
        A[i][j] -= A[i][k] * A[k][j];
      }
    }
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

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A));

  /* Run kernel. */
  kernel_lu(n, POLYBENCH_ARRAY(A));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  FINALIZE;

  return 0;
}
