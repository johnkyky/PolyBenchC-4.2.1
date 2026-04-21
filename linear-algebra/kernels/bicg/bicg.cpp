/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* bicg.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "bicg.h"

/* Array initialization. */
static void init_array(INT_TYPE m, INT_TYPE n,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, M, n, m),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, r, N, n),
                       ARRAY_1D_FUNC_PARAM(DATA_TYPE, p, M, m)) {
  for (INT_TYPE i = 0; i < m; i++)
    ARRAY_1D_ACCESS(p, i) = (DATA_TYPE)(i % m) / m;
  for (INT_TYPE i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(r, i) = (DATA_TYPE)(i % n) / n;
    for (INT_TYPE j = 0; j < m; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)(i * (j + 1) % n) / n;
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE m, INT_TYPE n,
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, s, M, m),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, q, N, n)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (INT_TYPE i = 0; i < m; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(s, i));
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (INT_TYPE i = 0; i < n; i++) {
    if (i % 20 == 0)
      fprintf(POLYBENCH_DUMP_TARGET, "\n");
    fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ARRAY_1D_ACCESS(q, i));
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_bicg(INT_TYPE m, INT_TYPE n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, N, M, n, m),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, s, M, m),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, q, N, n),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, p, M, m),
                        ARRAY_1D_FUNC_PARAM(DATA_TYPE, r, N, n)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_1D_1 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m);
  const auto policy_1D_2 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, n);
#else // GPU
  const auto policy_1D_1 =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, m);
  const auto policy_1D_2 =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt,
                       "p0.l0 == 0, p1.l0 == 0, p0.u0 == m, m > 10">(
      "kernel", policy_1D_1,
      KOKKOS_LAMBDA(const INT_TYPE i) { s(i) = SCALAR_VAL(0.0); }, policy_1D_2,
      KOKKOS_LAMBDA(const INT_TYPE i) {
        q(i) = SCALAR_VAL(0.0);
        for (INT_TYPE j = 0; j < KOKKOS_LOOP_BOUND(m); j++) {
          s(j) = s(j) + r(i) * A(i, j);
          q(i) = q(i) + A(i, j) * p(j);
        }
      });

  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy_1D_1 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, m);
  const auto policy_1D_2 = Kokkos::RangePolicy<Kokkos::Serial>(0, n);

  Kokkos::parallel_for(
      policy_1D_1, KOKKOS_LAMBDA(const INT_TYPE j) {
        s(j) = 0;
        for (INT_TYPE i = 0; i < n; i++)
          s(j) = s(j) + r(i) * A(i, j);
      });
  Kokkos::parallel_for(
      policy_1D_2, KOKKOS_LAMBDA(const INT_TYPE i) {
        q(i) = SCALAR_VAL(0.0);
        for (INT_TYPE j = 0; j < m; j++) {
          q(i) = q(i) + A(i, j) * p(j);
        }
      });
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(A, n, m);
  polybench_GPU_array_1D(s, m);
  polybench_GPU_array_1D(q, n);
  polybench_GPU_array_1D(p, m);
  polybench_GPU_array_1D(r, n);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(s);
  polybench_GPU_array_copy_to_device(q);
  polybench_GPU_array_copy_to_device(p);
  polybench_GPU_array_copy_to_device(r);

  const auto policy_m =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, m);
  const auto policy_n =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, n);

  Kokkos::parallel_for(
      policy_m, KOKKOS_LAMBDA(const INT_TYPE j) {
        DATA_TYPE local_sum = SCALAR_VAL(0.0);
        for (INT_TYPE i = 0; i < n; i++) {
          local_sum += d_r(i) * d_A(i, j);
        }
        d_s(j) = local_sum;
      });

  Kokkos::parallel_for(
      policy_n, KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE local_sum = SCALAR_VAL(0.0);
        for (INT_TYPE j = 0; j < m; j++) {
          local_sum += d_A(i, j) * d_p(j);
        }
        d_q(i) = local_sum;
      });

  polybench_GPU_array_copy_to_host(s);
  polybench_GPU_array_copy_to_host(q);

  polybench_stop_instruments;

  polybench_GPU_array_sync_1D(s, m);
  polybench_GPU_array_sync_1D(q, n);
#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < m; i++)
    s[i] = 0;
  for (INT_TYPE i = 0; i < n; i++) {
    q[i] = SCALAR_VAL(0.0);
    for (INT_TYPE j = 0; j < m; j++) {
      s[j] = s[j] + r[i] * A[i][j];
      q[i] = q[i] + A[i][j] * p[j];
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
  INT_TYPE m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(p));

  /* Run kernel. */
  kernel_bicg(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q),
              POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(r));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(
      print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  FINALIZE;

  return 0;
}
