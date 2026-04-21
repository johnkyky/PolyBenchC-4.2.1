/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* doitgen.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "doitgen.h"

/* Array initialization. */
static void init_array(INT_TYPE nr, INT_TYPE nq, INT_TYPE np,
                       ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, NR, NQ, NP, nr, nq,
                                           np),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C4, NP, NP, np, np)) {
  for (INT_TYPE i = 0; i < nr; i++)
    for (INT_TYPE j = 0; j < nq; j++)
      for (INT_TYPE k = 0; k < np; k++)
        ARRAY_3D_ACCESS(A, i, j, k) = (DATA_TYPE)((i * j + k) % np) / np;
  for (INT_TYPE i = 0; i < np; i++)
    for (INT_TYPE j = 0; j < np; j++)
      ARRAY_2D_ACCESS(C4, i, j) = (DATA_TYPE)(i * j % np) / np;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE nr, INT_TYPE nq, INT_TYPE np,
                        ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, NR, NQ, NP, nr, nq,
                                            np)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (INT_TYPE i = 0; i < nr; i++)
    for (INT_TYPE j = 0; j < nq; j++)
      for (INT_TYPE k = 0; k < np; k++) {
        if ((i * nq * np + j * np + k) % 20 == 0)
          fprintf(POLYBENCH_DUMP_TARGET, "\n");
        fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
                ARRAY_3D_ACCESS(A, i, j, k));
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgen(INT_TYPE nr, INT_TYPE nq, INT_TYPE np,
                    ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, NR, NQ, NP, nr, nq, np),
                    ARRAY_2D_FUNC_PARAM(DATA_TYPE, C4, NP, NP, np, np),
                    ARRAY_1D_FUNC_PARAM(DATA_TYPE, sum, NP, np)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {nr, nq});
#else // GPU
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {nr, nq});
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt, "p0.l0 == 0, p0.l1 == 0">(
      policy_2D, KOKKOS_LAMBDA(const INT_TYPE r, const INT_TYPE q) {
        for (INT_TYPE p = 0; p < KOKKOS_LOOP_BOUND(np); p++) {
          sum(p) = 0.0;
          for (INT_TYPE s = 0; s < KOKKOS_LOOP_BOUND(np); s++)
            sum(p) += A(r, q, s) * C4(s, p);
        }
        for (INT_TYPE p = 0; p < KOKKOS_LOOP_BOUND(np); p++)
          A(r, q, p) = sum(p);
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy = Kokkos::RangePolicy<Kokkos::OpenMP>(0, np);

  for (INT_TYPE r = 0; r < nr; r++) {
    for (INT_TYPE q = 0; q < nq; q++) {
      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(const INT_TYPE p) {
            sum(p) = SCALAR_VAL(0.0);
            for (INT_TYPE s = 0; s < np; s++)
              sum(p) += A(r, q, s) * C4(s, p);
          });

      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(const INT_TYPE p) { A(r, q, p) = sum(p); });
    }
  }
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_3D(A, nr, nq, np);
  polybench_GPU_array_2D(C4, np, np);
  polybench_GPU_array_1D(sum, np);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(C4);
  polybench_GPU_array_copy_to_device(sum);

  auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, np);
  for (INT_TYPE r = 0; r < nr; r++) {
    for (INT_TYPE q = 0; q < nq; q++) {
      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(const INT_TYPE p) {
            DATA_TYPE local_sum = SCALAR_VAL(0.0);
            for (INT_TYPE s = 0; s < np; s++) {
              local_sum += d_A(r, q, s) * d_C4(s, p);
            }
            d_sum(p) = local_sum;
          });

      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(const INT_TYPE p) { d_A(r, q, p) = d_sum(p); });
    }
  }

  polybench_GPU_array_copy_to_host(sum);
  polybench_GPU_array_copy_to_host(A);

  polybench_stop_instruments;

  polybench_GPU_array_sync_1D(sum, np);
  polybench_GPU_array_sync_3D(A, nr, nq, np);
#endif
#else
  polybench_start_instruments;
#pragma scop
  for (INT_TYPE r = 0; r < nr; r++) {
    for (INT_TYPE q = 0; q < nq; q++) {
      for (INT_TYPE p = 0; p < np; p++) {
        sum[p] = SCALAR_VAL(0.0);
        for (INT_TYPE s = 0; s < np; s++)
          sum[p] += A[r][q][s] * C4[s][p];
      }
      for (INT_TYPE p = 0; p < np; p++)
        A[r][q][p] = sum[p];
    }
  }
#pragma endscop
  polybench_stop_instruments;
#endif
}

int main(int argc, char **argv) {

  INITIALIZE;

  /* Retrieve problem size. */
  INT_TYPE nr = NR;
  INT_TYPE nq = NQ;
  INT_TYPE np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, NR, NQ, NP, nr, nq, np);
  POLYBENCH_1D_ARRAY_DECL(sum, DATA_TYPE, NP, np);
  POLYBENCH_2D_ARRAY_DECL(C4, DATA_TYPE, NP, NP, np, np);

  /* Initialize array(s). */
  init_array(nr, nq, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C4));

  /* Run kernel. */
  kernel_doitgen(nr, nq, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C4),
                 POLYBENCH_ARRAY(sum));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);

  FINALIZE;

  return 0;
}
