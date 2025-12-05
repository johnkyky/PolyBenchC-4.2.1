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
static void init_array(int nr, int nq, int np,
                       ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, NR, NQ, NP, nr, nq,
                                           np),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C4, NP, NP, np, np)) {
  for (int i = 0; i < nr; i++)
    for (int j = 0; j < nq; j++)
      for (int k = 0; k < np; k++)
        ARRAY_3D_ACCESS(A, i, j, k) = (DATA_TYPE)((i * j + k) % np) / np;
  for (int i = 0; i < np; i++)
    for (int j = 0; j < np; j++)
      ARRAY_2D_ACCESS(C4, i, j) = (DATA_TYPE)(i * j % np) / np;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nr, int nq, int np,
                        ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, NR, NQ, NP, nr, nq,
                                            np)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (int i = 0; i < nr; i++)
    for (int j = 0; j < nq; j++)
      for (int k = 0; k < np; k++) {
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
void kernel_doitgen(size_t nr, size_t nq, size_t np,
                    ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, NR, NQ, NP, nr, nq, np),
                    ARRAY_3D_FUNC_PARAM(DATA_TYPE, tmp, NR, NQ, NP, nr, nq, np),
                    ARRAY_2D_FUNC_PARAM(DATA_TYPE, C4, NP, NP, np, np),
                    ARRAY_1D_FUNC_PARAM(DATA_TYPE, sum, NP, np)) {
#if defined(POLYBENCH_USE_POLLY)
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {nr, nq});

  Kokkos::parallel_for<usePolyOpt, "p0.l0 == 0, p0.l1 == 0">(
      policy_2D, KOKKOS_LAMBDA(const size_t r, const size_t q) {
        for (size_t p = 0; p < KOKKOS_LOOP_BOUND(np); p++) {
          sum(p) = 0.0;
          for (size_t s = 0; s < KOKKOS_LOOP_BOUND(np); s++)
            sum(p) += A(r, q, s) * C4(s, p);
        }
        for (size_t p = 0; p < KOKKOS_LOOP_BOUND(np); p++)
          A(r, q, p) = sum(p);
      });
#elif defined(POLYBENCH_KOKKOS)
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nr, nq});

  Kokkos::parallel_for(
      policy_2D, KOKKOS_LAMBDA(const size_t r, const size_t q) {
        for (size_t p = 0; p < np; p++) {
          tmp(r, q, p) = 0.0;
          for (size_t s = 0; s < np; s++)
            tmp(r, q, p) += A(r, q, s) * ARRAY_2D_ACCESS(C4, s, p);
        }
        for (size_t p = 0; p < np; p++)
          A(r, q, p) = tmp(r, q, p);
      });
#else
#pragma scop
  for (size_t r = 0; r < nr; r++)
    for (size_t q = 0; q < nq; q++) {
      for (size_t p = 0; p < np; p++) {
        sum[p] = SCALAR_VAL(0.0);
        for (size_t s = 0; s < np; s++)
          sum[p] += A[r][q][s] * C4[s][p];
      }
      for (size_t p = 0; p < np; p++)
        A[r][q][p] = sum[p];
    }
#pragma endscop
#endif
}

int main(int argc, char **argv) {

  INITIALIZE;

  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, NR, NQ, NP, nr, nq, np);
  POLYBENCH_1D_ARRAY_DECL(sum, DATA_TYPE, NP, np);
  POLYBENCH_2D_ARRAY_DECL(C4, DATA_TYPE, NP, NP, np, np);
  POLYBENCH_3D_ARRAY_DECL(tmp, DATA_TYPE, NR, NQ, NP, nr, nq, np);

  /* Initialize array(s). */
  init_array(nr, nq, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C4));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_doitgen(nr, nq, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(tmp),
                 POLYBENCH_ARRAY(C4), POLYBENCH_ARRAY(sum));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);
  POLYBENCH_FREE_ARRAY(tmp);

  FINALIZE;

  return 0;
}
