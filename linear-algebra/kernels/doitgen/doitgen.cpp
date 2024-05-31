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
void kernel_doitgen(int nr, int nq, int np,
                    ARRAY_3D_FUNC_PARAM(DATA_TYPE, A, NR, NQ, NP, nr, nq, np),
                    ARRAY_2D_FUNC_PARAM(DATA_TYPE, C4, NP, NP, np, np),
                    ARRAY_1D_FUNC_PARAM(DATA_TYPE, sum, NP, np)) {
#pragma scop
  for (int r = 0; r < _PB_NR; r++)
    for (int q = 0; q < _PB_NQ; q++) {
      for (int p = 0; p < _PB_NP; p++) {
        ARRAY_1D_ACCESS(sum, p) = SCALAR_VAL(0.0);
        for (int s = 0; s < _PB_NP; s++)
          ARRAY_1D_ACCESS(sum, p) +=
              ARRAY_3D_ACCESS(A, r, q, s) * ARRAY_2D_ACCESS(C4, s, p);
      }
      for (int p = 0; p < _PB_NP; p++)
        ARRAY_3D_ACCESS(A, r, q, p) = ARRAY_1D_ACCESS(sum, p);
    }
#pragma endscop
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

  /* Initialize array(s). */
  init_array(nr, nq, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C4));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_doitgen(nr, nq, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C4),
                 POLYBENCH_ARRAY(sum));

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

  FINALIZE;

  return 0;
}
