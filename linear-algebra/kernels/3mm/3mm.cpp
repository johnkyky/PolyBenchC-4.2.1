/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 3mm.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "3mm.h"

/* Array initialization. */
static void init_array(int ni, int nj, int nk, int nl, int nm,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NJ, NM, nj, nm),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, D, NM, NL, nm, nl)) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nk; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i * j + 1) % ni) / (5 * ni);
  for (int i = 0; i < nk; i++)
    for (int j = 0; j < nj; j++)
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)((i * (j + 1) + 2) % nj) / (5 * nj);
  for (int i = 0; i < nj; i++)
    for (int j = 0; j < nm; j++)
      ARRAY_2D_ACCESS(C, i, j) = (DATA_TYPE)(i * (j + 3) % nl) / (5 * nl);
  for (int i = 0; i < nm; i++)
    for (int j = 0; j < nl; j++)
      ARRAY_2D_ACCESS(D, i, j) = (DATA_TYPE)((i * (j + 2) + 2) % nk) / (5 * nk);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nl,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, G, NI, NL, ni, nl)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("G");
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(G, i, j));
    }
  POLYBENCH_DUMP_END("G");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, E, NI, NJ, ni, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, F, NJ, NL, nj, nl),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NJ, NM, nj, nm),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, D, NM, NL, nm, nl),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, G, NI, NL, ni, nl)) {
#pragma scop
  /* E := A*B */
  for (int i = 0; i < _PB_NI; i++)
    for (int j = 0; j < _PB_NJ; j++) {
      ARRAY_2D_ACCESS(E, i, j) = SCALAR_VAL(0.0);
      for (int k = 0; k < _PB_NK; ++k)
        ARRAY_2D_ACCESS(E, i, j) +=
            ARRAY_2D_ACCESS(A, i, k) * ARRAY_2D_ACCESS(B, k, j);
    }
  /* F := C*D */
  for (int i = 0; i < _PB_NJ; i++)
    for (int j = 0; j < _PB_NL; j++) {
      ARRAY_2D_ACCESS(F, i, j) = SCALAR_VAL(0.0);
      for (int k = 0; k < _PB_NM; ++k)
        ARRAY_2D_ACCESS(F, i, j) +=
            ARRAY_2D_ACCESS(C, i, k) * ARRAY_2D_ACCESS(D, k, j);
    }
  /* G := E*F */
  for (int i = 0; i < _PB_NI; i++)
    for (int j = 0; j < _PB_NL; j++) {
      ARRAY_2D_ACCESS(G, i, j) = SCALAR_VAL(0.0);
      for (int k = 0; k < _PB_NJ; ++k)
        ARRAY_2D_ACCESS(G, i, j) +=
            ARRAY_2D_ACCESS(E, i, k) * ARRAY_2D_ACCESS(F, k, j);
    }
#pragma endscop
}

int main(int argc, char **argv) {

  INITIALIZE;

  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);

  /* Initialize array(s). */
  init_array(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
             POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_3mm(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(E), POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(C),
             POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(G));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);

  FINALIZE;

  return 0;
}
