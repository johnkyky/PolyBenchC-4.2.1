/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* nussinov.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "nussinov.h"

/* RNA bases represented as chars, range is ,0,3) */
typedef char base;

#define match(b1, b2) (((b1) + (b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

/* Array initialization. */
static void init_array(int n, ARRAY_1D_FUNC_PARAM(base, seq, N, n),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, table, N, N, n, n)) {
  // base is AGCT/0..3
  for (int i = 0; i < n; i++) {
    ARRAY_1D_ACCESS(seq, i) = (base)((i + 1) % 4);
  }

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      ARRAY_2D_ACCESS(table, i, j) = 0;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, table, N, N, n, n)) {
  int t = 0;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("table");
  fprintf(POLYBENCH_DUMP_TARGET, "\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      /*if (t % 20 == 0)*/
      /*  fprintf(POLYBENCH_DUMP_TARGET, "\n");*/
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(table, i, j));
      t++;
    }
    fprintf(POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("table");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/*
  Original version by Dave Wonnacott at Haverford College
  <davew@cs.haverford.edu>, with help from Allison Lake, Ting Zhou, and Tian
  Jin, based on algorithm by Nussinov, described in Allison Lake's senior
  thesis.
*/
/// TODO: convert to kokkos kernel
static void kernel_nussinov(int n, ARRAY_1D_FUNC_PARAM(base, seq, N, n),
                            ARRAY_2D_FUNC_PARAM(DATA_TYPE, table, N, N, n, n)) {
#if defined(POLYBENCH_KOKKOS)
  const auto policy = Kokkos::RangePolicy<Kokkos::Serial>(0, n);

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(const int i) {
        for (int j = _PB_N - i; j < _PB_N; j++) {

          if (j - 1 >= 0) {
            table(i, j) = max_score(table(i, j), table(i, j - 1));
          }
          if (i > 0) {
            table(i, j) = max_score(table(i, j), table(i - 1, j));
          }

          if (j - 1 >= 0 && i > 0) {
            /* don't allow adjacent elements to bond */
            if (_PB_N - i < j) {
              table(i, j) =
                  max_score(table(i, j), table(i - 1, j - 1) +
                                             match(seq(_PB_N - i - 1), seq(j)));
            } else {
              table(i, j) = max_score(table(i, j), table(i - 1, j - 1));
            }
          }

          for (int k = _PB_N - i; k < j; k++) {
            table(i, j) =
                max_score(table(i, j), table(i, k) + table(_PB_N - k - 1, j));
          }
        }
      });

#else
#pragma scop
  for (int i = 0; i < _PB_N; i++) {
    for (int j = _PB_N - i; j < _PB_N; j++) {

      if (j - 1 >= 0) {
        table[i][j] = max_score(table[i][j], table[i][j - 1]);
      }
      if (i > 0) {
        table[i][j] = max_score(table[i][j], table[i - 1][j]);
      }

      if (j - 1 >= 0 && i > 0) {
        /* don't allow adjacent elements to bond */
        if (_PB_N - i < j) {
          table[i][j] =
              max_score(table[i][j], table[i - 1][j - 1] +
                                         match(seq[_PB_N - i - 1], seq[j]));
        } else {
          table[i][j] = max_score(table[i][j], table[i - 1][j - 1]);
        }
      }

      for (int k = _PB_N - i; k < j; k++) {
        table[i][j] =
            max_score(table[i][j], table[i][k] + table[_PB_N - k - 1][j]);
      }
    }
  }
#pragma endscop
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(seq, base, N, n);
  POLYBENCH_2D_ARRAY_DECL(table, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_nussinov(n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(table)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(seq);
  POLYBENCH_FREE_ARRAY(table);

  FINALIZE;

  return 0;
}
