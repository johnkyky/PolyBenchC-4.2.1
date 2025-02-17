/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* deriche.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "deriche.h"

/* Array initialization. */
static void init_array(int w, int h, DATA_TYPE *alpha,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, imgIn, W, H, w, h),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, imgOut, W, H, w, h)) {
  *alpha = 0.25; // parameter of the filter

  // input should be between 0 and 1 (grayscale image pixel)
  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
      ARRAY_2D_ACCESS(imgIn, i, j) =
          (DATA_TYPE)((313 * i + 991 * j) % 65536) / 65535.0f;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int w, int h,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, imgOut, W, H, w, h)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("imgOut");
  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(imgOut, i, j));
    }
  POLYBENCH_DUMP_END("imgOut");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
static void kernel_deriche(int w, int h, DATA_TYPE alpha,
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, imgIn, W, H, w, h),
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, imgOut, W, H, w, h),
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, y1, W, H, w, h),
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, y2, W, H, w, h)) {
  DATA_TYPE xm1, tm1, ym1, ym2;
  DATA_TYPE xp1, xp2;
  DATA_TYPE tp1, tp2;
  DATA_TYPE yp1, yp2;

  DATA_TYPE k;
  DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
  DATA_TYPE b1, b2, c1, c2;

#if defined(POLYBENCH_KOKKOS)
  k = (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) *
      (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) /
      (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * alpha * EXP_FUN(-alpha) -
       EXP_FUN(SCALAR_VAL(2.0) * alpha));
  a1 = a5 = k;
  a2 = a6 = k * EXP_FUN(-alpha) * (alpha - SCALAR_VAL(1.0));
  a3 = a7 = k * EXP_FUN(-alpha) * (alpha + SCALAR_VAL(1.0));
  a4 = a8 = -k * EXP_FUN(SCALAR_VAL(-2.0) * alpha);
  b1 = POW_FUN(SCALAR_VAL(2.0), -alpha);
  b2 = -EXP_FUN(SCALAR_VAL(-2.0) * alpha);
  c1 = c2 = 1;

  const auto policy_1D_1 = Kokkos::RangePolicy<>(0, w);
  const auto policy_1D_2 = Kokkos::RangePolicy<>(0, h);
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {w, h});

  Kokkos::parallel_for<usePolyOpt>(
      policy_1D_1, KOKKOS_LAMBDA(const int i) {
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);
        DATA_TYPE xm1 = SCALAR_VAL(0.0);
        for (int j = 0; j < _PB_H; j++) {
          y1(i, j) = a1 * imgIn(i, j) + a2 * xm1 + b1 * ym1 + b2 * ym2;
          xm1 = imgIn(i, j);
          ym2 = ym1;
          ym1 = y1(i, j);
        }
      });

  Kokkos::parallel_for<usePolyOpt>(
      policy_1D_1, KOKKOS_LAMBDA(const int i) {
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);
        DATA_TYPE xp1 = SCALAR_VAL(0.0);
        DATA_TYPE xp2 = SCALAR_VAL(0.0);
        for (int j = _PB_H - 1; j >= 0; j--) {
          y2(i, j) = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
          xp2 = xp1;
          xp1 = imgIn(i, j);
          yp2 = yp1;
          yp1 = y2(i, j);
        }
      });

  Kokkos::parallel_for<usePolyOpt>(
      policy_2D_1, KOKKOS_LAMBDA(const int i, const int j) {
        imgOut(i, j) = c1 * (y1(i, j) + y2(i, j));
      });

  Kokkos::parallel_for<usePolyOpt>(
      policy_1D_2, KOKKOS_LAMBDA(const int j) {
        DATA_TYPE tm1 = SCALAR_VAL(0.0);
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);
        for (int i = 0; i < _PB_W; i++) {
          y1(i, j) = a5 * imgOut(i, j) + a6 * tm1 + b1 * ym1 + b2 * ym2;
          tm1 = imgOut(i, j);
          ym2 = ym1;
          ym1 = y1(i, j);
        }
      });

  Kokkos::parallel_for<usePolyOpt>(
      policy_1D_2, KOKKOS_LAMBDA(const int j) {
        DATA_TYPE tp1 = SCALAR_VAL(0.0);
        DATA_TYPE tp2 = SCALAR_VAL(0.0);
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);
        for (int i = _PB_W - 1; i >= 0; i--) {
          y2(i, j) = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
          tp2 = tp1;
          tp1 = imgOut(i, j);
          yp2 = yp1;
          yp1 = y2(i, j);
        }
      });

  Kokkos::parallel_for<usePolyOpt>(
      policy_2D_1, KOKKOS_LAMBDA(const int i, const int j) {
        imgOut(i, j) = c2 * (y1(i, j) + y2(i, j));
      });
#else
#pragma scop
  k = (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) *
      (SCALAR_VAL(1.0) - EXP_FUN(-alpha)) /
      (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * alpha * EXP_FUN(-alpha) -
       EXP_FUN(SCALAR_VAL(2.0) * alpha));
  a1 = a5 = k;
  a2 = a6 = k * EXP_FUN(-alpha) * (alpha - SCALAR_VAL(1.0));
  a3 = a7 = k * EXP_FUN(-alpha) * (alpha + SCALAR_VAL(1.0));
  a4 = a8 = -k * EXP_FUN(SCALAR_VAL(-2.0) * alpha);
  b1 = POW_FUN(SCALAR_VAL(2.0), -alpha);
  b2 = -EXP_FUN(SCALAR_VAL(-2.0) * alpha);
  c1 = c2 = 1;

  for (int i = 0; i < _PB_W; i++) {
    ym1 = SCALAR_VAL(0.0);
    ym2 = SCALAR_VAL(0.0);
    xm1 = SCALAR_VAL(0.0);
    for (int j = 0; j < _PB_H; j++) {
      y1[i][j] = a1 * imgIn[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
      xm1 = imgIn[i][j];
      ym2 = ym1;
      ym1 = y1[i][j];
    }
  }

  for (int i = 0; i < _PB_W; i++) {
    yp1 = SCALAR_VAL(0.0);
    yp2 = SCALAR_VAL(0.0);
    xp1 = SCALAR_VAL(0.0);
    xp2 = SCALAR_VAL(0.0);
    for (int j = _PB_H - 1; j >= 0; j--) {
      y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
      xp2 = xp1;
      xp1 = imgIn[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }

  for (int i = 0; i < _PB_W; i++)
    for (int j = 0; j < _PB_H; j++) {
      imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
    }

  for (int j = 0; j < _PB_H; j++) {
    tm1 = SCALAR_VAL(0.0);
    ym1 = SCALAR_VAL(0.0);
    ym2 = SCALAR_VAL(0.0);
    for (int i = 0; i < _PB_W; i++) {
      y1[i][j] = a5 * imgOut[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
      tm1 = imgOut[i][j];
      ym2 = ym1;
      ym1 = y1[i][j];
    }
  }

  for (int j = 0; j < _PB_H; j++) {
    tp1 = SCALAR_VAL(0.0);
    tp2 = SCALAR_VAL(0.0);
    yp1 = SCALAR_VAL(0.0);
    yp2 = SCALAR_VAL(0.0);
    for (int i = _PB_W - 1; i >= 0; i--) {
      y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
      tp2 = tp1;
      tp1 = imgOut[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }

  for (int i = 0; i < _PB_W; i++)
    for (int j = 0; j < _PB_H; j++)
      imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
#pragma endscop
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  int w = W;
  int h = H;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);

  /* Initialize array(s). */
  init_array(w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_deriche(w, h, alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut),
                 POLYBENCH_ARRAY(y1), POLYBENCH_ARRAY(y2));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(w, h, POLYBENCH_ARRAY(imgOut)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(imgIn);
  POLYBENCH_FREE_ARRAY(imgOut);
  POLYBENCH_FREE_ARRAY(y1);
  POLYBENCH_FREE_ARRAY(y2);

  FINALIZE;

  return 0;
}
