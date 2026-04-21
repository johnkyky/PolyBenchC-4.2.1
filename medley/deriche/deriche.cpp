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
static void init_array(INT_TYPE w, INT_TYPE h, DATA_TYPE *alpha,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, imgIn, W, H, w, h),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, imgOut, W, H, w, h)) {
  *alpha = 0.25; // parameter of the filter

  // input should be between 0 and 1 (grayscale image pixel)
  for (INT_TYPE i = 0; i < w; i++)
    for (INT_TYPE j = 0; j < h; j++)
      ARRAY_2D_ACCESS(imgIn, i, j) =
          (DATA_TYPE)((313 * i + 991 * j) % 65536) / 65535.0f;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE w, INT_TYPE h,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, imgOut, W, H, w, h)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("imgOut");
  for (INT_TYPE i = 0; i < w; i++)
    for (INT_TYPE j = 0; j < h; j++) {
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
static void kernel_deriche(INT_TYPE w, INT_TYPE h, DATA_TYPE alpha,
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, imgIn, W, H, w, h),
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, imgOut, W, H, w, h),
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, y1, W, H, w, h),
                           ARRAY_2D_FUNC_PARAM(DATA_TYPE, y2, W, H, w, h)) {
  DATA_TYPE k;
  DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
  DATA_TYPE b1, b2, c1, c2;

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

#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_1D_1 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, w);
  const auto policy_1D_2 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, h);
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {w, h},
                                                             {32, 32});
#else // GPU
  const auto policy_1D_1 =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, w);
  const auto policy_1D_2 =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, h);
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {w, h}, {32, 32});
#endif
  // "p0.l0==0, p1.l0==0, p2.l0==0, p2.l1==0, p3.l0==0, p4.l0==0, p5.l0==0, "
  //       "p5.l1 == 0, p0.u0 == p1.u0, p2.u0 == p5.u0, p2.u1 == p5.u1, p3.u0 ==
  //       " "p4.u0, p0.u0==p2.u0, p3.u0==p2.u1, p0.u0==w, p3.u0==h"

  // "p0.l0 == 0, p0.u0 > 10, p0.u0 == w, p0. == p1., "
  // "p2.l == 0, p2.u0 == w, p2.u1 == h, "
  // "p3.l0 == 0, p3.u0 > 10, p3.u0 == h, p3. == p4., "
  // "p2. == p5."
  Kokkos::parallel_for<Kokkos::usePolyOpt,
                       "p0.l0 == 0, p0.u0 > 10, p0.u0 == w, p0. == p1., "
                       "p2.l == 0, p2.u0 == w, p2.u1 == h, "
                       "p3.l0 == 0, p3.u0 > 10, p3.u0 == h, p3. == p4., "
                       "p2. == p5.">(
      "kernel", policy_1D_1,
      KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);
        DATA_TYPE xm1 = SCALAR_VAL(0.0);
        for (INT_TYPE j = 0; j < KOKKOS_LOOP_BOUND(h); j++) {
          y1(i, j) = a1 * imgIn(i, j) + a2 * xm1 + b1 * ym1 + b2 * ym2;
          xm1 = imgIn(i, j);
          ym2 = ym1;
          ym1 = y1(i, j);
        }
      },
      policy_1D_1,
      KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);
        DATA_TYPE xp1 = SCALAR_VAL(0.0);
        DATA_TYPE xp2 = SCALAR_VAL(0.0);
        for (SINT_TYPE j = KOKKOS_LOOP_BOUND(h) - 1; j >= 0; j--) {
          y2(i, j) = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
          xp2 = xp1;
          xp1 = imgIn(i, j);
          yp2 = yp1;
          yp1 = y2(i, j);
        }
      },
      policy_2D_1,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        imgOut(i, j) = c1 * (y1(i, j) + y2(i, j));
      },
      policy_1D_2,
      KOKKOS_LAMBDA(const INT_TYPE j) {
        DATA_TYPE tm1 = SCALAR_VAL(0.0);
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);
        for (INT_TYPE i = 0; i < KOKKOS_LOOP_BOUND(w); i++) {
          y1(i, j) = a5 * imgOut(i, j) + a6 * tm1 + b1 * ym1 + b2 * ym2;
          tm1 = imgOut(i, j);
          ym2 = ym1;
          ym1 = y1(i, j);
        }
      },
      policy_1D_2,
      KOKKOS_LAMBDA(const INT_TYPE j) {
        DATA_TYPE tp1 = SCALAR_VAL(0.0);
        DATA_TYPE tp2 = SCALAR_VAL(0.0);
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);
        for (SINT_TYPE i = KOKKOS_LOOP_BOUND(w) - 1; i >= 0; i--) {
          y2(i, j) = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
          tp2 = tp1;
          tp1 = imgOut(i, j);
          yp2 = yp1;
          yp1 = y2(i, j);
        }
      },
      policy_2D_1,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        imgOut(i, j) = c2 * (y1(i, j) + y2(i, j));
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy_1D_1 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, w);
  const auto policy_1D_2 = Kokkos::RangePolicy<Kokkos::OpenMP>(0, h);
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {w, h},
                                                             {32, 32});

  Kokkos::parallel_for(
      policy_1D_1, KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);
        DATA_TYPE xm1 = SCALAR_VAL(0.0);
        for (INT_TYPE j = 0; j < h; j++) {
          y1(i, j) = a1 * imgIn(i, j) + a2 * xm1 + b1 * ym1 + b2 * ym2;
          xm1 = imgIn(i, j);
          ym2 = ym1;
          ym1 = y1(i, j);
        }
      });

  Kokkos::parallel_for(
      policy_1D_1, KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);
        DATA_TYPE xp1 = SCALAR_VAL(0.0);
        DATA_TYPE xp2 = SCALAR_VAL(0.0);
        for (SINT_TYPE j = h - 1; j >= 0; j--) {
          y2(i, j) = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
          xp2 = xp1;
          xp1 = imgIn(i, j);
          yp2 = yp1;
          yp1 = y2(i, j);
        }
      });

  Kokkos::parallel_for(
      policy_2D_1, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        imgOut(i, j) = c1 * (y1(i, j) + y2(i, j));
      });

  Kokkos::parallel_for(
      policy_1D_2, KOKKOS_LAMBDA(const INT_TYPE j) {
        DATA_TYPE tm1 = SCALAR_VAL(0.0);
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);
        for (INT_TYPE i = 0; i < w; i++) {
          y1(i, j) = a5 * imgOut(i, j) + a6 * tm1 + b1 * ym1 + b2 * ym2;
          tm1 = imgOut(i, j);
          ym2 = ym1;
          ym1 = y1(i, j);
        }
      });

  Kokkos::parallel_for(
      policy_1D_2, KOKKOS_LAMBDA(const INT_TYPE j) {
        DATA_TYPE tp1 = SCALAR_VAL(0.0);
        DATA_TYPE tp2 = SCALAR_VAL(0.0);
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);
        for (SINT_TYPE i = w - 1; i >= 0; i--) {
          y2(i, j) = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
          tp2 = tp1;
          tp1 = imgOut(i, j);
          yp2 = yp1;
          yp1 = y2(i, j);
        }
      });

  Kokkos::parallel_for(
      policy_2D_1, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        imgOut(i, j) = c2 * (y1(i, j) + y2(i, j));
      });
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(imgIn, w, h);
  polybench_GPU_array_2D(imgOut, w, h);
  polybench_GPU_array_2D(y1, w, h);
  polybench_GPU_array_2D(y2, w, h);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(imgIn);

  const auto policy_1D_w =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, w);
  const auto policy_1D_h =
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>>(0, h);
  const auto policy_2D =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {w, h});

  Kokkos::parallel_for(
      policy_1D_w, KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);
        DATA_TYPE xm1 = SCALAR_VAL(0.0);

        for (INT_TYPE j = 0; j < h; j++) {
          d_y1(i, j) = a1 * d_imgIn(i, j) + a2 * xm1 + b1 * ym1 + b2 * ym2;
          xm1 = d_imgIn(i, j);
          ym2 = ym1;
          ym1 = d_y1(i, j);
        }
      });

  Kokkos::parallel_for(
      policy_1D_w, KOKKOS_LAMBDA(const INT_TYPE i) {
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);
        DATA_TYPE xp1 = SCALAR_VAL(0.0);
        DATA_TYPE xp2 = SCALAR_VAL(0.0);

        for (SINT_TYPE j = h - 1; j >= 0; j--) {
          d_y2(i, j) = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
          xp2 = xp1;
          xp1 = d_imgIn(i, j);
          yp2 = yp1;
          yp1 = d_y2(i, j);
        }
      });

  Kokkos::parallel_for(
      policy_2D, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        d_imgOut(i, j) = c1 * (d_y1(i, j) + d_y2(i, j));
      });

  Kokkos::parallel_for(
      policy_1D_h, KOKKOS_LAMBDA(const INT_TYPE j) {
        DATA_TYPE tm1 = SCALAR_VAL(0.0);
        DATA_TYPE ym1 = SCALAR_VAL(0.0);
        DATA_TYPE ym2 = SCALAR_VAL(0.0);

        for (INT_TYPE i = 0; i < w; i++) {
          d_y1(i, j) = a5 * d_imgOut(i, j) + a6 * tm1 + b1 * ym1 + b2 * ym2;
          tm1 = d_imgOut(i, j);
          ym2 = ym1;
          ym1 = d_y1(i, j);
        }
      });

  Kokkos::parallel_for(
      policy_1D_h, KOKKOS_LAMBDA(const INT_TYPE j) {
        DATA_TYPE tp1 = SCALAR_VAL(0.0);
        DATA_TYPE tp2 = SCALAR_VAL(0.0);
        DATA_TYPE yp1 = SCALAR_VAL(0.0);
        DATA_TYPE yp2 = SCALAR_VAL(0.0);

        for (SINT_TYPE i = w - 1; i >= 0; i--) {
          d_y2(i, j) = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
          tp2 = tp1;
          tp1 = d_imgOut(i, j);
          yp2 = yp1;
          yp1 = d_y2(i, j);
        }
      });

  Kokkos::parallel_for(
      policy_2D, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        d_imgOut(i, j) = c2 * (d_y1(i, j) + d_y2(i, j));
      });

  polybench_GPU_array_copy_to_host(ImgOut);
  polybench_GPU_array_copy_to_host(y1);
  polybench_GPU_array_copy_to_host(y2);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(imgOut, w, h);
  polybench_GPU_array_sync_2D(y1, w, h);
  polybench_GPU_array_sync_2D(y2, w, h);

#endif
#else
  DATA_TYPE xm1, tm1, ym1, ym2;
  DATA_TYPE xp1, xp2;
  DATA_TYPE tp1, tp2;
  DATA_TYPE yp1, yp2;

  polybench_start_instruments;
#pragma scop
  for (INT_TYPE i = 0; i < w; i++) {
    ym1 = SCALAR_VAL(0.0);
    ym2 = SCALAR_VAL(0.0);
    xm1 = SCALAR_VAL(0.0);
    for (INT_TYPE j = 0; j < h; j++) {
      y1[i][j] = a1 * imgIn[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
      xm1 = imgIn[i][j];
      ym2 = ym1;
      ym1 = y1[i][j];
    }
  }

  for (INT_TYPE i = 0; i < w; i++) {
    yp1 = SCALAR_VAL(0.0);
    yp2 = SCALAR_VAL(0.0);
    xp1 = SCALAR_VAL(0.0);
    xp2 = SCALAR_VAL(0.0);
    for (SINT_TYPE j = h - 1; j >= 0; j--) {
      y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
      xp2 = xp1;
      xp1 = imgIn[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }

  for (INT_TYPE i = 0; i < w; i++)
    for (INT_TYPE j = 0; j < h; j++) {
      imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
    }

  for (INT_TYPE j = 0; j < h; j++) {
    tm1 = SCALAR_VAL(0.0);
    ym1 = SCALAR_VAL(0.0);
    ym2 = SCALAR_VAL(0.0);
    for (INT_TYPE i = 0; i < w; i++) {
      y1[i][j] = a5 * imgOut[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
      tm1 = imgOut[i][j];
      ym2 = ym1;
      ym1 = y1[i][j];
    }
  }

  for (INT_TYPE j = 0; j < h; j++) {
    tp1 = SCALAR_VAL(0.0);
    tp2 = SCALAR_VAL(0.0);
    yp1 = SCALAR_VAL(0.0);
    yp2 = SCALAR_VAL(0.0);
    for (SINT_TYPE i = w - 1; i >= 0; i--) {
      y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
      tp2 = tp1;
      tp1 = imgOut[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }

  for (INT_TYPE i = 0; i < w; i++)
    for (INT_TYPE j = 0; j < h; j++)
      imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
#pragma endscop
  polybench_stop_instruments;
#endif
}

int main(int argc, char **argv) {
  INITIALIZE;

  /* Retrieve problem size. */
  INT_TYPE w = W;
  INT_TYPE h = H;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);

  /* Initialize array(s). */
  init_array(w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut));

  /* Run kernel. */
  kernel_deriche(w, h, alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut),
                 POLYBENCH_ARRAY(y1), POLYBENCH_ARRAY(y2));

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
