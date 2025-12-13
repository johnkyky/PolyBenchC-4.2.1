#include <math.h>
#define EXP_FUN(x) expf(x)
#define POW_FUN(x, y) powf(x, y)

void kernel_deriche(int w, int h, double alpha, double imgIn[w][h],
                    double imgOut[w][h], double y1[w][h], double y2[w][h]) {
  double xm1, tm1, ym1, ym2;
  double xp1, xp2;
  double tp1, tp2;
  double yp1, yp2;
  double k;
  double a1, a2, a3, a4, a5, a6, a7, a8;
  double b1, b2, c1, c2;

  k = (1.0 - EXP_FUN(-alpha)) * (1.0 - EXP_FUN(-alpha)) /
      (1.0 + 2.0 * alpha * EXP_FUN(-alpha) - EXP_FUN(2.0 * alpha));
  a1 = a5 = k;
  a2 = a6 = k * EXP_FUN(-alpha) * (alpha - 1.0);
  a3 = a7 = k * EXP_FUN(-alpha) * (alpha + 1.0);
  a4 = a8 = -k * EXP_FUN(-2.0 * alpha);
  b1 = POW_FUN(2.0, -alpha);
  b2 = -EXP_FUN(-2.0 * alpha);
  c1 = c2 = 1;

#pragma scop
  for (int i = 0; i < w; i++) {
    ym1 = 0.0;
    ym2 = 0.0;
    xm1 = 0.0;
    for (int j = 0; j < h; j++) {
      y1[i][j] = a1 * imgIn[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
      xm1 = imgIn[i][j];
      ym2 = ym1;
      ym1 = y1[i][j];
    }
  }

  for (int i = 0; i < w; i++) {
    yp1 = 0.0;
    yp2 = 0.0;
    xp1 = 0.0;
    xp2 = 0.0;
    for (int j = h - 1; j >= 0; j--) {
      y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
      xp2 = xp1;
      xp1 = imgIn[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }

  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++) {
      imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
    }

  for (int j = 0; j < h; j++) {
    tm1 = 0.0;
    ym1 = 0.0;
    ym2 = 0.0;
    for (int i = 0; i < w; i++) {
      y1[i][j] = a5 * imgOut[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
      tm1 = imgOut[i][j];
      ym2 = ym1;
      ym1 = y1[i][j];
    }
  }

  for (int j = 0; j < h; j++) {
    tp1 = 0.0;
    tp2 = 0.0;
    yp1 = 0.0;
    yp2 = 0.0;
    for (int i = w - 1; i >= 0; i--) {
      y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
      tp2 = tp1;
      tp1 = imgOut[i][j];
      yp2 = yp1;
      yp1 = y2[i][j];
    }
  }

  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
      imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
#pragma endscop
}
