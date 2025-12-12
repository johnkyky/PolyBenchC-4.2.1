static void kernel_gesummv(int n, double alpha, double beta, double A[n][n],
                           double B[n][n], double tmp[n], double x[n],
                           double y[n]) {
#pragma scop
  for (int i = 0; i < n; i++) {
    tmp[i] = 0.0;
    y[i] = 0.0;
    for (int j = 0; j < n; j++) {
      tmp[i] = A[i][j] * x[j] + tmp[i];
      y[i] = B[i][j] * x[j] + y[i];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
#pragma endscop
}
