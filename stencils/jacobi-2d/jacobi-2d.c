void kernel_jacobi_2d(int tsteps, int n, double A[n][n], double B[n][n]) {
#pragma scop
  for (int t = 0; t < tsteps; t++) {
    for (int i = 1; i < n - 1; i++)
      for (int j = 1; j < n - 1; j++)
        B[i][j] = 0.2 * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] +
                         A[i - 1][j]);
    for (int i = 1; i < n - 1; i++)
      for (int j = 1; j < n - 1; j++)
        A[i][j] = 0.2 * (B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] +
                         B[i - 1][j]);
  }
#pragma endscop
}
