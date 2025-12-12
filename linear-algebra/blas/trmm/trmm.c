void kernel_trmm(int m, int n, double alpha, double A[m][m], double B[m][n]) {
// BLAS parameters
// SIDE   = 'L'
// UPLO   = 'L'
// TRANSA = 'T'
// DIAG   = 'U'
//  => Form  B := alpha*A**T*B.
//  A is MxM
//  B is MxN
#pragma scop
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
  }
#pragma endscop
}
