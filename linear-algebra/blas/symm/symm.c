static void kernel_symm(int m, int n, double alpha, double beta, double C[m][n],
                        double A[m][m], double B[m][n]) {
  double temp2;

  // BLAS PARAMS
  // SIDE = 'L'
  // UPLO = 'L'
  //  =>  Form  C := alpha*A*B + beta*C
  //  A is MxM
  //  B is MxN
  //  C is MxN
  // note that due to Fortran array layout, the code below more closely
  // resembles upper triangular case in BLAS

#pragma scop
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      temp2 = 0;
      for (int k = 0; k < i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
#pragma endscop
}
