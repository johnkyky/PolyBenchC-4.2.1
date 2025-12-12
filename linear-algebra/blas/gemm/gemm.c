void kernel_gemm(int ni, int nj, int nk, double alpha, double beta,
                 double C[ni][nj], double A[ni][nk], double B[nk][nj]) {
// BLAS PARAMS
// TRANSA = 'N'
// TRANSB = 'N'
//  => Form C := alpha*A*B + beta*C,
// A is NIxNK
// B is NKxNJ
// C is NIxNJ
#pragma scop
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++)
      C[i][j] *= beta;
    for (int k = 0; k < nk; k++) {
      for (int j = 0; j < nj; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop
}
