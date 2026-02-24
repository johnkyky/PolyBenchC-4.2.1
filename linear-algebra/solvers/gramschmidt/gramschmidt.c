#include <math.h>
void kernel_gramschmidt(int m, int n, double A[m][n], double R[n][n],
                        double Q[m][n]) {
#pragma scop
  for (int k = 0; k < n; k++) {
    double nrm = 0.0;

    for (int i = 0; i < m; i++)
      nrm += A[i][k] * A[i][k];

    R[k][k] = sqrt(nrm);

    for (int i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];

    for (int j = k + 1; j < n; j++) {
      R[k][j] = 0.0;
      for (int i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (int i = 0; i < m; i++)
        A[i][j] = A[i][j] - Q[i][k] * R[k][j];
    }
  }
#pragma endscop
}
