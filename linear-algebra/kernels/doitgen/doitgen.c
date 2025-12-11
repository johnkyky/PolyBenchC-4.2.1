void kernel_doitgen(int nr, int nq, int np, double A[nr][nq][np],
                    double tmp[nr][nq][np], double C4[np][np], double sum[np]) {
#pragma scop
  for (int r = 0; r < nr; r++)
    for (int q = 0; q < nq; q++) {
      for (int p = 0; p < np; p++) {
        sum[p] = 0.0;
        for (int s = 0; s < np; s++)
          sum[p] += A[r][q][s] * C4[s][p];
      }
      for (int p = 0; p < np; p++)
        A[r][q][p] = sum[p];
    }
#pragma endscop
}
