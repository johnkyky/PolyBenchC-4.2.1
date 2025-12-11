static void kernel_fdtd_2d(int tmax, int nx, int ny, double ex[nx][ny],
                           double ey[nx][ny], double hz[nx][ny],
                           double _fict_[tmax]) {
#pragma scop
  for (int t = 0; t < tmax; t++) {
    for (int j = 0; j < ny; j++)
      ey[0][j] = _fict_[t];
    for (int i = 1; i < nx; i++)
      for (int j = 0; j < ny; j++)
        ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i - 1][j]);
    for (int i = 0; i < nx; i++)
      for (int j = 1; j < ny; j++)
        ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j - 1]);
    for (int i = 0; i < nx - 1; i++)
      for (int j = 0; j < ny - 1; j++)
        hz[i][j] = hz[i][j] -
                   0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
  }
#pragma endscop
}
