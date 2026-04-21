/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 3mm.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "3mm.h"

/* Array initialization. */
static void init_array(INT_TYPE ni, INT_TYPE nj, INT_TYPE nk, INT_TYPE nl,
                       INT_TYPE nm,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NJ, NM, nj, nm),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, D, NM, NL, nm, nl)) {
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nk; j++)
      ARRAY_2D_ACCESS(A, i, j) = (DATA_TYPE)((i * j + 1) % ni) / (5 * ni);
  for (INT_TYPE i = 0; i < nk; i++)
    for (INT_TYPE j = 0; j < nj; j++)
      ARRAY_2D_ACCESS(B, i, j) = (DATA_TYPE)((i * (j + 1) + 2) % nj) / (5 * nj);
  for (INT_TYPE i = 0; i < nj; i++)
    for (INT_TYPE j = 0; j < nm; j++)
      ARRAY_2D_ACCESS(C, i, j) = (DATA_TYPE)(i * (j + 3) % nl) / (5 * nl);
  for (INT_TYPE i = 0; i < nm; i++)
    for (INT_TYPE j = 0; j < nl; j++)
      ARRAY_2D_ACCESS(D, i, j) = (DATA_TYPE)((i * (j + 2) + 2) % nk) / (5 * nk);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(INT_TYPE ni, INT_TYPE nl,
                        ARRAY_2D_FUNC_PARAM(DATA_TYPE, G, NI, NL, ni, nl)) {
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("G");
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER,
              ARRAY_2D_ACCESS(G, i, j));
    }
  POLYBENCH_DUMP_END("G");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_3mm(INT_TYPE ni, INT_TYPE nj, INT_TYPE nk, INT_TYPE nl,
                       INT_TYPE nm,
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, E, NI, NJ, ni, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, A, NI, NK, ni, nk),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, B, NK, NJ, nk, nj),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, F, NJ, NL, nj, nl),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, C, NJ, NM, nj, nm),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, D, NM, NL, nm, nl),
                       ARRAY_2D_FUNC_PARAM(DATA_TYPE, G, NI, NL, ni, nl)) {
#if defined(POLYBENCH_USE_POLLY)
  polybench_start_instruments;
#if not defined(POLYBENCH_GPU) // CPU
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {ni, nj});
  const auto policy_2D_2 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {nj, nl});
  const auto policy_2D_3 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {ni, nl});
#else // GPU
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {ni, nj});
  const auto policy_2D_2 =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {nj, nl});
  const auto policy_2D_3 =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {ni, nl});
#endif

  Kokkos::parallel_for<Kokkos::usePolyOpt,
                       "p0.l==0, p1.l==0, p2.l==0,"
                       "p0.u0== p2.u0, p0.u1==p1.u0, p1.u1==p2.u1,"
                       // "p0.u1==nj,"
                       "p0.u0>10, p0.u1>10, p1.u1>10">(
      "kernel", policy_2D_1,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        E(i, j) = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < KOKKOS_LOOP_BOUND(nk); ++k)
          E(i, j) += A(i, k) * B(k, j);
      },
      policy_2D_2,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        F(i, j) = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < KOKKOS_LOOP_BOUND(nm); ++k)
          F(i, j) += C(i, k) * D(k, j);
      },
      policy_2D_3,
      KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        G(i, j) = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < KOKKOS_LOOP_BOUND(nj); ++k)
          G(i, j) += E(i, k) * F(k, j);
      });
  polybench_stop_instruments;
#elif defined(POLYBENCH_KOKKOS)
#if not defined(POLYBENCH_GPU) // CPU
  polybench_start_instruments;
  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {ni, nj},
                                                             {32, 32});
  const auto policy_2D_2 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {nj, nl},
                                                             {32, 32});
  const auto policy_2D_3 =
      Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<2>>({0, 0}, {ni, nl},
                                                             {32, 32});

  Kokkos::parallel_for(
      policy_2D_1, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        E(i, j) = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < nk; ++k)
          E(i, j) += A(i, k) * B(k, j);
      });

  Kokkos::parallel_for(
      policy_2D_2, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        F(i, j) = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < nm; ++k)
          F(i, j) += C(i, k) * D(k, j);
      });

  Kokkos::parallel_for(
      policy_2D_3, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        G(i, j) = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < nj; ++k)
          G(i, j) += E(i, k) * F(k, j);
      });
  polybench_stop_instruments;
#else                          // GPU
  polybench_GPU_array_2D(A, ni, nk);
  polybench_GPU_array_2D(B, nk, nj);
  polybench_GPU_array_2D(C, nj, nm);
  polybench_GPU_array_2D(D, nm, nl);
  polybench_GPU_array_2D(E, ni, nj);
  polybench_GPU_array_2D(F, nj, nl);
  polybench_GPU_array_2D(G, ni, nl);

  polybench_start_instruments;

  polybench_GPU_array_copy_to_device(A);
  polybench_GPU_array_copy_to_device(B);
  polybench_GPU_array_copy_to_device(C);
  polybench_GPU_array_copy_to_device(D);
  polybench_GPU_array_copy_to_device(E);
  polybench_GPU_array_copy_to_device(F);
  polybench_GPU_array_copy_to_device(G);

  const auto policy_2D_1 =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {ni, nj});
  const auto policy_2D_2 =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {nj, nl});
  const auto policy_2D_3 =
      Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::IndexType<int64_t>,
                            Kokkos::Rank<2>>({0, 0}, {ni, nl});

  Kokkos::parallel_for(
      policy_2D_1, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        DATA_TYPE sum = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < nk; ++k)
          sum += d_A(i, k) * d_B(k, j);
        d_E(i, j) = sum;
      });

  Kokkos::parallel_for(
      policy_2D_2, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        DATA_TYPE sum = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < nm; ++k)
          sum += d_C(i, k) * d_D(k, j);
        d_F(i, j) = sum;
      });

  Kokkos::parallel_for(
      policy_2D_3, KOKKOS_LAMBDA(const INT_TYPE i, const INT_TYPE j) {
        DATA_TYPE sum = SCALAR_VAL(0.0);
        for (INT_TYPE k = 0; k < nj; ++k)
          sum += d_E(i, k) * d_F(k, j);
        d_G(i, j) = sum;
      });

  polybench_GPU_array_copy_to_host(E);
  polybench_GPU_array_copy_to_host(F);
  polybench_GPU_array_copy_to_host(G);

  polybench_stop_instruments;

  polybench_GPU_array_sync_2D(E, ni, nj);
  polybench_GPU_array_sync_2D(F, nj, nl);
  polybench_GPU_array_sync_2D(G, ni, nl);
#endif
#else
  polybench_start_instruments;
#pragma scop
  /* E := A*B */
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nj; j++) {
      E[i][j] = SCALAR_VAL(0.0);
      for (INT_TYPE k = 0; k < nk; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }
  /* F := C*D */
  for (INT_TYPE i = 0; i < nj; i++)
    for (INT_TYPE j = 0; j < nl; j++) {
      F[i][j] = SCALAR_VAL(0.0);
      for (INT_TYPE k = 0; k < nm; ++k)
        F[i][j] += C[i][k] * D[k][j];
    }
  /* G := E*F */
  for (INT_TYPE i = 0; i < ni; i++)
    for (INT_TYPE j = 0; j < nl; j++) {
      G[i][j] = SCALAR_VAL(0.0);
      for (INT_TYPE k = 0; k < nj; ++k)
        G[i][j] += E[i][k] * F[k][j];
    }
#pragma endscop
  polybench_stop_instruments;
#endif
}

int main(int argc, char **argv) {

  INITIALIZE;

  /* Retrieve problem size. */
  INT_TYPE ni = NI;
  INT_TYPE nj = NJ;
  INT_TYPE nk = NK;
  INT_TYPE nl = NL;
  INT_TYPE nm = NM;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);

  /* Initialize array(s). */
  init_array(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
             POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

  /* Run kernel. */
  kernel_3mm(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(E), POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(C),
             POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(G));

  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);

  FINALIZE;

  return 0;
}
