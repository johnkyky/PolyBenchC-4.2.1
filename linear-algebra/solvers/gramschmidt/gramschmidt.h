/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#ifndef _GRAMSCHMIDT_H
#define _GRAMSCHMIDT_H

/* Default to MEDIUM_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) &&                       \
    !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) &&                     \
    !defined(EXTRALARGE_DATASET)
#define MEDIUM_DATASET
#endif

#if !defined(M) && !defined(N)
/* Define sample dataset sizes. */
#ifdef MINI_DATASET
#define M 32
#define N 40
#endif

#ifdef SMALL_DATASET
#define M 256
#define N 272
#endif

#ifdef MEDIUM_DATASET
#define M 1024
#define N 1048
#endif

#ifdef LARGE_DATASET
#define M 2048
#define N 2080
#endif

#ifdef EXTRALARGE_DATASET
#define M 4096
#define N 4136
#endif

#endif /* !(M N) */

#define _PB_M POLYBENCH_LOOP_BOUND(M, m)
#define _PB_N POLYBENCH_LOOP_BOUND(N, n)

/* Default data type */
#if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) &&              \
    !defined(DATA_TYPE_IS_DOUBLE)
#define DATA_TYPE_IS_DOUBLE
#endif

#ifdef DATA_TYPE_IS_INT
#define DATA_TYPE int
#define DATA_PRINTF_MODIFIER "%d "
#endif

#ifdef DATA_TYPE_IS_FLOAT
#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.2f "
#define SCALAR_VAL(x) x##f
#define SQRT_FUN(x) sqrtf(x)
#define EXP_FUN(x) expf(x)
#define POW_FUN(x, y) powf(x, y)
#endif

#ifdef DATA_TYPE_IS_DOUBLE
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define SCALAR_VAL(x) x
#define SQRT_FUN(x) sqrt(x)
#define EXP_FUN(x) exp(x)
#define POW_FUN(x, y) pow(x, y)
#endif

#endif /* !_GRAMSCHMIDT_H */
