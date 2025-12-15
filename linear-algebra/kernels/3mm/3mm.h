/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#ifndef _3MM_H
#define _3MM_H

/* Default to MEDIUM_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) &&                       \
    !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) &&                     \
    !defined(EXTRALARGE_DATASET)
#define MEDIUM_DATASET
#endif

#if !defined(NI) && !defined(NJ) && !defined(NK) && !defined(NL) && !defined(NM)
/* Define sample dataset sizes. */
#ifdef MINI_DATASET
#define NI 32
#define NJ 40
#define NK 48
#define NL 56
#define NM 64
#endif

#ifdef SMALL_DATASET
#define NI 528
#define NJ 536
#define NK 544
#define NL 552
#define NM 568
#endif

#ifdef MEDIUM_DATASET
#define NI 1012
#define NJ 1028
#define NK 1044
#define NL 1060
#define NM 1076
#endif

#ifdef LARGE_DATASET
#define NI 5024
#define NJ 5048
#define NK 5072
#define NL 5096
#define NM 5120
#endif

#ifdef EXTRALARGE_DATASET
#define NI 10048
#define NJ 10080
#define NK 10112
#define NL 10144
#define NM 10176
#endif

#endif /* !(NI NJ NK NL NM) */

#define _PB_NI POLYBENCH_LOOP_BOUND(NI, ni)
#define _PB_NJ POLYBENCH_LOOP_BOUND(NJ, nj)
#define _PB_NK POLYBENCH_LOOP_BOUND(NK, nk)
#define _PB_NL POLYBENCH_LOOP_BOUND(NL, nl)
#define _PB_NM POLYBENCH_LOOP_BOUND(NM, nm)

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

#endif /* !_3MM_H */
