/*******************************************************************************
 * Copyright (c) 2025 Cadence Design Systems, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to use this Software with Cadence processor cores only and
 * not with any other processors and platforms, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ******************************************************************************/

#include "xa_type_def.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_internal.h"
#include "tanhf_tbl.h"
#include "expf_tbl.h"

WORD32 xa_nn_tanh_f32_f32(FLOAT32 *p_out,
        const FLOAT32 *p_inp,
        WORD32 vec_length)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);

    /* vec_length should be greater than 0 */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), UNSUPPORTED_PARAM);

    xb_vecMxf32 xin, xmag, s, y, r, zout;

    /* Exponent and significand values, fixed-point polynomial value */
    xb_vecMx32 ex, fr, p;

    xb_vecMxf32 *__restrict__ p_x0 = (xb_vecMxf32*) p_inp;
    valign ax0 = PDX_LA_MXF32_PP(p_x0);

    xb_vecMxf32 *__restrict__ p_x1 = (xb_vecMxf32*) p_inp;
    valign ax1 = PDX_LA_MXF32_PP(p_x1);

    xb_vecMxf32 *__restrict__ p_z = (xb_vecMxf32*) p_out;
    valign az = PDX_Z_ALIGN();

    xb_vecMxf32 *__restrict__ p_z0 = (xb_vecMxf32*) p_out;

    const WORD32 *__restrict__ p_const = tbl;

    xb_vecMx80 w;

    xb_vecMxf32 half = PDX_CONST_MXF32(CONST_THREE);
    xb_vecMxf32 one = PDX_CONST_MXF32(CONST_ONE);
    xb_vecMxf32 zero_f = PDX_CONST_MXF32(0);

    vboolM b_sx, b_gt;

    /* Floating-point polynomial coeffs for tanh(x) */
    xb_vecMxf32 tn0, tn1, tn2, tn3;

    /* Powers of input value */
    xb_vecMxf32 x1, x2, x3;

    /* Pointer to polynomial coeffs. */
    const FLOAT32 *__restrict__ p_polytanhf_tbl = (FLOAT32*) polytanhf_tbl;

    circb tt = PDX_MOVC_AU32D((UWORD32) (tbl) + sizeof(tbl), (UWORD32) tbl);
    xb_vecMx32 t;

    WORD32 i;

    /* For values greater or equal to 0.5 */
    for (i = 0; i < (vec_length + PDX_M - CONST_ONE) / PDX_M; i++)
    {
        PDX_LAV_MXF32_XP(xin, ax0, p_x0,
                (UWORD8*) p_inp + vec_length * SIZE_OF_FLOAT - (UWORD8*) p_x0);


        /*
         * Convert floating-point input values to Q24. There is no special
         * processing for subnormal numbers: they are just flushed to zero,
         * and that's okay for this algorithm. Input sign is ignored.
         */
        xin = PDX_ABS_MXF32(xin);
        xin = PDX_MUL_MXF32(PDX_CONST_MXF32(CONST_TWO), xin);
        xin = PDX_MIN_MXF32(xin, MAX_LIMIT);
        fr = PDX_TRUNCU32_MXF32(xin, Q24_SHIFT_BITS);

        /*
         * Multiply by 1/ln2, extract the integer
         * and fractional (Q32) components.
         */

        /* Q54 <- Q24*Q30 */
        PDX_LSR_32_IP(t, p_const, SIZE_OF_INT);
        w = PDX_MULW_MX32(fr, t);
        /* Q0 <- Q54 - 54 */
        ex = PDX_PACKUSIV_MX80(w, EXPONENT_SHIFT_BITS);
        /* Q32 <- Q54 - 22  */
        w = PDX_SRAI_MX80(w, FRACTIONAL_COMPONENT_SHIFT);
        /* Unsigned Q32 */
        fr = PDX_PACKV_MX80(w);

        /*
         * Compute a polynomial approximation for 2^fr, Q30.
         */

        /* Q30 <- Q30 + [ Q32*Q30 - 32 w/ asym. rounding and saturation ] */
        p = expftbl_q30[0];
        w = PDX_MULUUW_MX32(fr, p);
        p = PDX_PACKQSRV_MX80(w, 0);
        p = PDX_ADD_MX32(expftbl_q30[CONST_ONE], p);
        w = PDX_MULUUW_MX32(fr, p);
        p = PDX_PACKQSRV_MX80(w, 0);
        p = PDX_ADD_MX32(expftbl_q30[CONST_TWO], p);
        w = PDX_MULUUW_MX32(fr, p);
        p = PDX_PACKQSRV_MX80(w, 0);
        p = PDX_ADD_MX32(expftbl_q30[CONST_THREE], p);
        w = PDX_MULUUW_MX32(fr, p);
        p = PDX_PACKQSRV_MX80(w, 0);
        p = PDX_ADD_MX32(expftbl_q30[CONST_FOUR], p);
        w = PDX_MULUUW_MX32(fr, p);
        p = PDX_PACKQSRV_MX80(w, 0);
        p = PDX_ADD_MX32(expftbl_q30[CONST_FIVE], p);
        w = PDX_MULUUW_MX32(fr, p);
        p = PDX_PACKQSRV_MX80(w, 0);
        p = PDX_ADD_MX32(expftbl_q30[CONST_SIX], p);

        /*
         * Convert (p*2^(ex-30))/2 to floating-point y == exp(2*x)/2
         */

        PDX_LSR_32_IP(t, p_const, SIZE_OF_INT);
        ex = PDX_ADD_MX32(ex, t);
        PDX_LSR_32_IC(t, p_const, tt);
        ex = PDX_MIN_MX32(ex, t);
        ex = PDX_SLLI_MX32(ex, Q24_SHIFT_BITS_MINUS_ONE);

        s = PDX_MOV_MXF32_FROM_4MX8(PDX_MOV_4MX8_FROM_MX32(ex));

        y = PDX_FLOATUF32_MX32(p, 0);
        /* y <- ( 1 + exp(2*x) )/2 */
        r = half;
        PDX_MULA_MXF32(r, s, y);

        /*
         * Compute the reciprocal value r <- 1/(0.5+y) == 2/(1+exp(2*x))
         */

        /* Due to Newton-Raphson refinement procedure,
         * 1/Inf turns into a NaN! */
        y = PDX_RECIP_MXF32(r);

        /*
         * Compute the result: z <- 1 - 1/y == 1 - 2/(1+exp(2*x))
         */

        r = PDX_SUB_MXF32(one, y);

        PDX_SAV_MXF32_XP(r, az, p_z,
                (UWORD8*) p_out + vec_length * SIZE_OF_FLOAT - (UWORD8*) p_z);
    }

    PDX_SAPOS_MXF32_FP(az, p_z);

    p_z = (xb_vecMxf32*) p_out;

    valign az0 = PDX_LA_MXF32_PP(p_z0);

    /* For values smaller than 0.5 */
    for(i = 0; i < (vec_length + PDX_M - CONST_ONE) / PDX_M; i++)
    {
        PDX_LAV_MXF32_XP(xin, ax1, p_x1,
                (UWORD8*) p_inp + vec_length * SIZE_OF_FLOAT - (UWORD8*) p_x1);

        /*
         * Take absolute value and classify it.
         */

        b_sx = PDX_LT_MX32(
                PDX_MOV_MX32_FROM_4MX8(PDX_MOV_4MX8_FROM_MXF32(xin)), 0);

        xmag = PDX_ABS_MXF32(xin);
        b_gt = PDX_OLT_MXF32(halfln3, xmag);

        /* Load results of indirect approximation. */
        PDX_LAV_MXF32_XP(y, az0, p_z0,
                (UWORD8*) p_out + vec_length * SIZE_OF_FLOAT - (UWORD8*) p_z0);

        /*
         * Compute the polynomial approximation to tanh(x).
         * This branch also provides the NaN propagation.
         */

        PDX_LSR_F32_IP(tn0, p_polytanhf_tbl, SIZE_OF_FLOAT);
        PDX_LSR_F32_IP(tn1, p_polytanhf_tbl, SIZE_OF_FLOAT);
        PDX_LSR_F32_IP(tn2, p_polytanhf_tbl, SIZE_OF_FLOAT);
        PDX_LSR_F32_XP(tn3, p_polytanhf_tbl, OFFSET_TO_BASE * SIZE_OF_FLOAT);

        /* Preset the output vector with indirect approximation results.
         * Puncture the argument vector to protect the results. */
        x1 = PDX_MOV_MXF32_T(zero_f, xmag, b_gt);
        zout = PDX_MOV_MXF32_T(y, x1, b_gt);

        x2 = PDX_MUL_MXF32(x1, x1);
        x3 = PDX_MUL_MXF32(x1, x2);

        PDX_MULA_MXF32(tn1, tn0, x2);
        PDX_MULA_MXF32(tn2, tn1, x2);
        PDX_MULA_MXF32(tn3, tn2, x2);
        PDX_MULA_MXF32(zout, tn3, x3);

        /*
         * Restore the input sign for both variants of approximation, then save
         * the resulting vector.
         */

        PDX_NEG_MXF32_T(zout, zout, b_sx);

        PDX_SAV_MXF32_XP(zout, az, p_z,
                (UWORD8*) p_out + vec_length * SIZE_OF_FLOAT - (UWORD8*) p_z);
    }

    PDX_SAPOS_MXF32_FP(az, p_z);

    return 0;
}
