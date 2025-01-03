/*******************************************************************************
* Copyright (c) 2024 Cadence Design Systems, Inc.
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
#include "expf_tbl.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_internal.h"

WORD32 xa_nn_elm_exp_f32_f32(FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp,
        WORD32 num_elm)
{

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    const xb_vecMxf32 *restrict p_input = (xb_vecMxf32*) p_inp;
    xb_vecMxf32 *restrict p_output = (xb_vecMxf32*) p_out;

    xb_vecMxf32 x0, out_val;
    valign align_x, align_z;

    align_x = PDX_LA_MXF32_PP(p_input);
    align_z = PDX_Z_ALIGN();

    WORD32 n;
    for (n = 0; n < (num_elm + PDX_M - CONST_ONE) >> LOG2_PDX_M; n++)
    {

        PDX_LAV_MXF32_XP(x0, align_x, p_input, num_elm * SIZE_OF_FLOAT );

        xb_vecMxf32 approx;
        xb_vecMx32 in_int, frac, exp, t, exp0, exp1;

        /* Q24 <- Q0 + 24 */
        in_int = PDX_TRUNC32_MXF32(x0, Q24_SHIFT_BITS);

        /* Multiply by 1/ln2, extract the integer and fractional (Q32) components.*/

        /* Q54 <- Q24*Q30 */
        xb_vecMx80 w0 = PDX_MULW_MX32(in_int, invln2_q30);

        /* Unsigned Q32 <- fract( Q54 - 22 )  */
        xb_vecMx80 w1 = PDX_SRAI_MX80(w0, FRACTIONAL_COMPONENT_SHIFT);

        /* Q31 <- Unsigned Q32 - 1 */
        frac = PDX_SRLI_MX32(PDX_PACKV_MX80(w1), CONST_ONE);

        /* Q0 <- Q54 - 54 */
        w1 = PDX_SRAI_MX80(w0, EXPONENT_SHIFT_BITS);
        exp = PDX_PACKV_MX80(w1);

        /*
         * Compute 2^fr through a polynomial approximation g(fr).
         */

        /* Q30 <- Q31*Q30 - 31 */
        xb_vecMx32 f2 = PDX_PACKSIV_MX80(PDX_MULW_MX32(frac, frac),
        POLYNOMIAL_APPROXIMATION_SHIFT);

        /* Load initial values from expftbl_q30 */
        xb_vecMx32 y1 = PDX_LSR_32_I(expftbl_q30, 0);
        xb_vecMx32 y2 = PDX_LSR_32_I(expftbl_q30, SIZE_OF_INT);

        xb_vecMx32 c1, c2;

        /* Series of polynomial approximations */
        c1 = PDX_LSR_32_I(expftbl_q30, CONST_TWO * SIZE_OF_INT);
        t = PDX_PACKSIV_MX80(PDX_MULW_MX32(f2, y1),
        POLYNOMIAL_APPROXIMATION_SHIFT);
        y1 = PDX_ADD_MX32(c1, t);

        c2 = PDX_LSR_32_I(expftbl_q30, CONST_THREE * SIZE_OF_INT);
        t = PDX_PACKSIV_MX80(PDX_MULW_MX32(f2, y2),
        POLYNOMIAL_APPROXIMATION_SHIFT);
        y2 = PDX_ADD_MX32(c2, t);

        c1 = PDX_LSR_32_I(expftbl_q30, CONST_FOUR * SIZE_OF_INT);
        t = PDX_PACKSIV_MX80(PDX_MULW_MX32(f2, y1),
        POLYNOMIAL_APPROXIMATION_SHIFT);
        y1 = PDX_ADD_MX32(c1, t);

        c2 = PDX_LSR_32_I(expftbl_q30, CONST_FIVE * SIZE_OF_INT);
        t = PDX_PACKSIV_MX80(PDX_MULW_MX32(f2, y2),
        POLYNOMIAL_APPROXIMATION_SHIFT);
        y2 = PDX_ADD_MX32(c2, t);

        c1 = PDX_LSR_32_I(expftbl_q30, CONST_SIX * SIZE_OF_INT);
        t = PDX_PACKSIV_MX80(PDX_MULW_MX32(f2, y1),
        POLYNOMIAL_APPROXIMATION_SHIFT);
        y1 = PDX_ADD_MX32(c1, t);

        /* Final polynomial approximation calculation */
        xb_vecMx32 g = PDX_ADD_MX32(y1,
        PDX_PACKSIV_MX80(PDX_MULW_MX32(frac, y2),
        POLYNOMIAL_APPROXIMATION_SHIFT));

        /* Convert fixed-point result to floating-point */
        approx = PDX_FLOATF32_MX32(g, Q31_SHIFT_BITS);

        /*
         *  Convert back to the floating point taking the original exponent into account
         *  Note: the dynamic range of exponent exceeds 8 bits so it is splitted by 2 smaller
         *  values
         */
        exp1 = PDX_SRAI_MX32(exp, CONST_ONE);
        exp0 = PDX_SUB_MX32(exp, exp1);

        exp0 = PDX_ADD_MX32(EXPONENT_BIAS, exp0);
        exp1 = PDX_ADD_MX32(EXPONENT_BIAS, exp1);

        exp0 = PDX_SLLI_MX32(exp0, Q24_SHIFT_BITS_MINUS_ONE);
        exp1 = PDX_SLLI_MX32(exp1, Q24_SHIFT_BITS_MINUS_ONE);

        /* Move exp0 and exp1 to scale0 and scale1 as 32-bit floating-point vectors */
        xb_vecMxf32 scale0 = PDX_MOV_MXF32_FROM_4MX8(
        PDX_MOV_4MX8_FROM_MX32(exp0));

        xb_vecMxf32 scale1 = PDX_MOV_MXF32_FROM_4MX8(
        PDX_MOV_4MX8_FROM_MX32(exp1));

        /* Compute final scaled values */
        out_val = PDX_MUL_MXF32(approx, scale0);
        out_val = PDX_MUL_MXF32(out_val, scale1);

        PDX_SAV_MXF32_XP(out_val, align_z, p_output, num_elm * SIZE_OF_FLOAT );

    }

    PDX_SAPOS_MXF32_FP(align_z, p_output);

    return 0;
}
