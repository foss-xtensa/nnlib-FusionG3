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
#include "sigmoid_tbl.h"

WORD32 xa_nn_sigmoid_f32_f32(FLOAT32 *p_out,
        const FLOAT32 *p_inp,
        WORD32 vec_length)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), UNSUPPORTED_PARAM);

    xb_vecMxf32 *restrict p_input = (xb_vecMxf32*) p_inp;
    xb_vecMxf32 *restrict p_output = (xb_vecMxf32*) p_out;

    valign az, ax;
    ax = PDX_LA_MXF32_PP(p_input);
    az = PDX_Z_ALIGN();

    xb_vecMxf32 x, x0, y, t, d, d2, z0, z1, z, s0, in_val, out_val;
    vboolM s;
    xb_vecMx32 n, n0, n1;
    WORD32 i;
    for (i = 0; i < (vec_length + PDX_M - CONST_ONE) >> LOG2_PDX_M; i++)
    {
        PDX_LAV_MXF32_XP(in_val, ax, p_input,
                (UWORD8*) p_inp + vec_length * SIZE_OF_FLOAT
                        - (UWORD8*) p_input);

        /* Apply condition: if x < 0, negate it, else leave it as is */
        s = PDX_OLT_MXF32(in_val, 0.f);
        x = PDX_NEG_MXF32(PDX_ABS_MXF32(in_val));
        x = PDX_MOV_MXF32_T(SIGMOID_MIN_BOUND, x,
        PDX_OLT_MXF32(x, SIGMOID_MIN_BOUND));

        /* Compute d+n=log2(e)*x */
        y = PDX_FIROUND_MXF32(PDX_MUL_MXF32(x, c[0]));
        d = PDX_NEG_MXF32(y);
        PDX_MULAN_MXF32(d, x, c[0]);
        PDX_MULAN_MXF32(d, x, c[1]);
        n = PDX_TRUNC32_MXF32(y, 0);
        {
            /* Approx 2^d */
            d2 = PDX_MUL_MXF32(d, d);
            z0 = p[0];
            t = p[2];
            PDX_MULAN_MXF32(t, d2, z0);
            z0 = t;
            t = p[4];
            PDX_MULAN_MXF32(t, d2, z0);
            z0 = t;
            z1 = p[1];
            t = p[3];
            PDX_MULAN_MXF32(t, d2, z1);
            z1 = t;
            t = p[5];
            PDX_MULAN_MXF32(t, d2, z1);
            z1 = t;
            PDX_MULAN_MXF32(z1, z0, d);
            z = z1;
        }
        t = CONST_ONE;
        PDX_MULAN_MXF32(t, d, z);
        z = t;

        /* Calculate scaling factor for the exponent */
        s0 = PDX_MOV_MXF32_FROM_MX32(
        PDX_SLLI_MX32(PDX_MAX_MX32(PDX_ADD_MX32(n, EXPONENT_BIAS), 0),
        Q24_SHIFT_BITS_MINUS_ONE));
        x0 = z;
        x0 = PDX_MUL_MXF32(x0, s0);

        /* Simplified ldexpf */
        n0 = PDX_SRAI_MX32(n, CONST_ONE);
        n1 = PDX_SUB_MX32(n, n0);
        n0 = PDX_SLLI_MX32(PDX_ADD_MX32(n0, EXPONENT_BIAS),
        Q24_SHIFT_BITS_MINUS_ONE);
        n1 = PDX_SLLI_MX32(PDX_ADD_MX32(n1, EXPONENT_BIAS),
        Q24_SHIFT_BITS_MINUS_ONE);

        /* Apply final scaling to x */
        x = PDX_MUL_MXF32(z,
        PDX_MOV_MXF32_FROM_MX32(n0));
        x = PDX_MUL_MXF32(x,
        PDX_MOV_MXF32_FROM_MX32(n1));
        /* approx y=1/(1+x); */
        out_val = PDX_RECIP_MXF32(PDX_ADD_MXF32(CONST_ONE, x0));
        PDX_MUL_MXF32_T(out_val, out_val, x, s);

        /* Store the output */
        PDX_SAV_MXF32_XP(out_val, az, p_output,
                (UWORD8*) p_out + vec_length * SIZE_OF_FLOAT
                        - (UWORD8*) p_output);

    }
    /* Flushing the output register */
    PDX_SAPOS_MXF32_FP(az, p_output);
    return 0;
}
