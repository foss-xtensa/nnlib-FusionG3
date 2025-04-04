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

WORD32 xa_nn_elm_rsqrt_f32_f32(FLOAT32 *p_out,
        const FLOAT32 *p_inp,
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

    const xb_vecMxf32 *__restrict__ p_input = (xb_vecMxf32*) p_inp;
    xb_vecMxf32 *__restrict__ p_output = (xb_vecMxf32*) p_out;

    xb_vecMxf32 x0, z0, vt1, vt2, vt3, t4, t5, t6, t7, t8;
    valign ax, az;

    /* Priming align registers */
    ax = PDX_LA_MXF32_PP(p_input);
    az = PDX_Z_ALIGN();

    WORD32 n;

    /* Process the input vector in chunks of size PDX_M */
    for (n = 0; n < (num_elm + PDX_M - CONST_ONE) >> LOG2_PDX_M; n++)
    {
        /* Load a block of input data */
        PDX_LAV_MXF32_XP(x0, ax, p_input,
                (UWORD8*) p_inp + num_elm * SIZE_OF_FLOAT - (UWORD8*) p_input);

        /* Steps to compute the reciprocal square root of the input block */

        /* Initial reciprocal square root approximation */
        vt1 = PDX_RSQRT0_MXF32(x0);

        /* Multiply input with the approximation */
        vt2 = PDX_MUL_MXF32(x0, vt1);
        vt3 = PDX_CONST_MXF32(CONST_THREE);

        t4 = vt1;
        /* Adjust exponent for better precision */
        PDX_ADDEXP_MXF32(vt1, vt3);

        t5 = PDX_CONST_MXF32(CONST_ONE);
        PDX_MULS_MXF32(t5, vt2, t4);

        t7 = vt1;
        PDX_MULAN_MXF32(t4, t7, t5);

        /* Multiply refined value with original input */
        t6 = PDX_MUL_MXF32(x0, t4);

        z0 = t4;
        PDX_ADDEXP_MXF32(t4, vt3);

        t8 = PDX_CONST_MXF32(CONST_ONE);
        PDX_MULS_MXF32(t8, t6, z0);
        PDX_MULAN_MXF32(z0, t4, t8);

        /* Store the result */
        PDX_SAV_MXF32_XP(z0, az, p_output,
               (UWORD8*) p_out + num_elm * SIZE_OF_FLOAT - (UWORD8*) p_output);
    }
    /* Flushing output align register */
    PDX_SAPOS_MXF32_FP(az, p_output);

    return 0;
}
