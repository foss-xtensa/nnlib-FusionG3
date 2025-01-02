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

#include "xa_type_def.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_internal.h"

/* Without broadcasting */
WORD32 xa_nn_elm_div_f32xf32_f32(FLOAT32 *p_out,
        const FLOAT32 *p_inp1,
        const FLOAT32 *p_inp2,
        WORD32 mode,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), UNSUPPORTED_PARAM);

    /* Invalid input checks */
    /* mode should be either 0 or 1 or 2 */
    XA_NNLIB_ARG_CHK_COND((mode < 0) || (mode > CONST_TWO), UNSUPPORTED_PARAM);
    /* num_elm should be greater than 0 */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    /* Variables for output, input1 and input2 */
    xb_vecMxf32 z, x, y;
    valign ax, ay, az;

    /* Pointer for input1 */
    const xb_vecMxf32 *__restrict__ p_x = (const xb_vecMxf32*) p_inp1;
    /* Priming for input1 load */
    ax = PDX_LA_MXF32_PP(p_x);

    /* Pointer for input2 */
    const xb_vecMxf32 *__restrict__ p_y = (const xb_vecMxf32*) p_inp2;
    /* Priming for input2 load */
    ay = PDX_LA_MXF32_PP(p_y);

    /* Pointer for output */
    xb_vecMxf32 *__restrict__ p_z = (xb_vecMxf32*) p_out;
    /* Priming for output store */
    az = PDX_Z_ALIGN();

    WORD32 num_simd4_ops = num_elm >> LOG2_PDX_M;
    WORD32 m = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;

    WORD32 i;

    switch (mode)
    {
        case 0:  /* Normal div output */
            /* Unroll the loop by x4 for SIMD */
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load the 4 elements from input1 */
                PDX_LA_MXF32_IP(x, ax, p_x);
                /* Load the 4 elements from input2 */
                PDX_LA_MXF32_IP(y, ay, p_y);
                /* Getting x/y */
                z = PDX_DIV_MXF32(x, y);
                /* Store the output */
                PDX_SA_MXF32_IP(z, az, p_z);
            }
            break;
        case CONST_ONE:  /* Truncated div output */
            /* Unroll the loop by x4 for SIMD */
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load the 4 elements from input1 */
                PDX_LA_MXF32_IP(x, ax, p_x);
                /* Load the 4 elements from input2 */
                PDX_LA_MXF32_IP(y, ay, p_y);
                /* Getting x/y */
                z = PDX_DIV_MXF32(x, y);
                /* Truncating the output */
                z = PDX_FITRUNC_MXF32(z);
                /* Store the output */
                PDX_SA_MXF32_IP(z, az, p_z);
            }
            break;
        case CONST_TWO:  /* Floored div output */
            /* Unroll the loop by x4 for SIMD */
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load the 4 elements from input1 */
                PDX_LA_MXF32_IP(x, ax, p_x);
                /* Load the 4 elements from input2 */
                PDX_LA_MXF32_IP(y, ay, p_y);
                /* Getting x/y */
                z = PDX_DIV_MXF32(x, y);
                /* Getting floored output */
                z = PDX_FIFLOOR_MXF32(z);
                /* Store the output */
                PDX_SA_MXF32_IP(z, az, p_z);
            }
    }
    /* Remaining iterations of inner loop */
    PDX_LAV_MXF32_XP(x, ax, p_x, m);
    PDX_LAV_MXF32_XP(y, ay, p_y, m);
    z = PDX_DIV_MXF32(x, y);
    switch (mode)
    {
        case CONST_ONE:  /* Truncated div output */
            z = PDX_FITRUNC_MXF32(z);
            break;
        case CONST_TWO:  /* Floored div output */
            z = PDX_FIFLOOR_MXF32(z);
    }
    PDX_SAV_MXF32_XP(z, az, p_z, m);
    PDX_SAPOS_MXF32_FP(az, p_z);

    return 0;
}

/* Scalar operations */
WORD32 xa_nn_elm_div_scalar_f32xf32_f32(FLOAT32 *p_out,
        const FLOAT32 *p_inp1,
        const FLOAT32 inp2,
        WORD32 mode,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), UNSUPPORTED_PARAM);

    /* Invalid input checks */
    /* mode should be either 0 or 1 or 2 */
    XA_NNLIB_ARG_CHK_COND((mode < 0) || (mode > CONST_TWO), UNSUPPORTED_PARAM);
    /* num_elm should be greater than 0 */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    /* Variables for output, input1 and input2 */
    xb_vecMxf32 z, x, y;
    valign ax, az;

    /* Pointer for input1 */
    const xb_vecMxf32 *__restrict__ p_x = (const xb_vecMxf32*) p_inp1;
    /* Priming for input1 load */
    ax = PDX_LA_MXF32_PP(p_x);

    y = inp2;

#ifndef ENABLE_HIGH_PRECISION
    xb_vecMxf32 one = CONST_ONE;
    /* Calculating 1/y */
    xb_vecMxf32 one_over_y = PDX_DIV_MXF32(one, y);
#endif

    /* Pointer for output */
    xb_vecMxf32 *__restrict__ p_z = (xb_vecMxf32*) p_out;
    /* Priming for output store */
    az = PDX_Z_ALIGN();

    WORD32 num_simd4_ops = num_elm >> LOG2_PDX_M;
    WORD32 m = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;

    WORD32 i;

    switch (mode)
    {
        case 0:  /* Normal div output */
            /* Unroll the loop by x4 for SIMD */
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load the 4 elements from input1 */
                PDX_LA_MXF32_IP(x, ax, p_x);
#ifdef ENABLE_HIGH_PRECISION
                z = PDX_DIV_MXF32(x, y);
#else
                z = PDX_MUL_MXF32(x, one_over_y);
#endif
                PDX_SA_MXF32_IP(z, az, p_z);
            }
            break;
        case CONST_ONE:  /* Truncated div output */
            /* Unroll the loop by x4 for SIMD */
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load the 4 elements from input1 */
                PDX_LA_MXF32_IP(x, ax, p_x);
#ifdef ENABLE_HIGH_PRECISION
                z = PDX_DIV_MXF32(x, y);
#else
                /* Multiplying x and 1/y */
                z = PDX_MUL_MXF32(x, one_over_y);
#endif
                /* Truncating the output */
                z = PDX_FITRUNC_MXF32(z);
                PDX_SA_MXF32_IP(z, az, p_z);
            }
            break;
        case CONST_TWO:  /* Floored div output */
            /* Unroll the loop by x4 for SIMD */
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load the 4 elements from input1 */
                PDX_LA_MXF32_IP(x, ax, p_x);
#ifdef ENABLE_HIGH_PRECISION
                z = PDX_DIV_MXF32(x, y);
#else
                /* Multiplying x and 1/y */
                z = PDX_MUL_MXF32(x, one_over_y);
#endif
                /* Getting floored output */
                z = PDX_FIFLOOR_MXF32(z);
                PDX_SA_MXF32_IP(z, az, p_z);
            }
    }
    PDX_LAV_MXF32_XP(x, ax, p_x, m);
#ifdef ENABLE_HIGH_PRECISION
    z = PDX_DIV_MXF32(x, y);
#else
    z = PDX_MUL_MXF32(x, one_over_y);
#endif
    switch (mode)
    {
        case CONST_ONE:  /* Truncated div output */
            z = PDX_FITRUNC_MXF32(z);
            break;
        case CONST_TWO:  /* Floored div output */
            z = PDX_FIFLOOR_MXF32(z);
    }
    PDX_SAV_MXF32_XP(z, az, p_z, m);
    PDX_SAPOS_MXF32_FP(az, p_z);

    return 0;
}

static inline WORD32 check_shapes(const WORD32 *const p_inp1_shape,
        const WORD32 *const p_inp2_shape,
        const WORD32 *const p_out_shape)
{
    WORD32 i;
    /* Check the shapes of input and output */
    for (i = 0; i < MAX_DIMS; i++)
    {
        if (((p_inp1_shape[i] != p_inp2_shape[i])
                && (p_inp1_shape[i] != CONST_ONE)
                && (p_inp2_shape[i] != CONST_ONE))
                || (p_out_shape[i]
                        != (p_inp1_shape[i] > p_inp2_shape[i] ?
                                p_inp1_shape[i] : p_inp2_shape[i])))
        {
            return UNSUPPORTED_PARAM;
        }
    }
    return 0;
}

static inline void shapes_convert_5D(WORD32 *const __restrict__ p_5d_out_shape,
        WORD32 *const __restrict__ p_5d_inp1_shape,  /* new input1 shapes */
        WORD32 *const __restrict__ p_5d_inp2_shape,  /* new input2 shapes */
        const WORD32 *const __restrict__ p_out_shape,
        const WORD32 *const __restrict__ p_inp1_shape,  /* original input1 shapes */
        const WORD32 *const __restrict__ p_inp2_shape,  /* original input1 shapes */
        const WORD32 num_inp_dims)
{
    WORD32 i;
    /* Convert number of dimension less than 5D to 5D */
    for (i = 0; i < num_inp_dims; i++)
    {
        p_5d_out_shape[i + MAX_DIMS - num_inp_dims] = p_out_shape[i];
        p_5d_inp1_shape[i + MAX_DIMS - num_inp_dims] = p_inp1_shape[i];
        p_5d_inp2_shape[i + MAX_DIMS - num_inp_dims] = p_inp2_shape[i];
    }
}

static inline void strides_calculation(const WORD32 *const p_5d_inp1_shape,
        const WORD32 *const p_5d_inp2_shape,
        WORD32 *const inp1_strides,
        WORD32 *const inp2_strides)
{
    WORD32 i;

    inp1_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    inp2_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    for (i = MAX_DIMS - CONST_TWO; i >= 0; i--)
    {
        inp1_strides[i] = inp1_strides[i + CONST_ONE]
                * p_5d_inp1_shape[i + CONST_ONE];
        inp2_strides[i] = inp2_strides[i + CONST_ONE]
                * p_5d_inp2_shape[i + CONST_ONE];
    }
}

static inline void internal_elm_div_broadcast_2D_f32xf32_f32(
        FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp1,
        const FLOAT32 *__restrict__ p_inp2,
        WORD32 out_lc,
        WORD32 in_lc,
        const WORD32 *input1_shapes,
        const WORD32 mode)
{
    /* Variables for quotient, input1 and input2 */
    xb_vecMxf32 z0, z1, x0, x1, y0, y1;

    /* Align registers for input1 and input2 */
    valign ax0, ax1, ay0, ay1;
    /* align registers for output */
    valign az0, az1;

    /* Pointer for base address for input1 */
    const xb_vecMxf32 *__restrict__ p_x0 = (const xb_vecMxf32*) p_inp1;
    /* Pointer for middle address for input1 */
    const xb_vecMxf32 *__restrict__ p_x1 = (const xb_vecMxf32*) (p_inp1
            + ((out_lc / CONST_TWO) * in_lc));

    /* Priming for input1 loads */
    ax0 = PDX_LA_MXF32_PP(p_x0);
    ax1 = PDX_LA_MXF32_PP(p_x1);

    /* Pointer for base address for input2 */
    const xb_vecMxf32 *__restrict__ p_y0 = (const xb_vecMxf32*) p_inp2;
    /* Pointer for middle address for input2 */
    const xb_vecMxf32 *__restrict__ p_y1 = (const xb_vecMxf32*) (p_inp2
            + ((out_lc / CONST_TWO) * in_lc));

    /* Priming for input2 loads */
    ay0 = PDX_LA_MXF32_PP(p_y0);
    ay1 = PDX_LA_MXF32_PP(p_y1);

    /* Pointer for base address for output */
    xb_vecMxf32 *__restrict__ p_z0 = (xb_vecMxf32*) p_out;
    /* Pointer for middle address for output */
    xb_vecMxf32 *__restrict__ p_z1 = (xb_vecMxf32*) (p_out
            + ((out_lc / CONST_TWO) * in_lc));

    /* Priming for output stores */
    az0 = PDX_Z_ALIGN();
    az1 = PDX_Z_ALIGN();

    WORD32 num_simd4_ops = in_lc >> LOG2_PDX_M;
    WORD32 m = (in_lc & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;

    WORD32 i, j;

    /* If the second from last dim of input1 itself is broadcastable */
    if (input1_shapes[3] == CONST_ONE)
    {
        switch (mode)
        {
            case 0:  /* Normal div output */
                /* Unroll the loop by x2 for SIMD */
                for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        /* Load the 4 elements from input1 */
                        PDX_LA_MXF32_IP(x0, ax0, p_x0);

                        /* Load the 4 elements from input2 base address */
                        PDX_LA_MXF32_IP(y0, ay0, p_y0);
                        /* Load the 4 elements from input2 Middle address */
                        PDX_LA_MXF32_IP(y1, ay1, p_y1);

                        /* Dividing x0 by y0 */
                        z0 = PDX_DIV_MXF32(x0, y0);
                        /* Dividing x0 by y1 */
                        z1 = PDX_DIV_MXF32(x0, y1);

                        /* Store the output */
                        PDX_SA_MXF32_IP(z0, az0, p_z0);
                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }
                    /* Remaining iterations of inner loop */
                    PDX_LAV_MXF32_XP(x0, ax0, p_x0, m);

                    PDX_LAV_MXF32_XP(y0, ay0, p_y0, m);
                    PDX_LAV_MXF32_XP(y1, ay1, p_y1, m);

                    z0 = PDX_DIV_MXF32(x0, y0);
                    z1 = PDX_DIV_MXF32(x0, y1);

                    PDX_SAV_MXF32_XP(z0, az0, p_z0, m);
                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);

                    /* Input1 Pointer updates to base address
                     * as input1 is broadcasted
                     */
                    p_x0 = (const xb_vecMxf32*) p_inp1;
                    ax0 = PDX_LA_MXF32_PP(p_x0);
                }
                PDX_SAPOS_MXF32_FP(az0, p_z0);

                /* Loop through remaining iterations of outer loop */
                if ((out_lc % CONST_TWO) != 0)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        PDX_LA_MXF32_IP(x0, ax0, p_x0);

                        PDX_LA_MXF32_IP(y1, ay1, p_y1);

                        z1 = PDX_DIV_MXF32(x0, y1);

                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }
                    /* Remaining iterations */
                    PDX_LAV_MXF32_XP(x0, ax0, p_x0, m);

                    PDX_LAV_MXF32_XP(y1, ay1, p_y1, m);

                    z1 = PDX_DIV_MXF32(x0, y1);

                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);
                }
                PDX_SAPOS_MXF32_FP(az1, p_z1);
                break;
            case CONST_ONE:  /* Truncated div output */
                /* Unroll the loop by x2 for SIMD */
                for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        /* Load the 4 elements from input1 */
                        PDX_LA_MXF32_IP(x0, ax0, p_x0);

                        /* Load the 4 elements from input2 base address */
                        PDX_LA_MXF32_IP(y0, ay0, p_y0);
                        /* Load the 4 elements from input2 Middle address */
                        PDX_LA_MXF32_IP(y1, ay1, p_y1);

                        /* Dividing x0 by y0 */
                        z0 = PDX_DIV_MXF32(x0, y0);
                        /* Truncating result in z0 towards 0 */
                        z0 = PDX_FITRUNC_MXF32(z0);

                        /* Dividing x0 by y1 */
                        z1 = PDX_DIV_MXF32(x0, y1);
                        /* Truncating result in z1 towards 0 */
                        z1 = PDX_FITRUNC_MXF32(z1);

                        /* Store the output */
                        PDX_SA_MXF32_IP(z0, az0, p_z0);
                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }
                    /* Remaining iterations of inner loop */
                    PDX_LAV_MXF32_XP(x0, ax0, p_x0, m);

                    PDX_LAV_MXF32_XP(y0, ay0, p_y0, m);
                    PDX_LAV_MXF32_XP(y1, ay1, p_y1, m);

                    z0 = PDX_DIV_MXF32(x0, y0);
                    z1 = PDX_DIV_MXF32(x0, y1);

                    z0 = PDX_FITRUNC_MXF32(z0);
                    z1 = PDX_FITRUNC_MXF32(z1);

                    PDX_SAV_MXF32_XP(z0, az0, p_z0, m);
                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);

                    /* Input1 Pointer updates to base address
                     * as input1 is broadcasted
                     */
                    p_x0 = (const xb_vecMxf32*) p_inp1;
                    ax0 = PDX_LA_MXF32_PP(p_x0);
                }
                PDX_SAPOS_MXF32_FP(az0, p_z0);

                /* Loop through remaining iterations of outer loop */
                if ((out_lc % CONST_TWO) != 0)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        PDX_LA_MXF32_IP(x0, ax0, p_x0);

                        PDX_LA_MXF32_IP(y1, ay1, p_y1);

                        z1 = PDX_DIV_MXF32(x0, y1);
                        z1 = PDX_FITRUNC_MXF32(z1);

                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }

                    /* Remaining iterations */
                    PDX_LAV_MXF32_XP(x0, ax0, p_x0, m);

                    PDX_LAV_MXF32_XP(y1, ay1, p_y1, m);

                    z1 = PDX_DIV_MXF32(x0, y1);
                    z1 = PDX_FITRUNC_MXF32(z1);

                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);
                }
                PDX_SAPOS_MXF32_FP(az1, p_z1);
                break;
            case CONST_TWO:  /* Floored div output */
                /* Unroll the loop by x2 for SIMD */
                for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        /* Load the 4 elements from input1 */
                        PDX_LA_MXF32_IP(x0, ax0, p_x0);

                        /* Load the 4 elements from input2 base address */
                        PDX_LA_MXF32_IP(y0, ay0, p_y0);
                        /* Load the 4 elements from input2 Middle address */
                        PDX_LA_MXF32_IP(y1, ay1, p_y1);

                        /* Dividing x0 by y0 */
                        z0 = PDX_DIV_MXF32(x0, y0);
                        /* Dividing x0 by y1 */
                        z1 = PDX_DIV_MXF32(x0, y1);

                        /* Flooring the result in z0 */
                        z0 = PDX_FIFLOOR_MXF32(z0);
                        /* Flooring the result in z1 */
                        z1 = PDX_FIFLOOR_MXF32(z1);

                        /* store the output */
                        PDX_SA_MXF32_IP(z0, az0, p_z0);
                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }
                    /* Remaining iterations of inner loop */
                    PDX_LAV_MXF32_XP(x0, ax0, p_x0, m);

                    PDX_LAV_MXF32_XP(y0, ay0, p_y0, m);
                    PDX_LAV_MXF32_XP(y1, ay1, p_y1, m);

                    z0 = PDX_DIV_MXF32(x0, y0);
                    z1 = PDX_DIV_MXF32(x0, y1);

                    z0 = PDX_FIFLOOR_MXF32(z0);
                    z1 = PDX_FIFLOOR_MXF32(z1);

                    PDX_SAV_MXF32_XP(z0, az0, p_z0, m);
                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);

                    /* Input1 Pointer updates to base address
                     * as input1 is broadcasted
                     */
                    p_x0 = (const xb_vecMxf32*) p_inp1;
                    ax0 = PDX_LA_MXF32_PP(p_x0);
                }
                PDX_SAPOS_MXF32_FP(az0, p_z0);

                /* Loop through remaining iterations of outer loop */
                if ((out_lc % CONST_TWO) != 0)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        PDX_LA_MXF32_IP(x0, ax0, p_x0);

                        PDX_LA_MXF32_IP(y1, ay1, p_y1);

                        z1 = PDX_DIV_MXF32(x0, y1);
                        z1 = PDX_FIFLOOR_MXF32(z1);

                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }

                    /* Remaining iterations */
                    PDX_LAV_MXF32_XP(x0, ax0, p_x0, m);

                    PDX_LAV_MXF32_XP(y1, ay1, p_y1, m);

                    z1 = PDX_DIV_MXF32(x0, y1);
                    z1 = PDX_FIFLOOR_MXF32(z1);

                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);
                }
                PDX_SAPOS_MXF32_FP(az1, p_z1);
        }
    }
    /* If the second from last dim of input2 itself is broadcastable */
    else
    {
#ifndef ENABLE_HIGH_PRECISION
    /* Variable for 1/y0 */
    xb_vecMxf32 one_over_y0;
    /* vector variable for CONST_ONE */
    xb_vecMxf32 one = CONST_ONE;
#endif
        switch (mode)
        {
            case 0:  /* Normal div output */
                /* Unroll the loop by x2 for SIMD */
                for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        /* Load the 4 elements from input1 base address */
                        PDX_LA_MXF32_IP(x0, ax0, p_x0);
                        /* Load the 4 elements from input1 Middle address */
                        PDX_LA_MXF32_IP(x1, ax1, p_x1);

                        /* Load the 4 elements from input2 */
                        PDX_LA_MXF32_IP(y0, ay0, p_y0);

#ifdef ENABLE_HIGH_PRECISION
                        z0 = PDX_DIV_MXF32(x0, y0);
                        z1 = PDX_DIV_MXF32(x1, y0);
#else
                        /* Getting 1/y0 */
                        one_over_y0 = PDX_DIV_MXF32(one, y0);
                        /* Multiplying x0 and 1/y0 */
                        z0 = PDX_MUL_MXF32(x0, one_over_y0);
                        /* Multiplying x1 and 1/y0 */
                        z1 = PDX_MUL_MXF32(x1, one_over_y0);
#endif
                        /* Store the output */
                        PDX_SA_MXF32_IP(z0, az0, p_z0);
                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }

                    /* Remaining iterations of inner loop */
                    PDX_LAV_MXF32_XP(x0, ax0, p_x0, m);
                    PDX_LAV_MXF32_XP(x1, ax1, p_x1, m);

                    PDX_LAV_MXF32_XP(y0, ay0, p_y0, m);

#ifdef ENABLE_HIGH_PRECISION
                    z0 = PDX_DIV_MXF32(x0, y0);
                    z1 = PDX_DIV_MXF32(x1, y0);
#else
                    one_over_y0 = PDX_DIV_MXF32(one, y0);
                    z0 = PDX_MUL_MXF32(x0, one_over_y0);
                    z1 = PDX_MUL_MXF32(x1, one_over_y0);
#endif

                    PDX_SAV_MXF32_XP(z0, az0, p_z0, m);
                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);

                    /* Input2 Pointer updates to base address
                     * as input2 is broadcasted
                     */
                    p_y0 = (const xb_vecMxf32*) p_inp2;
                    ay0 = PDX_LA_MXF32_PP(p_y0);
                }
                PDX_SAPOS_MXF32_FP(az0, p_z0);

                /* Loop through remaining iterations of outer loop */
                if ((out_lc % CONST_TWO) != 0)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        PDX_LA_MXF32_IP(y0, ay0, p_y0);

                        PDX_LA_MXF32_IP(x1, ax1, p_x1);

                        z1 = PDX_DIV_MXF32(x1, y0);

                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }

                    /* Remaining iterations */
                    PDX_LAV_MXF32_XP(x1, ax1, p_x1, m);

                    PDX_LAV_MXF32_XP(y0, ay0, p_y0, m);

                    z1 = PDX_DIV_MXF32(x1, y0);

                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);
                }
                PDX_SAPOS_MXF32_FP(az1, p_z1);
                break;
            case CONST_ONE:  /* Truncated div output */
                /* Unroll the loop by x2 for SIMD */
                for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        /* Load the 4 elements from input2 */
                        PDX_LA_MXF32_IP(y0, ay0, p_y0);

                        /* Load the 4 elements from input1 base address */
                        PDX_LA_MXF32_IP(x0, ax0, p_x0);
                        /* Load the 4 elements from input1 Middle address */
                        PDX_LA_MXF32_IP(x1, ax1, p_x1);

#ifdef ENABLE_HIGH_PRECISION
                        z0 = PDX_DIV_MXF32(x0, y0);
                        z1 = PDX_DIV_MXF32(x1, y0);
#else
                        /* Getting 1/y0 */
                        one_over_y0 = PDX_DIV_MXF32(one, y0);
                        /* Multiplying x0 and 1/y0 */
                        z0 = PDX_MUL_MXF32(x0, one_over_y0);
                        /* Multiplying x1 and 1/y0 */
                        z1 = PDX_MUL_MXF32(x1, one_over_y0);
#endif
                        /* Truncating the result */
                        z0 = PDX_FITRUNC_MXF32(z0);

                        /* Truncating the result */
                        z1 = PDX_FITRUNC_MXF32(z1);

                        /* Store the output */
                        PDX_SA_MXF32_IP(z0, az0, p_z0);
                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }

                    /* Remaining iterations of inner loop */
                    PDX_LAV_MXF32_XP(y0, ay0, p_y0, m);

                    PDX_LAV_MXF32_XP(x0, ax0, p_x0, m);
                    PDX_LAV_MXF32_XP(x1, ax1, p_x1, m);
#ifdef ENABLE_HIGH_PRECISION
                    z0 = PDX_DIV_MXF32(x0, y0);
                    z1 = PDX_DIV_MXF32(x1, y0);
#else
                    one_over_y0 = PDX_DIV_MXF32(one, y0);
                    z0 = PDX_MUL_MXF32(x0, one_over_y0);
                    z1 = PDX_MUL_MXF32(x1, one_over_y0);
#endif
                    z0 = PDX_FITRUNC_MXF32(z0);
                    z1 = PDX_FITRUNC_MXF32(z1);

                    PDX_SAV_MXF32_XP(z0, az0, p_z0, m);
                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);

                    /* Input2 Pointer updates to base address
                     * as input2 is broadcasted
                     */
                    p_y0 = (const xb_vecMxf32*) p_inp2;
                    ay0 = PDX_LA_MXF32_PP(p_y0);
                }
                PDX_SAPOS_MXF32_FP(az0, p_z0);

                /* Loop through remaining iterations of outer loop */
                if ((out_lc % CONST_TWO) != 0)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        PDX_LA_MXF32_IP(x1, ax1, p_x1);

                        PDX_LA_MXF32_IP(y0, ay0, p_y0);

                        z1 = PDX_DIV_MXF32(x1, y0);
                        z1 = PDX_FITRUNC_MXF32(z1);

                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }

                    /* Remaining iterations */
                    PDX_LAV_MXF32_XP(y0, ay0, p_y0, m);

                    PDX_LAV_MXF32_XP(x1, ax1, p_x1, m);

                    z1 = PDX_DIV_MXF32(x1, y0);
                    z1 = PDX_FITRUNC_MXF32(z1);

                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);
                }
                PDX_SAPOS_MXF32_FP(az1, p_z1);
                break;
            case CONST_TWO:  /* Floored div output */
                /* Unroll the loop by x2 for SIMD */
                for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        /* Load the 4 elements from input2 */
                        PDX_LA_MXF32_IP(y0, ay0, p_y0);

                        /* Load the 4 elements from input1 base address */
                        PDX_LA_MXF32_IP(x0, ax0, p_x0);
                        /* Load the 4 elements from input1 Middle address */
                        PDX_LA_MXF32_IP(x1, ax1, p_x1);

#ifdef ENABLE_HIGH_PRECISION
                        z0 = PDX_DIV_MXF32(x0, y0);
                        z1 = PDX_DIV_MXF32(x1, y0);
#else
                        /* Getting 1/y0 */
                        one_over_y0 = PDX_DIV_MXF32(one, y0);
                        /* Multiplying x0 and 1/y0 */
                        z0 = PDX_MUL_MXF32(x0, one_over_y0);
                        /* Multiplying x1 and 1/y0 */
                        z1 = PDX_MUL_MXF32(x1, one_over_y0);
#endif
                        /* Flooring the ouputs */
                        z0 = PDX_FIFLOOR_MXF32(z0);
                        z1 = PDX_FIFLOOR_MXF32(z1);

                        /* Store the output */
                        PDX_SA_MXF32_IP(z0, az0, p_z0);
                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }
                    /* Remaining iterations of inner loop */
                    PDX_LAV_MXF32_XP(y0, ay0, p_y0, m);

                    PDX_LAV_MXF32_XP(x0, ax0, p_x0, m);
                    PDX_LAV_MXF32_XP(x1, ax1, p_x1, m);
#ifdef ENABLE_HIGH_PRECISION
                    z0 = PDX_DIV_MXF32(x0, y0);
                    z1 = PDX_DIV_MXF32(x1, y0);
#else
                    /* Getting 1/y0 */
                    one_over_y0 = PDX_DIV_MXF32(one, y0);
                    /* Multiplying x0 and 1/y0 */
                    z0 = PDX_MUL_MXF32(x0, one_over_y0);
                    /* Multiplying x1 and 1/y0 */
                    z1 = PDX_MUL_MXF32(x1, one_over_y0);
#endif
                    z0 = PDX_FIFLOOR_MXF32(z0);
                    z1 = PDX_FIFLOOR_MXF32(z1);

                    PDX_SAV_MXF32_XP(z0, az0, p_z0, m);
                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);

                    /* Input2 Pointer updates to base address
                     * as input2 is broadcasted
                     */
                    p_y0 = (const xb_vecMxf32*) p_inp2;
                    ay0 = PDX_LA_MXF32_PP(p_y0);
                }
                PDX_SAPOS_MXF32_FP(az0, p_z0);

                /* Loop through remaining iterations of outer loop */
                if ((out_lc % CONST_TWO) != 0)
                {
                    /* Unroll the loop by x4 for SIMD */
                    for (j = 0; j < num_simd4_ops; j++)
                    {
                        PDX_LA_MXF32_IP(y0, ay0, p_y0);

                        PDX_LA_MXF32_IP(x1, ax1, p_x1);

                        z1 = PDX_DIV_MXF32(x1, y0);
                        z1 = PDX_FIFLOOR_MXF32(z1);

                        PDX_SA_MXF32_IP(z1, az1, p_z1);
                    }

                    /* Remaining iterations */
                    PDX_LAV_MXF32_XP(y0, ay0, p_y0, m);

                    PDX_LAV_MXF32_XP(x1, ax1, p_x1, m);

                    z1 = PDX_DIV_MXF32(x1, y0);
                    z1 = PDX_FIFLOOR_MXF32(z1);

                    PDX_SAV_MXF32_XP(z1, az1, p_z1, m);
                }
                PDX_SAPOS_MXF32_FP(az1, p_z1);
        }
    }

}

static inline void internal_elm_div_broadcast_1D_scalar_f32xf32_f32(
        FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp1,
        const FLOAT32 *__restrict__ p_inp2,
        WORD32 num_elm,
        const WORD32 *__restrict__ input1_shapes,
        const WORD32 inp1_const,
        const WORD32 inp2_const,
        const WORD32 mode)
{
    /* Variables for output, input1 and input2 */
    xb_vecMxf32 z, x, y;
    valign ax, ay, az;

    /* Pointer for input1 */
    const xb_vecMxf32 *__restrict__ p_x = (xb_vecMxf32*) p_inp1;
    /* Priming for input1 load */
    ax = PDX_LA_MXF32_PP(p_x);

    /* Pointer for input2 */
    const xb_vecMxf32 *__restrict__ p_y = (xb_vecMxf32*) p_inp2;
    /* Priming for input2 load */
    ay = PDX_LA_MXF32_PP(p_y);

    /* Pointer for output */
    xb_vecMxf32 *__restrict__ p_z = (xb_vecMxf32*) p_out;
    /* Priming for output store */
    az = PDX_Z_ALIGN();

    WORD32 num_simd4_ops = num_elm >> LOG2_PDX_M;
    WORD32 m = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;

    WORD32 i;

    /* Broadcast for input1 when 4th dim is 1 or is constant */
    if ((input1_shapes[4] == CONST_ONE && inp2_const != CONST_ONE)
            || (inp1_const == CONST_ONE))
    {
        x = PDX_LSR_F32_I(p_inp1, 0);
        switch (mode)
        {
            case 0:  /* Normal div output */
                /* Unroll the loop by x4 for SIMD */
                for (i = 0; i < num_simd4_ops; i++)
                {
                    /* Load the 4 elements from input2 */
                    PDX_LA_MXF32_IP(y, ay, p_y);
                    /* Calculating x/y */
                    z = PDX_DIV_MXF32(x, y);
                    /* Store the result */
                    PDX_SA_MXF32_IP(z, az, p_z);
                }
                break;
            case CONST_ONE:  /* Truncated div output */
                /* Unroll the loop by x4 for SIMD */
                for (i = 0; i < num_simd4_ops; i++)
                {
                    /* Load the 4 elements from input2 */
                    PDX_LA_MXF32_IP(y, ay, p_y);
                    /* Calculating x/y */
                    z = PDX_DIV_MXF32(x, y);
                    /* Truncating the output */
                    z = PDX_FITRUNC_MXF32(z);
                    /* Store the output */
                    PDX_SA_MXF32_IP(z, az, p_z);
                }
                break;
            case CONST_TWO:  /* Floored div output */
                /* Unroll the loop by x4 for SIMD */
                for (i = 0; i < num_simd4_ops; i++)
                {
                    /* Load the 4 elements from input2 */
                    PDX_LA_MXF32_IP(y, ay, p_y);
                    /* Calculating x/y */
                    z = PDX_DIV_MXF32(x, y);
                    /* Getting floored output */
                    z = PDX_FIFLOOR_MXF32(z);
                    /* Store the output */
                    PDX_SA_MXF32_IP(z, az, p_z);
                }
        }
        PDX_LAV_MXF32_XP(y, ay, p_y, m);
        z = PDX_DIV_MXF32(x, y);
        switch (mode)
        {
            case CONST_ONE:  /* Truncated div output */
                z = PDX_FITRUNC_MXF32(z);
                break;
            case CONST_TWO:  /* Floored div output */
                z = PDX_FIFLOOR_MXF32(z);
        }
        PDX_SAV_MXF32_XP(z, az, p_z, m);
        PDX_SAPOS_MXF32_FP(az, p_z);
    }
    /* Broadcast for input2 when 4th dim is 1 or is constant */
    else
    {
        y = PDX_LSR_F32_I(p_inp2, 0);
#ifndef ENABLE_HIGH_PRECISION
        xb_vecMxf32 one = CONST_ONE;
        xb_vecMxf32 one_over_y = PDX_DIV_MXF32(one, y);
#endif
        switch (mode)
        {
            case 0:  /* Normal div output */
                /* Unroll the loop by x4 for SIMD */
                for (i = 0; i < num_simd4_ops; i++)
                {
                    /* Load the 4 elements from input1 */
                    PDX_LA_MXF32_IP(x, ax, p_x);
#ifdef ENABLE_HIGH_PRECISION
                    z = PDX_DIV_MXF32(x, y);
#else
                    /* Multiplying x and 1/y */
                    z = PDX_MUL_MXF32(x, one_over_y);
#endif
                    /* Store the output */
                    PDX_SA_MXF32_IP(z, az, p_z);
                }
                break;
            case CONST_ONE:  /* Truncated div output */
                /* Unroll the loop by x4 for SIMD */
                for (i = 0; i < num_simd4_ops; i++)
                {
                    /* Load the 4 elements from input1 */
                    PDX_LA_MXF32_IP(x, ax, p_x);
#ifdef ENABLE_HIGH_PRECISION
                    z = PDX_DIV_MXF32(x, y);
#else
                    /* Multiplying x and 1/y */
                    z = PDX_MUL_MXF32(x, one_over_y);
#endif
                    /* Truncating the output */
                    z = PDX_FITRUNC_MXF32(z);
                    /* Store the output */
                    PDX_SA_MXF32_IP(z, az, p_z);
                }
                break;
            case CONST_TWO:  /* Floored div output */
                for (i = 0; i < num_simd4_ops; i++)
                {
                    /* Load the 4 elements from input1 */
                    PDX_LA_MXF32_IP(x, ax, p_x);
#ifdef ENABLE_HIGH_PRECISION
                    z = PDX_DIV_MXF32(x, y);
#else
                    /* Multiplying x and 1/y */
                    z = PDX_MUL_MXF32(x, one_over_y);
#endif
                    /* Getting floored output */
                    z = PDX_FIFLOOR_MXF32(z);
                    /* Store the output */
                    PDX_SA_MXF32_IP(z, az, p_z);
                }
        }
        PDX_LAV_MXF32_XP(x, ax, p_x, m);
#ifdef ENABLE_HIGH_PRECISION
        z = PDX_DIV_MXF32(x, y);
#else
        z = PDX_MUL_MXF32(x, one_over_y);
#endif
        switch (mode)
        {
            case CONST_ONE:  /* Truncated div output */
                z = PDX_FITRUNC_MXF32(z);
                break;
            case CONST_TWO:  /* Floored div output */
                z = PDX_FIFLOOR_MXF32(z);
        }
        PDX_SAV_MXF32_XP(z, az, p_z, m);
        PDX_SAPOS_MXF32_FP(az, p_z);
    }
}

WORD32 xa_nn_elm_div_broadcast_5D_f32xf32_f32(FLOAT32 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        const FLOAT32 *__restrict__ p_inp1,
        const WORD32 *const p_inp1_shape,
        const FLOAT32 *__restrict__ p_inp2,
        const WORD32 *const p_inp2_shape,
        WORD32 mode,
        WORD32 num_inp_dims)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), UNSUPPORTED_PARAM);

    /* Invalid input checks */
    /* mode should be either 0 or 1 or 2 */
    XA_NNLIB_ARG_CHK_COND((mode < 0) || (mode > CONST_TWO), UNSUPPORTED_PARAM);
    /* num_inp_dims should be greater than 0 and less than or equal to 5 */
    XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);

    WORD32 i;

    /* Shape checks */
    /* Shapes should be greater than zero */
    for(i = 0; i < num_inp_dims; i++)
    {
        XA_NNLIB_ARG_CHK_COND((p_out_shape[i] <= 0), UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_COND((p_inp1_shape[i] <= 0), UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_COND((p_inp2_shape[i] <= 0), UNSUPPORTED_PARAM);
    }

    /* 5D shapes initialization */
    WORD32 p_5d_out_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE};
    WORD32 p_5d_inp1_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE};
    WORD32 p_5d_inp2_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE};

    shapes_convert_5D(p_5d_out_shape, p_5d_inp1_shape, p_5d_inp2_shape,
                     p_out_shape, p_inp1_shape, p_inp2_shape, num_inp_dims);

    /* Check shapes for broadcast compatibility */
    WORD32 error = 0;
    error = check_shapes(p_5d_inp1_shape, p_5d_inp2_shape, p_5d_out_shape);
    if (error)
    {
        return UNSUPPORTED_PARAM;
    }

    /* Strides calculation */
    WORD32 inp1_strides[MAX_DIMS], inp2_strides[MAX_DIMS];
    strides_calculation(p_5d_inp1_shape, p_5d_inp2_shape, inp1_strides,
            inp2_strides);

    /* Check for broadcast need */
    WORD32 need_broadcast = 0;
    WORD32 inp1_const = CONST_ONE, inp2_const = CONST_ONE;
    for (i = 0; i < MAX_DIMS; i++)
    {
        if (p_5d_inp1_shape[i] != p_5d_inp2_shape[i])
        {
            if (p_5d_inp1_shape[i] == CONST_ONE)
            {
                inp1_strides[i] = 0;
            }
            else
            {
                inp2_strides[i] = 0;
            }
            need_broadcast = CONST_ONE;
        }

        if (p_5d_inp1_shape[i] != CONST_ONE)
            inp1_const &= 0;
        if (p_5d_inp2_shape[i] != CONST_ONE)
            inp2_const &= 0;
    }

    const FLOAT32 *__restrict__ p_inp1_base = p_inp1;
    const FLOAT32 *__restrict__ p_inp2_base = p_inp2;
    FLOAT32 *p_out_base = p_out;

    WORD32 itr0, itr1, itr2, itr3;

    /* If broadcast is not needed */
    if (need_broadcast == 0)
    {
        xa_nn_elm_div_f32xf32_f32(p_out,
                p_inp1,
                p_inp2,
                mode,
                p_5d_out_shape[0] * inp1_strides[0]);
    }
    /* If broadcast is needed */
    /* If one of input is scalar */
    else if (inp1_const == CONST_ONE || inp2_const == CONST_ONE)
    {
        internal_elm_div_broadcast_1D_scalar_f32xf32_f32(p_out_base,
                p_inp1_base,
                p_inp2_base,
                p_5d_out_shape[0] * p_5d_out_shape[1] * p_5d_out_shape[2]
                        * p_5d_out_shape[3] * p_5d_out_shape[4],
                p_5d_inp1_shape,
                inp1_const,
                inp2_const,
                mode);
    }
    /* Check if 4th dim in both inputs is the same */
    else if (inp1_strides[4] == inp2_strides[4])
    {
        WORD32 in_lc, out_lc;
        /* Check if 3rd dim needs to be broadcasted */
        if (inp1_strides[3] == 0 || inp2_strides[3] == 0)
        {
            /* Repeat the 4th dimension as the
             * 3rd dimension needs to be broadcasted
             */
            in_lc = p_5d_out_shape[4];
            out_lc = p_5d_out_shape[3];
            for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
            {
                const FLOAT32 *__restrict__ p_inp1_itr0 = p_inp1_base;
                const FLOAT32 *__restrict__ p_inp2_itr0 = p_inp2_base;
                for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
                {
                    const FLOAT32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                    const FLOAT32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                    for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                    {
                        internal_elm_div_broadcast_2D_f32xf32_f32(
                                p_out_base,
                                p_inp1_itr1,
                                p_inp2_itr1,
                                out_lc,
                                in_lc,
                                p_5d_inp1_shape,
                                mode);
                        p_out_base += in_lc * out_lc;
                        p_inp1_itr1 += inp1_strides[2];
                        p_inp2_itr1 += inp2_strides[2];
                    }
                    p_inp1_itr0 += inp1_strides[1];
                    p_inp2_itr0 += inp2_strides[1];
                }
                p_inp1_base += inp1_strides[0];
                p_inp2_base += inp2_strides[0];
            }
        }
        else
        {
            /* 3rd and 4th dimensions need not be broadcasted. The lower
             * dimension broadcasting (0th, 1st, 2nd) will be taken care
             * while calculating the input addresses
             */
            in_lc = p_5d_out_shape[3] * p_5d_out_shape[4];
            for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
            {
                const FLOAT32 *__restrict__ p_inp1_itr0 = p_inp1_base;
                const FLOAT32 *__restrict__ p_inp2_itr0 = p_inp2_base;
                for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
                {
                    const FLOAT32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                    const FLOAT32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                    for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                    {
                        xa_nn_elm_div_f32xf32_f32(p_out_base,
                                p_inp1_itr1,
                                p_inp2_itr1,
                                mode,
                                in_lc);
                        p_out_base += in_lc;
                        p_inp1_itr1 += inp1_strides[2];
                        p_inp2_itr1 += inp2_strides[2];
                    }
                    p_inp1_itr0 += inp1_strides[1];
                    p_inp2_itr0 += inp2_strides[1];
                }
                p_inp1_base += inp1_strides[0];
                p_inp2_base += inp2_strides[0];
            }
        }
    }
    /* If the last dim itself is broadcastable */
    else
    {
        for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
        {
            const FLOAT32 *__restrict__ p_inp1_itr0 = p_inp1_base;
            const FLOAT32 *__restrict__ p_inp2_itr0 = p_inp2_base;
            for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
            {
                const FLOAT32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                const FLOAT32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                {
                    const FLOAT32 *__restrict__ p_inp1_itr2 = p_inp1_itr1;
                    const FLOAT32 *__restrict__ p_inp2_itr2 = p_inp2_itr1;
                    for (itr3 = 0; itr3 < p_5d_out_shape[3]; itr3++)
                    {
                        internal_elm_div_broadcast_1D_scalar_f32xf32_f32(
                                p_out_base,
                                p_inp1_itr2,
                                p_inp2_itr2,
                                p_5d_out_shape[4],
                                p_5d_inp1_shape,
                                inp1_const,
                                inp2_const,
                                mode);
                        p_out_base += p_5d_out_shape[4];
                        p_inp1_itr2 += inp1_strides[3];
                        p_inp2_itr2 += inp2_strides[3];
                    }
                    p_inp1_itr1 += inp1_strides[2];
                    p_inp2_itr1 += inp2_strides[2];
                }
                p_inp1_itr0 += inp1_strides[1];
                p_inp2_itr0 += inp2_strides[1];
            }
            p_inp1_base += inp1_strides[0];
            p_inp2_base += inp2_strides[0];
        }
    }

    return 0;
}


