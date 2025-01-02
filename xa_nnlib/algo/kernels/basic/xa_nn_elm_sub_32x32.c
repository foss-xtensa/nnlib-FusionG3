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

WORD32 xa_nn_elm_sub_32x32_32(WORD32 *p_out,
        const WORD32 *p_inp1,
        const WORD32 *p_inp2,
        WORD32 alpha,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, SIZE_OF_INT, UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    WORD32 n, m;

    xb_vecMx32 x0, y0, z0;
    valign ax, ay, az;

    const xb_vecMx32 *__restrict__ p_x = (const xb_vecMx32*) p_inp1;
    const xb_vecMx32 *__restrict__ p_y = (const xb_vecMx32*) p_inp2;
    xb_vecMx32 *__restrict__ p_z = (xb_vecMx32*) p_out;

    /* Move from scalar register to 4-way 32-bit vec register with replicate */
    xb_vecMx32 vec_alpha = PDX_MOVR32_A32(alpha);

    /* Align load priming */
    ax = PDX_LA_MX32_PP(p_x);
    ay = PDX_LA_MX32_PP(p_y);

    /* Zeroing align register */
    az = PDX_Z_ALIGN();

    /* Unroll the loop by x4 for SIMD */
    for (n = 0; n < (num_elm >> LOG2_PDX_M); n++)
    {
        PDX_LA_MX32_IP(x0, ax, p_x);
        PDX_LA_MX32_IP(y0, ay, p_y);

        /* 4-way (Input2 * alpha) */
        y0 = PDX_MUL_MX32(y0, vec_alpha);

        /* 4-way Sub (Input1 - alpha*Input2) */
        z0 = PDX_SUB_MX32(x0, y0);
        PDX_SA_MX32_IP(z0, az, p_z);
    }
    /* Remaining iterations */
    m = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_INT;
    PDX_LAV_MX32_XP(x0, ax, p_x, m);
    PDX_LAV_MX32_XP(y0, ay, p_y, m);

    y0 = PDX_MUL_MX32(y0, vec_alpha);
    z0 = PDX_SUB_MX32(x0, y0);

    PDX_SAV_MX32_XP(z0, az, p_z, m);
    PDX_SAPOS_MX32_FP(az, p_z);

    return 0;
} /* xa_nn_elm_sub_32x32_32() */

WORD32 xa_nn_elm_sub_scalar_32x32_32(WORD32 *p_out,
        const WORD32 *p_inp1,
        const WORD32 inp2,
        WORD32 alpha,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, SIZE_OF_INT, UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    WORD32 n, m;

    xb_vecMx32 x0, y0, z0;
    valign ax, az;
    const xb_vecMx32 *__restrict__ p_x = (const xb_vecMx32*) p_inp1;
    xb_vecMx32 *__restrict__ p_z = (xb_vecMx32*) p_out;
    WORD32 in2 = alpha * inp2;

    /* Vectorize input2 for SIMD operation */
    y0 = in2;

    /* Align load priming */
    ax = PDX_LA_MX32_PP(p_x);

    /* Zeroing align register */
    az = PDX_Z_ALIGN();

    /* Unroll the loop by x4 for SIMD */
    for (n = 0; n < (num_elm >> LOG2_PDX_M); n++)
    {
        PDX_LA_MX32_IP(x0, ax, p_x);
        z0 = PDX_SUB_MX32(x0, y0);
        PDX_SA_MX32_IP(z0, az, p_z);
    }

    /* Remaining iterations */
    m = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_INT;
    PDX_LAV_MX32_XP(x0, ax, p_x, m);

    z0 = PDX_SUB_MX32(x0, y0);

    PDX_SAV_MX32_XP(z0, az, p_z, m);
    PDX_SAPOS_MX32_FP(az, p_z);

    return 0;
} /* xa_nn_elm_sub_scalar_32x32_32 */

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

static inline void strides_calculation(const WORD32 *const inp1_shape,
        const WORD32 *const inp2_shape,
        WORD32 *const inp1_strides,
        WORD32 *const inp2_strides)
{

    WORD32 i;
    inp1_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    inp2_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    for (i = MAX_DIMS - CONST_TWO; i >= 0; i--)
    {
        inp1_strides[i] = inp1_strides[i + CONST_ONE]
                * inp1_shape[i + CONST_ONE];
        inp2_strides[i] = inp2_strides[i + CONST_ONE]
                * inp2_shape[i + CONST_ONE];
    }
}

static inline void internal_elm_sub_broadcast_2D_32x32_32(
        WORD32 *__restrict__ p_out,
        const WORD32 *__restrict__ p_inp1,
        const WORD32 *__restrict__ p_inp2,
        WORD32 out_lc,
        WORD32 in_lc,
        const WORD32 *input1_shapes,
        const WORD32 *input2_shapes,
        WORD32 alpha)
{

    WORD32 n, m, i;

    /* Declaration of SIMD variables */
    xb_vecMx32 x0, x1, y0, y1, z0, z1;

    xb_vecMx32 vec_alpha = PDX_MOVR32_A32(alpha);

    valign ax0, ax1, ay0, ay1, az0, az1;

    const WORD32 *p_x_baseptr = &p_inp1[0];

    /* base address calculation for input1 */
    const xb_vecMx32 *__restrict__ p_x0 = (const xb_vecMx32*) &p_x_baseptr[0];
    /* Middle address calculation for input1*/
    const xb_vecMx32 *__restrict__ p_x1 = (const xb_vecMx32*) (&p_x_baseptr[0]
            + ((out_lc / CONST_TWO) * in_lc));

    /* priming */
    ax0 = PDX_LA_MX32_PP(p_x0);
    ax1 = PDX_LA_MX32_PP(p_x1);

    const WORD32 *p_y_baseptr = &p_inp2[0];

    /* base address calculation for input2 */
    const xb_vecMx32 *__restrict__ p_y0 = (const xb_vecMx32*) &p_y_baseptr[0];
    /* Middle address calculation for input2 */
    const xb_vecMx32 *__restrict__ p_y1 = (const xb_vecMx32*) (&p_y_baseptr[0]
            + ((out_lc / CONST_TWO) * in_lc));

    /* priming */
    ay0 = PDX_LA_MX32_PP(p_y0);
    ay1 = PDX_LA_MX32_PP(p_y1);

    WORD32 *p_z_baseptr = &p_out[0];

    /* base address calculation for output */
    xb_vecMx32 *__restrict__ p_z0 = (xb_vecMx32*) &p_z_baseptr[0];
    /* Middle address calculation for output */
    xb_vecMx32 *__restrict__ p_z1 = (xb_vecMx32*) (&p_z_baseptr[0]
            + ((out_lc / CONST_TWO) * in_lc));

    /* priming */
    az0 = PDX_Z_ALIGN();
    az1 = PDX_Z_ALIGN();

    m = (in_lc & (PDX_M - CONST_ONE)) * SIZE_OF_INT;

    /* if the last dim of input1 itself is broadcastable (input1_shapes[3] is 1) */
    if (input1_shapes[3] == CONST_ONE)
    {
        for (i = 0; i < out_lc - CONST_ONE; i += CONST_TWO)
        {
            /* unroll the loop by x4 for SIMD */
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                /* load the 4 elements from input1 */
                PDX_LA_MX32_IP(x0, ax0, p_x0);

                /* load the 4 elements from input2 base address */
                PDX_LA_MX32_IP(y0, ay0, p_y0);
                /* load the 4 elements from input2 Middle address */
                PDX_LA_MX32_IP(y1, ay1, p_y1);

                /* Subtraction of x0 and y0 */
                y0 = PDX_MUL_MX32(y0, vec_alpha);
                z0 = PDX_SUB_MX32(x0, y0);

                /* Subtraction of x0 and y1 */
                y1 = PDX_MUL_MX32(y1, vec_alpha);
                z1 = PDX_SUB_MX32(x0, y1);

                /* Store the output */
                PDX_SA_MX32_IP(z0, az0, p_z0);
                PDX_SA_MX32_IP(z1, az1, p_z1);
            }
            /* Remaining iterations of inner loop */
            PDX_LAV_MX32_XP(x0, ax0, p_x0, m);
            PDX_LAV_MX32_XP(y0, ay0, p_y0, m);
            PDX_LAV_MX32_XP(y1, ay1, p_y1, m);

            y0 = PDX_MUL_MX32(y0, vec_alpha);
            z0 = PDX_SUB_MX32(x0, y0);

            y1 = PDX_MUL_MX32(y1, vec_alpha);
            z1 = PDX_SUB_MX32(x0, y1);

            PDX_SAV_MX32_XP(z0, az0, p_z0, m);
            PDX_SAV_MX32_XP(z1, az1, p_z1, m);
            PDX_SAPOS_MX32_FP(az0, p_z0);
            PDX_SAPOS_MX32_FP(az1, p_z1);

            /* Input1 Pointer updates to base address as input1 is broadcasted */
            p_x0 = (const xb_vecMx32*) &p_inp1[0];
            ax0 = PDX_LA_MX32_PP(p_x0);
        }
        /* loop through remaining iterations of outer loop */
        if (out_lc % CONST_TWO != 0)
        {
            /* unroll the loop by x4 for SIMD */
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                PDX_LA_MX32_IP(y1, ay1, p_y1);
                PDX_LA_MX32_IP(x0, ax0, p_x0);

                y1 = PDX_MUL_MX32(y1, vec_alpha);
                z0 = PDX_SUB_MX32(x0, y1);

                PDX_SA_MX32_IP(z0, az1, p_z1);
            }
            /* Remaining iterations */
            PDX_LAV_MX32_XP(y1, ay1, p_y1, m);
            PDX_LAV_MX32_XP(x0, ax0, p_x0, m);

            y1 = PDX_MUL_MX32(y1, vec_alpha);
            z0 = PDX_SUB_MX32(x0, y1);

            PDX_SAV_MX32_XP(z0, az1, p_z1, m);
            PDX_SAPOS_MX32_FP(az1, p_z1);
        }
    }
    /* if the last dim of input2 itself is broadcastable (input2_shapes[3] is 1) */
    else
    {
        /* Unroll the loop by x2 for SIMD */
        for (i = 0; i < out_lc - CONST_ONE; i += CONST_TWO)
        {
            /* unroll the loop by x4 for SIMD */
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                /* load the 4 elements from input2 */
                PDX_LA_MX32_IP(y0, ay0, p_y0);

                /* load the 4 elements from input1 base address */
                PDX_LA_MX32_IP(x0, ax0, p_x0);
                /* load the 4 elements from input1 Middle address */
                PDX_LA_MX32_IP(x1, ax1, p_x1);

                /* Subtraction of x0 and y0 */
                y0 = PDX_MUL_MX32(y0, vec_alpha);
                z0 = PDX_SUB_MX32(x0, y0);

                /* Subtraction of x1 and y0 */
                z1 = PDX_SUB_MX32(x1, y0);

                /* Store the output */
                PDX_SA_MX32_IP(z0, az0, p_z0);
                PDX_SA_MX32_IP(z1, az1, p_z1);
            }
            /* Remaining iterations of inner loop */
            PDX_LAV_MX32_XP(y0, ay0, p_y0, m);
            PDX_LAV_MX32_XP(x0, ax0, p_x0, m);
            PDX_LAV_MX32_XP(x1, ax1, p_x1, m);

            y0 = PDX_MUL_MX32(y0, vec_alpha);
            z0 = PDX_SUB_MX32(x0, y0);

            z1 = PDX_SUB_MX32(x1, y0);

            PDX_SAV_MX32_XP(z0, az0, p_z0, m);
            PDX_SAV_MX32_XP(z1, az1, p_z1, m);
            PDX_SAPOS_MX32_FP(az0, p_z0);
            PDX_SAPOS_MX32_FP(az1, p_z1);

            /* Input2 Pointer updates to base address as input2 is broadcasted */
            p_y0 = (const xb_vecMx32*) &p_inp2[0];
            ay0 = PDX_LA_MX32_PP(p_y0);
        }
        /* loop through remaining iterations of outer loop */
        if (out_lc % CONST_TWO != 0)
        {
            /* unroll the loop by x4 for SIMD */
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                PDX_LA_MX32_IP(y0, ay0, p_y0);
                PDX_LA_MX32_IP(x1, ax1, p_x1);

                y0 = PDX_MUL_MX32(y0, vec_alpha);
                z0 = PDX_SUB_MX32(x1, y0);

                PDX_SA_MX32_IP(z0, az1, p_z1);
            }
            /* Remaining iterations */
            PDX_LAV_MX32_XP(y0, ay0, p_y0, m);
            PDX_LAV_MX32_XP(x0, ax1, p_x1, m);

            y0 = PDX_MUL_MX32(y0, vec_alpha);
            z0 = PDX_SUB_MX32(x0, y0);

            PDX_SAV_MX32_XP(z0, az1, p_z1, m);
            PDX_SAPOS_MX32_FP(az1, p_z1);
        }
    }
}/* internal_elm_sub_broadcast_2D_32x32_32 */

static inline void internal_elm_sub_broadcast_1D_scalar_32x32_32(
        WORD32 *__restrict__ p_out,
        const WORD32 *__restrict__ p_inp1,
        const WORD32 *__restrict__ p_inp2,
        WORD32 num_elm,
        const WORD32 *__restrict__ input1_shapes,
        const WORD32 inp1_const,
        const WORD32 inp2_const,
        const WORD32 alpha)
{
    /* Declaration of SIMD variables */
    xb_vecMx32 i1, i2, y, z, vec_alpha = alpha;
    /* Initialization of SIMD pointers */
    xb_vecMx32 *__restrict__ p_i1 = (xb_vecMx32*) p_inp1;
    xb_vecMx32 *__restrict__ p_i2 = (xb_vecMx32*) p_inp2;
    xb_vecMx32 *__restrict__ p_o = (xb_vecMx32*) p_out;

    /* Declaration of valign registers */
    valign ax, az;
    az = PDX_Z_ALIGN();

    WORD32 i, m;
    m = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_INT;


    if (((input1_shapes[4] == CONST_ONE) && (inp2_const != CONST_ONE))
            || (inp1_const == CONST_ONE))
    {
        i1 = PDX_LSR_32_I(p_inp1, 0);
        ax = PDX_LA_MX32_PP(p_i2);

        /* unroll the loop by x4 for SIMD */
        for (i = 0; i < (num_elm >> LOG2_PDX_M); i++)
        {
            PDX_LA_MX32_IP(i2, ax, p_i2);
            y = PDX_MUL_MX32(i2, vec_alpha);
            z = PDX_SUB_MX32(i1, y);
            PDX_SA_MX32_IP(z, az, p_o);
        }
        /* Remaining iterations */
        PDX_LAV_MX32_XP(i2, ax, p_i2, m);
        y = PDX_MUL_MX32(i2, vec_alpha);
        z = PDX_SUB_MX32(i1, y);
        PDX_SAV_MX32_XP(z, az, p_o, m);
    }
    else
    {
        i2 = PDX_LSR_32_I(p_inp2, 0);
        y = PDX_MUL_MX32(i2, vec_alpha);
        ax = PDX_LA_MX32_PP(p_i1);

        /* unroll the loop by x4 for SIMD */
        for (i = 0; i < (num_elm >> LOG2_PDX_M); i++)
        {
            PDX_LA_MX32_IP(i1, ax, p_i1);
            z = PDX_SUB_MX32(i1, y);
            PDX_SA_MX32_IP(z, az, p_o);
        }
        /* Remaining iterations */
        PDX_LAV_MX32_XP(i1, ax, p_i1, m);
        z = PDX_SUB_MX32(i1, y);
        PDX_SAV_MX32_XP(z, az, p_o, m);
    }
    PDX_SAPOS_MX32_FP(az, p_o);
}/*internal_elm_sub_broadcast_1D_scalar_32x32_32*/

WORD32 xa_nn_elm_sub_broadcast_5D_32x32_32(WORD32 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        const WORD32 *__restrict__ p_inp1,
        const WORD32 *const p_inp1_shape,
        const WORD32 *__restrict__ p_inp2,
        const WORD32 *const p_inp2_shape,
        WORD32 num_inp_dims,
        WORD32 alpha)
{

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);

    /* UNSUPPORTED_PARAM input checks */
    XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);

    WORD32 i;
    for (i = 0; i < num_inp_dims; i++)
    {
        XA_NNLIB_ARG_CHK_COND((p_out_shape[i] <= 0), UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_COND((p_inp1_shape[i] <= 0), UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_COND((p_inp2_shape[i] <= 0), UNSUPPORTED_PARAM);
    }

    /* 5D shapes initialization */
    WORD32 p_5d_out_shape[MAX_DIMS] = { CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE };
    WORD32 p_5d_inp1_shape[MAX_DIMS] = { CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE };
    WORD32 p_5d_inp2_shape[MAX_DIMS] = { CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE };

    shapes_convert_5D(p_5d_out_shape, p_5d_inp1_shape, p_5d_inp2_shape,
            p_out_shape, p_inp1_shape, p_inp2_shape, num_inp_dims);

    /* Check shapes for broadcast compatibility */
    WORD32 error = 0;
    error = check_shapes(p_5d_inp1_shape, p_5d_inp2_shape, p_5d_out_shape);
    if (error)
    {
        return UNSUPPORTED_PARAM;
    }

    /* strides calculation */
    WORD32 inp1_strides[MAX_DIMS], inp2_strides[MAX_DIMS];
    strides_calculation(p_5d_inp1_shape, p_5d_inp2_shape, inp1_strides,
            inp2_strides);

    /* check for broadcast need */
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

        if(p_5d_inp1_shape[i] != CONST_ONE)
             inp1_const &= 0;
        if (p_5d_inp2_shape[i] != CONST_ONE)
            inp2_const &= 0;
    }

    const WORD32 *__restrict__ p_inp1_base = p_inp1;
    const WORD32 *__restrict__ p_inp2_base = p_inp2;
    WORD32 *p_out_base = p_out;
    WORD32 itr0, itr1, itr2, itr3;

    /* if broadcast is not needed */
    if (need_broadcast == 0)
    {
        xa_nn_elm_sub_32x32_32(p_out_base,
                p_inp1_base,
                p_inp2_base,
                alpha,
                p_5d_out_shape[0] * inp1_strides[0]);
    }

    /* if broadcast is needed */
    else if (inp1_const == CONST_ONE || inp2_const == CONST_ONE)
    {
        internal_elm_sub_broadcast_1D_scalar_32x32_32(p_out_base,
                p_inp1_base,
                p_inp2_base,
                p_5d_out_shape[0] * p_5d_out_shape[1] * p_5d_out_shape[2]
                        * p_5d_out_shape[3] * p_5d_out_shape[4],
                p_5d_inp1_shape,
                inp1_const,
                inp2_const,
                alpha);
    }
    /* check if 4th dim in both inputs is the same */
    else if (inp1_strides[4] == inp2_strides[4])
    {
        WORD32 in_lc, out_lc;
        /* check if 3rd dim needs to be broadcasted */
        if (inp1_strides[3] == 0 || inp2_strides[3] == 0)
        {
            /* Repeat the 4th dim as the 3rd dim needs to be broadcasted */
            in_lc = p_5d_out_shape[4];
            out_lc = p_5d_out_shape[3];
            for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
            {
                const WORD32 *__restrict__ p_inp1_itr0 = p_inp1_base;
                const WORD32 *__restrict__ p_inp2_itr0 = p_inp2_base;
                for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
                {
                    const WORD32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                    const WORD32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                    for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                    {
                        internal_elm_sub_broadcast_2D_32x32_32(p_out_base,
                                p_inp1_itr1,
                                p_inp2_itr1,
                                out_lc,
                                in_lc,
                                p_5d_inp1_shape,
                                p_5d_inp2_shape,
                                alpha);
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
             * while calculating the input addresses */
            in_lc = p_5d_out_shape[3] * p_5d_out_shape[4];
            out_lc = CONST_ONE;
            for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
            {
                const WORD32 *__restrict__ p_inp1_itr0 = p_inp1_base;
                const WORD32 *__restrict__ p_inp2_itr0 = p_inp2_base;
                for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
                {
                    const WORD32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                    const WORD32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                    for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                    {
                        xa_nn_elm_sub_32x32_32(p_out_base,
                                p_inp1_itr1,
                                p_inp2_itr1,
                                alpha,
                                in_lc);
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
    }
    else
    {
        /* If the last dim itself is broadcastable */
        for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
        {
            const WORD32 *__restrict__ p_inp1_itr0 = p_inp1_base;
            const WORD32 *__restrict__ p_inp2_itr0 = p_inp2_base;
            for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
            {
                const WORD32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                const WORD32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                {
                    const WORD32 *__restrict__ p_inp1_itr2 = p_inp1_itr1;
                    const WORD32 *__restrict__ p_inp2_itr2 = p_inp2_itr1;
                    for (itr3 = 0; itr3 < p_5d_out_shape[3]; itr3++)
                    {
                        internal_elm_sub_broadcast_1D_scalar_32x32_32(
                                p_out_base,
                                p_inp1_itr2,
                                p_inp2_itr2,
                                p_5d_out_shape[4],
                                p_5d_inp1_shape,
                                inp1_const,
                                inp2_const,
                                alpha);
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
