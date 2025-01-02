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

WORD32 xa_nn_slice(WORD8 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        const WORD8 *__restrict__ p_inp,
        const WORD32 *const p_inp_shape,
        WORD32 num_inp_dims,  // number of dimensions in input
        WORD32 start,
        WORD32 end,
        WORD32 step,
        WORD32 axis,  // dimension along which slicing to be done
        WORD32 elm_size  // number of bytes for each element
        )
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_shape, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, elm_size, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, elm_size, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), UNSUPPORTED_PARAM);

    /* Invalid input checks */
    /* num_inp_dims should be greater than 0 */
    XA_NNLIB_ARG_CHK_COND((num_inp_dims <= 0), UNSUPPORTED_PARAM);
    /* axis should be in range [0, num_inp_dims-1] */
    XA_NNLIB_ARG_CHK_COND(((axis < 0) || (axis >= num_inp_dims)),
            UNSUPPORTED_PARAM);
    /* step should be greater than 0 */
    XA_NNLIB_ARG_CHK_COND((step <= 0), UNSUPPORTED_PARAM);
    /* start should be in range [0, p_inp_shape[axis]-1] */
    XA_NNLIB_ARG_CHK_COND(((start < 0) || (start >= p_inp_shape[axis])),
            UNSUPPORTED_PARAM);
    /* end should be in range [0, p_inp_shape[axis]-1] and end>=start */
    XA_NNLIB_ARG_CHK_COND(
            ((end < 0) || (end >= p_inp_shape[axis]) || (end < start)),
            UNSUPPORTED_PARAM);
    /* elm_size should be either 1 or 2 or 4 */
    XA_NNLIB_ARG_CHK_COND(
            (elm_size <= 0) || (elm_size == CONST_THREE) ||
            (elm_size > CONST_FOUR), UNSUPPORTED_PARAM);

    /* Length of the given dimension in input */
    WORD32 dim_length = p_inp_shape[axis];
    /* Length of given dimension after slicing */
    WORD32 num_values = (end - start) / step + CONST_ONE;

    WORD32 i;

    /* Checking shapes */
    for (i = 0; i < num_inp_dims; i++)
    {
        XA_NNLIB_ARG_CHK_COND((p_inp_shape[i] <= 0), UNSUPPORTED_PARAM);
        /* Output shape at axis should be equal to num_values */
        if (i == axis)
        {
            XA_NNLIB_ARG_CHK_COND((p_out_shape[i] != num_values),
                    UNSUPPORTED_PARAM);
        }
        /* Output shape at other than axis should be equal to input shape */
        else
        {
            XA_NNLIB_ARG_CHK_COND((p_out_shape[i] != p_inp_shape[i]),
                    UNSUPPORTED_PARAM);
        }
    }

    WORD32 leading_dims = CONST_ONE;
    for (i = 0; i < axis; i++)
    {
        leading_dims *= p_inp_shape[i];
    }

    WORD32 trailing_dims = CONST_ONE;
    for (i = axis + CONST_ONE; i < num_inp_dims; i++)
    {
        trailing_dims *= p_inp_shape[i];
    }

    /* Variable to hold 16 8-bit elements */
    xb_vec4Mx8 x;

    valign ax, az;

    /* Pointers for 16 8-bit elements */
    xb_vec4Mx8 *__restrict__ p_x;
    xb_vec4Mx8 *__restrict__ p_z = (xb_vec4Mx8*) p_out;

    /* Number of bytes to be skipped to go to next location
     * after copying elements along the axis.
     */
    WORD32 length_per_step = trailing_dims * elm_size;

    WORD32 num_simd16_ops = length_per_step >> PDX_M;
    WORD32 m = length_per_step & MASK_LOG2_PDX_4M;

    WORD8 *__restrict__ p_temp_x = (WORD8*) p_inp;
    WORD8 *__restrict__ src;

    /* Storing zeroes in az register */
    az = PDX_Z_ALIGN();

    WORD32 j;

    /* When axis is last dim */
    if (axis == num_inp_dims - CONST_ONE)
    {
        /* When step is greater than one, only one element is loaded
         * at once and stored.
         */
        if (step > CONST_ONE)
        {
            for (i = 0; i < leading_dims; i++)
            {
                src = p_temp_x + (i * dim_length + start) * length_per_step;
                for (j = 0; j < num_values; j++)
                {
                    p_x = (xb_vec4Mx8*) src;
                    ax = PDX_LA_4MX8_PP(p_x);
                    PDX_LAV_4MX8_XP(x, ax, p_x, m);
                    PDX_SAV_4MX8_XP(x, az, p_z, m);
                    src += step * length_per_step;
                }
            }
        }
        /* When step is equal to one, multiple elements are loaded
         * at a time and stored.
         */
        else
        {
            num_simd16_ops = (num_values * elm_size) >> PDX_M;
            m = (num_values * elm_size) & MASK_LOG2_PDX_4M;

            for (i = 0; i < leading_dims; i++)
            {
                p_x = (xb_vec4Mx8*) (p_temp_x
                        + ((i * dim_length + start) * elm_size));
                ax = PDX_LA_4MX8_PP(p_x);
                for (j = 0; j < num_simd16_ops; j++)
                {
                    PDX_LA_4MX8_IP(x, ax, p_x);
                    PDX_SA_4MX8_IP(x, az, p_z);
                }
                PDX_LAV_4MX8_XP(x, ax, p_x, m);
                PDX_SAV_4MX8_XP(x, az, p_z, m);
            }
        }
    }
    /* When axis is other than last dim */
    else
    {
        /* Loop iterates over the leading dims */
        for (i = 0; i < leading_dims; i++)
        {
            src = p_temp_x + (i * dim_length + start) * length_per_step;

            /* Loop iterates over the axis */
            for (j = 0; j < num_values; j++)
            {
                p_x = (xb_vec4Mx8*) src;
                ax = PDX_LA_4MX8_PP(p_x);
                for (WORD32 k = 0; k < num_simd16_ops; k++)
                {
                    PDX_LA_4MX8_IP(x, ax, p_x);
                    PDX_SA_4MX8_IP(x, az, p_z);
                }

                PDX_LAV_4MX8_XP(x, ax, p_x, m);
                PDX_SAV_4MX8_XP(x, az, p_z, m);

                src += step * length_per_step;
            }
        }
    }
    PDX_SAPOS_4MX8_FP(az, p_z);

    return 0;
}
