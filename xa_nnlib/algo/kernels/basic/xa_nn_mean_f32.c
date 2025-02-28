/*******************************************************************************
 * Copyright (c) 2024-2025 Cadence Design Systems, Inc.
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
#include <string.h>

// Function to check if two consecutive axes are contiguous
static inline WORD32 are_two_axes_contiguous(WORD32 a, WORD32 b)
{
    return (b == a + CONST_ONE);
}

// Check if an axis is in the given axes list
static inline WORD32 is_axis_in_list(WORD32 axis,
        const WORD32 *axes,
        WORD32 num_axes)
{
    WORD32 i;
    for (i = 0; i < num_axes; i++)
    {
        if (axes[i] == axis)
        {
            return CONST_ONE;
        }
    }
    return 0;
}

// Sort axes in ascending order (Bubble Sort)
static inline void sort_axes(WORD32 *axes, WORD32 num_axes)
{
    WORD32 temp;
    WORD32 i, j;
    for (i = 0; i < (num_axes - CONST_ONE); i++)
    {
        for (j = 0; j < (num_axes - i - CONST_ONE); j++)
        {
            if (axes[j] > axes[j + CONST_ONE])
            {
                temp = axes[j];
                axes[j] = axes[j + CONST_ONE];
                axes[j + CONST_ONE] = temp;
            }
        }
    }
}

// Merge contiguous axes
// Merge contiguous dimensions other than axes
static inline void merge_axes_dims(const WORD32 *const input_shape,
        WORD32 num_dims,
        WORD32 *axes,
        WORD32 num_axes,
        WORD32 *new_input_shape,
        WORD32 *new_num_dims,
        WORD32 *new_axes,
        WORD32 *new_num_axes)
{
    *new_num_dims = 0;
    *new_num_axes = 0;

    // Sort axes to ensure merging happens in correct order
    sort_axes(axes, num_axes);

    WORD32 i = 0;

    while (i < num_dims)
    {
        if (is_axis_in_list(i, axes, num_axes))
        {
            // If the axis is in the given axes list, check if it can merge
            WORD32 merged_size = input_shape[i];

            while (((i + CONST_ONE) < num_dims)
                    && is_axis_in_list((i + CONST_ONE), axes, num_axes)
                    && are_two_axes_contiguous(i, (i + CONST_ONE)))
            {
                merged_size *= input_shape[i + CONST_ONE];
                i++;
            }

            // Store merged dimension
            new_input_shape[*new_num_dims] = merged_size;
            new_axes[*new_num_axes] = *new_num_dims; // Store the merged axis position
            (*new_num_dims)++;
            (*new_num_axes)++;
            i++;
        }
        else
        {
            // Start merging contiguous dimensions
            WORD32 merged_size = input_shape[i];
            while (((i + CONST_ONE) < num_dims)
                    && !(is_axis_in_list((i + CONST_ONE), axes, num_axes)))
            {
                merged_size *= input_shape[i + CONST_ONE];
                i++; // Move to the next dimension
            }
            new_input_shape[*new_num_dims] = merged_size;
            (*new_num_dims)++;
            i++;
        }
    }
}

WORD32 xa_nn_mean_f32_f32(FLOAT32 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        WORD32 num_out_dims,
        const FLOAT32 *__restrict__ p_inp,
        const WORD32 *const p_inp_shape,
        WORD32 num_inp_dims,
        const WORD32 *__restrict__ p_axis,
        WORD32 num_axis_dims)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_shape, UNSUPPORTED_PARAM);

    /* Invalid input checks */
    XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND(((num_out_dims <= 0) || (num_out_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND(((num_axis_dims < 0) || (num_axis_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);

    WORD32 axis_itr = 0, inp_itr = 0, out_itr = 0;
	for (inp_itr = 0; inp_itr < num_inp_dims; inp_itr++)
    {
        XA_NNLIB_ARG_CHK_COND((p_inp_shape[inp_itr] <= 0), UNSUPPORTED_PARAM);
    }

    WORD32 out_length = CONST_ONE;
    for (out_itr = 0; out_itr < num_out_dims; out_itr++)
    {
        XA_NNLIB_ARG_CHK_COND((p_out_shape[out_itr] <= 0), UNSUPPORTED_PARAM);
        out_length *= p_out_shape[out_itr];
    }

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_axis, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), UNSUPPORTED_PARAM);

    WORD32 axis_itr = 0, inp_itr = 0, out_itr = 0;
    WORD32 num_elm_in_axis = CONST_ONE;

    WORD32 new_axes[MAX_DIMS] = {0};   // Buffer for new axes
    WORD32 i = 0;

    if (p_axis != NULL)
    {
        WORD32 current;
        WORD32 dim_exist[MAX_DIMS];
        WORD32 repeated_axis_dims = 0;
        memset(dim_exist, 0, sizeof(dim_exist));

        for (axis_itr = 0; axis_itr < num_axis_dims; axis_itr++)
        {
            current = p_axis[axis_itr];
            XA_NNLIB_ARG_CHK_COND(
                    ((current < 0) || (current > (num_inp_dims - CONST_ONE))),
                    UNSUPPORTED_PARAM);

            /* Avoid calculation in case of repeated axis dims*/
            if (dim_exist[current] == 0)
            {
                num_elm_in_axis *= p_inp_shape[current];
                dim_exist[current] = CONST_ONE;
                new_axes[i] = current;
                i++;
            }
            else
            {
                repeated_axis_dims++;
            }
        }
        num_axis_dims = num_axis_dims - repeated_axis_dims;
    }


    if ((p_axis == NULL ) || (num_axis_dims == 0) ||
             (num_axis_dims == num_inp_dims) || (num_inp_dims == CONST_ONE))
    {
        WORD32 num_elm = CONST_ONE;
        WORD32 len;
        for (len = 0; len < num_inp_dims; len++)
        {
            num_elm *= p_inp_shape[len];
        }
        FLOAT32 num_elm_f = num_elm;

        FLOAT32 one_over_num_elm = XT_DIV_S(CONST_ONE, num_elm_f);
        xtfloat out = 0;
        /* Input data vectors. */
        xb_vecMxf32 x0 = 0, x1 = 0;

        const xb_vecMxf32 *p_x;
        valign ax;
        p_x = (xb_vecMxf32*) p_inp;
        ax = PDX_LA_MXF32_PP(p_x);
        WORD32 rem_elm = ((num_elm & (PDX_M - CONST_ONE)) * sizeof(float));
        WORD32 Itr = 0;
        for (Itr = 0; Itr < (num_elm >> CONST_TWO); Itr++)
        {
            PDX_LA_MXF32_IP(x1, ax, p_x);
            x0 = PDX_ADD_MXF32(x0, x1);
        }
        x1 = 0;
        PDX_LAV_MXF32_XP(x1, ax, p_x, rem_elm);

        x0 = PDX_ADD_MXF32(x0, x1);
        /* Store output */
        out = PDX_RADD_MXF32(x0);

        *p_out = out * one_over_num_elm;

        return 0;
    }
    /* For contiguous axis merge */
    WORD32 new_input_shape[MAX_DIMS] = {0};  // Buffer for new shape
    WORD32 new_axes_data[MAX_DIMS] = {0};   // Buffer for new axes
    WORD32 new_num_inp_dims = 0, new_num_axis_dims = 0;
    merge_axes_dims(p_inp_shape, num_inp_dims, new_axes, num_axis_dims,
            new_input_shape, &new_num_inp_dims, new_axes_data,
            &new_num_axis_dims);

    WORD32 last_dim = 0;

    if (new_axes_data[new_num_axis_dims - CONST_ONE]
            == (new_num_inp_dims - CONST_ONE))
    {
        last_dim = CONST_ONE;
    }

    const xb_vecMxf32 *__restrict__ p_src1;
    const xb_vecMxf32 *__restrict__ p_src2;
    const xb_vecMxf32 *__restrict__ p_in_mxf32;
    xb_vecMxf32 *__restrict__ p_dst;
    valign align_src1, align_src2;
    valign align_dst;
    xb_vecMxf32 x0 = 0, x1 = 0, x2 = 0;
    WORD32 axis_count , out_loop_count;
    WORD32 rem_elm, rem_axis, inner_stride_bytes;
    xb_vecMxf32 sum = 0;

    switch(new_num_inp_dims)
    {
        case 2:
        {
            if(last_dim)
            {
                const xb_vecMxf32 *__restrict__ p_x = (xb_vecMxf32 *)p_inp;
                valign ax = PDX_LA_MXF32_PP(p_x);
                FLOAT32 out;
                axis_count = new_input_shape[CONST_ONE];
                out_loop_count = new_input_shape[0];

                rem_elm = (axis_count & (PDX_M - CONST_ONE)) * sizeof(float);
                WORD32 d0 , d1;
                for (d0 = 0; d0 < out_loop_count; d0++)
                {
                    sum = PDX_ZERO_MXF32();
                    for (d1 = 0; d1 < (axis_count >> CONST_TWO); d1++)
                    {
                        PDX_LA_MXF32_IP(x0, ax, p_x);
                        sum = PDX_ADD_MXF32(sum, x0);
                    }
                    PDX_LAV_MXF32_XP(x0, ax, p_x, rem_elm);
                    sum = PDX_ADD_MXF32(sum, x0);
                    // Reduce and store result
                    out = PDX_RADD_MXF32(sum);
                    p_out[d0] = out;
                }
            }
            else
            {
                p_src1 = (const xb_vecMxf32 *)(p_inp);
                p_src2 = (const xb_vecMxf32 *)(p_inp);
                p_in_mxf32 = (const xb_vecMxf32 *)(p_inp);
                align_src1 = PDX_LA_MXF32_PP(p_src1);
                align_src2 = PDX_LA_MXF32_PP(p_src2);
                p_dst = (xb_vecMxf32 *)p_out;
                align_dst = PDX_Z_ALIGN();

                axis_count = new_input_shape[0];
                out_loop_count = new_input_shape[1];
                inner_stride_bytes = out_loop_count << CONST_TWO;
                rem_elm = ((out_loop_count & (PDX_M - CONST_ONE))
                        * sizeof(float));
                rem_axis = ((axis_count - CONST_ONE) & (CONST_ONE));
                if(axis_count == CONST_ONE)
                {
                    WORD32 j;
                    for (j = 0; j < (out_loop_count - CONST_THREE); j +=
                            CONST_FOUR)
                    {
                        PDX_LA_MXF32_IP(x1, align_src1, p_src1);
                        PDX_SA_MXF32_IP(x1, align_dst, p_dst);
                    }
                    PDX_LAV_MXF32_XP(x1, align_src1, p_src1, rem_elm);
                    PDX_SAV_MXF32_XP(x1, align_dst, p_dst, rem_elm);
                    PDX_SAPOS_MXF32_FP(align_dst, p_dst);
                }
                else
                {
                    WORD32 d1, d0;
                    for (d1 = 0; d1 < (out_loop_count - CONST_THREE); d1 +=
                            CONST_FOUR)
                    {
                        sum = PDX_ZERO_MXF32();
                        p_src1 = (const xb_vecMxf32 *)(p_inp + d1);
                        p_src2 = (const xb_vecMxf32*) (p_inp + d1
                                + out_loop_count);
                        align_src1 = PDX_LA_MXF32_PP(p_src1);
                        PDX_LA_MXF32_XP(sum, align_src1, p_src1,
                                (inner_stride_bytes * CONST_TWO));
                        for (d0 = 0;
                                d0 < ((axis_count - CONST_ONE) >> CONST_ONE);
                                d0++)
                        {
                            /* Align load priming of input */
                            align_src2 = PDX_LA_MXF32_PP(p_src2);
                            align_src1 = PDX_LA_MXF32_PP(p_src1);

                            /* Load input elements with stride "inner_stride" */
                            PDX_LA_MXF32_XP(x1, align_src2, p_src2,
                                    (inner_stride_bytes * CONST_TWO));
                            PDX_LA_MXF32_XP(x2, align_src1, p_src1,
                                    (inner_stride_bytes * CONST_TWO));

                            /* Calculate sum across each lane of vector */
                            sum = sum + PDX_ADD_MXF32(x1, x2);
                        }
                        if (rem_axis)
                        {
                            /* Align load priming of input */
                            align_src2 = PDX_LA_MXF32_PP(p_src2);

                            /* Load input elements with stride "inner_stride" */
                            PDX_LA_MXF32_XP(x1, align_src2, p_src2,
                                    (inner_stride_bytes * CONST_TWO));

                            /* Calculate maximum across each lane of vector */
                            sum = PDX_ADD_MXF32(sum, x1);
                        }
                        /* Store output */
                        PDX_SA_MXF32_IP(sum, align_dst, p_dst);
                    }
                    if(rem_elm)
                    {
                         /* Process remaining elements */
                        FLOAT32 *p_inp1 = (FLOAT32 *)(p_inp + d1);
                        p_in_mxf32 = (const xb_vecMxf32 *)(p_inp1);
                        WORD32 k;
                        xb_vecMxf32 rem_sum = 0;
                        align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);
                        PDX_LAV_MXF32_XP(rem_sum, align_src1, p_in_mxf32,
                                rem_elm);

                        for (k = 0; k < (axis_count - CONST_ONE); k++)
                        {
                            p_inp1 += out_loop_count;
                            p_in_mxf32 = (const xb_vecMxf32 *)(p_inp1);

                            /* Align load priming of input */
                            align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);

                            /* Load input elements with stride "inner_stride" */
                            PDX_LAV_MXF32_XP(x1, align_src1, p_in_mxf32,
                                    rem_elm);

                            rem_sum = PDX_ADD_MXF32(rem_sum, x1);
                        }

                        /* Store output */
                        PDX_SAV_MXF32_XP(rem_sum, align_dst, p_dst, rem_elm);
                    }
                    PDX_SAPOS_MXF32_FP(align_dst, p_dst);
                }

            }
        }
            break;
        case 3:
        {
            if(last_dim)
            {
                FLOAT32 out  = 0;
                const xb_vecMxf32 *__restrict__ p_x = (xb_vecMxf32 *)p_inp;
                valign a_x = PDX_LA_MXF32_PP(p_x);

                WORD32 Dim0 = new_input_shape[0];
                WORD32 Dim1 = new_input_shape[CONST_ONE];
                WORD32 Dim2 = new_input_shape[CONST_TWO];

                rem_elm = ((Dim2 & (PDX_M - CONST_ONE)) * sizeof(float));
                WORD32 d1, d0, d2;
                for(d1 = 0; d1 < Dim1; d1++)
                {
                    WORD32 base_offset = d1 * Dim2;
                    sum = PDX_ZERO_MXF32();
                    for(d0 = 0; d0 < Dim0; d0++)
                    {
                        FLOAT32 *p_inp1 = (FLOAT32*) (p_inp + (d0 * Dim1 * Dim2)
                                + base_offset);
                        p_x = (xb_vecMxf32 *)p_inp1;
                        a_x = PDX_LA_MXF32_PP(p_x);
                        for(d2 = 0; d2 < (Dim2 >> CONST_TWO); d2++)
                        {
                            PDX_LA_MXF32_IP(x0, a_x, p_x);
                            sum = PDX_ADD_MXF32(sum, x0);
                        }
                        PDX_LAV_MXF32_XP(x0, a_x, p_x, rem_elm);
                        sum = PDX_ADD_MXF32(sum, x0);
                    }
                    // Reduce and store result
                    out = PDX_RADD_MXF32(sum);
                    p_out[d1] = out;
                }
            }
            else
            {
                p_src1 = (const xb_vecMxf32 *)(p_inp);
                p_src2 = (const xb_vecMxf32 *)(p_inp);
                p_in_mxf32 = (const xb_vecMxf32 *)(p_inp);
                align_src1 = PDX_LA_MXF32_PP(p_src1);
                align_src2 = PDX_LA_MXF32_PP(p_src2);
                p_dst = (xb_vecMxf32 *)p_out;
                align_dst = PDX_Z_ALIGN();

                WORD32 Dim0 = new_input_shape[0];
                WORD32 Dim1 = new_input_shape[CONST_ONE];
                WORD32 Dim2 = new_input_shape[CONST_TWO];

                axis_count = Dim1;
                out_loop_count = Dim0 * Dim2;

                inner_stride_bytes = Dim2 << CONST_TWO;
                rem_axis = ((axis_count - CONST_ONE) & (CONST_ONE));
                if(axis_count == CONST_ONE)
                {
                    rem_elm = ((out_loop_count & (PDX_M - CONST_ONE))
                            * sizeof(float));
                    WORD32 j;
                    for (j = 0; j < (out_loop_count - CONST_THREE); j +=
                            CONST_FOUR)
                    {
                        PDX_LA_MXF32_IP(x1, align_src1, p_src1);
                        PDX_SA_MXF32_IP(x1, align_dst, p_dst);
                    }
                    PDX_LAV_MXF32_XP(x1, align_src1, p_src1, rem_elm);
                    PDX_SAV_MXF32_XP(x1, align_dst, p_dst, rem_elm);
                    PDX_SAPOS_MXF32_FP(align_dst, p_dst);
                }
                else
                {
                    rem_elm = ((Dim2 & (PDX_M - CONST_ONE)) * sizeof(float));
                    WORD32 d0, d2, d1;
                    for(d0 = 0; d0 < Dim0; d0++)
                    {
                        WORD32 base_offset = d0 * Dim1 * Dim2;
                        sum = PDX_ZERO_MXF32();
                        for (d2 = 0; d2 < (Dim2 - CONST_THREE);
                                d2 += CONST_FOUR)
                        {
                            sum = PDX_ZERO_MXF32();
                            p_src1 = (const xb_vecMxf32*) (p_inp + d2
                                    + base_offset);
                            p_src2 = (const xb_vecMxf32*) (p_inp + d2
                                    + base_offset + Dim2);
                            align_src1 = PDX_LA_MXF32_PP(p_src1);
                            PDX_LA_MXF32_XP(sum, align_src1, p_src1,
                                    (inner_stride_bytes * CONST_TWO));
                            for (d1 = 0;
                                  d1 < ((axis_count - CONST_ONE) >> CONST_ONE);
                                  d1++)
                            {
                                /* Align load priming of input */
                                align_src2 = PDX_LA_MXF32_PP(p_src2);
                                align_src1 = PDX_LA_MXF32_PP(p_src1);

                                /* Load input elements with stride "inner_stride" */
                                PDX_LA_MXF32_XP(x1, align_src2, p_src2,
                                        (inner_stride_bytes * CONST_TWO));
                                PDX_LA_MXF32_XP(x2, align_src1, p_src1,
                                        (inner_stride_bytes * CONST_TWO));

                                /* Calculate sum across each lane of vector */
                                sum = sum + PDX_ADD_MXF32(x1, x2);
                            }
                            if (rem_axis)
                            {
                                /* Align load priming of input */
                                align_src2 = PDX_LA_MXF32_PP(p_src2);

                                /* Load input elements with stride "inner_stride" */
                                PDX_LA_MXF32_XP(x1, align_src2, p_src2,
                                        (inner_stride_bytes * CONST_TWO));

                                /* Calculate maximum across each lane of vector */
                                sum = PDX_ADD_MXF32(sum, x1);
                            }
                            /* Store output */
                            PDX_SA_MXF32_IP(sum, align_dst, p_dst);
                        }
                        if(rem_elm)
                        {
                             /* Process remaining elements */
                            FLOAT32 *p_inp2 = (FLOAT32*) (p_inp + d2
                                    + base_offset);

                            p_in_mxf32 = (const xb_vecMxf32 *)(p_inp2);
                            WORD32 k;
                            xb_vecMxf32 rem_sum = 0;
                            align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);
                            PDX_LAV_MXF32_XP(rem_sum, align_src1, p_in_mxf32,
                                    rem_elm);

                            for (k = 0; k < (axis_count - CONST_ONE); k++)
                            {
                                p_inp2 += Dim2;
                                p_in_mxf32 = (const xb_vecMxf32 *)(p_inp2);

                                /* Align load priming of input */
                                align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);

                                /* Load input elements with stride "inner_stride" */
                                PDX_LAV_MXF32_XP(x1, align_src1, p_in_mxf32,
                                        rem_elm);

                                rem_sum = PDX_ADD_MXF32(rem_sum, x1);
                            }
                            /* Store output */
                            PDX_SAV_MXF32_XP(rem_sum, align_dst, p_dst,
                                    rem_elm);
                        }
                        PDX_SAPOS_MXF32_FP(align_dst, p_dst);
                    }
                }
            }
        }
            break;
        case 4:
        {
            if(last_dim)
            {
                FLOAT32 out;
                const xb_vecMxf32 *__restrict__ p_x = (xb_vecMxf32 *)p_inp;
                xtfloat *p_z = (xtfloat *)p_out;
                valign a_x = PDX_LA_MXF32_PP(p_x);

                WORD32 Dim0 = new_input_shape[0];
                WORD32 Dim1 = new_input_shape[CONST_ONE];
                WORD32 Dim2 = new_input_shape[CONST_TWO];
                WORD32 Dim3 = new_input_shape[CONST_THREE];

                WORD32 d2_offset = 0, d0_offset = 0;
                rem_elm = ((Dim3 & (PDX_M - CONST_ONE)) * sizeof(float));
                WORD32 d0, d2, d1, d3;
                for(d0 = 0; d0 < Dim0; d0++)
                {
                    d0_offset = d0 * Dim1 * Dim2 * Dim3;
                    for(d2 = 0; d2 < Dim2; d2++)
                    {
                        d2_offset = d2 * Dim3;
                        sum = PDX_ZERO_MXF32();
                        for(d1 = 0; d1 < Dim1; d1++)
                        {
                            FLOAT32 *p_inp1 =
                                    (FLOAT32*) (p_inp + (d1 * Dim2 * Dim3)
                                            + d0_offset + d2_offset);
                            p_x = (xb_vecMxf32 *)p_inp1;
                            a_x = PDX_LA_MXF32_PP(p_x);
                            for(d3 = 0; d3 < (Dim3 >> CONST_TWO); d3++)
                            {
                                PDX_LA_MXF32_IP(x0, a_x, p_x);
                                sum = PDX_ADD_MXF32(sum, x0);
                            }
                            PDX_LAV_MXF32_XP(x0, a_x, p_x, rem_elm);
                            sum = PDX_ADD_MXF32(sum, x0);
                        }
                        // Reduce and store result
                        out = PDX_RADD_MXF32(sum);
                        xtfloat_storeip(out, p_z, CONST_FOUR);
                    }
                }
            }
            else
            {
                p_src1 = (const xb_vecMxf32 *)(p_inp);
                p_src2 = (const xb_vecMxf32 *)(p_inp);
                p_in_mxf32 = (const xb_vecMxf32 *)(p_inp);
                align_src1 = PDX_LA_MXF32_PP(p_src1);
                align_src2 = PDX_LA_MXF32_PP(p_src2);
                p_dst = (xb_vecMxf32 *)p_out;
                align_dst = PDX_Z_ALIGN();

                WORD32 Dim0 = new_input_shape[0];
                WORD32 Dim1 = new_input_shape[1];
                WORD32 Dim2 = new_input_shape[2];
                WORD32 Dim3 = new_input_shape[3];

                inner_stride_bytes = Dim3 << CONST_TWO;
                rem_elm = ((Dim3 & (PDX_M - CONST_ONE)) * sizeof(float));
                rem_axis = ((Dim2 - CONST_ONE) & (CONST_ONE));
                WORD32 d0, d1, d2, d3;
                for(d1 = 0; d1 < Dim1; d1++)
                {
                    WORD32 base_offset = (d1 * Dim2 * Dim3);
                    sum = PDX_ZERO_MXF32();
                    for (d3 = 0; d3 < (Dim3 - CONST_THREE) ; d3 += CONST_FOUR)
                    {
                        sum = PDX_ZERO_MXF32();
                        for(d0 = 0; d0 < Dim0; d0++)
                        {
                            p_src1 = (const xb_vecMxf32*) ((p_inp + d3
                                    + base_offset) + (d0 * Dim1 * Dim2 * Dim3));
                            p_src2 = (const xb_vecMxf32*) ((p_inp + d3
                                    + base_offset) + (d0 * Dim1 * Dim2 * Dim3)
                                    + Dim3);
                            align_src1 = PDX_LA_MXF32_PP(p_src1);
                            PDX_LA_MXF32_XP(x0, align_src1, p_src1,
                                    (inner_stride_bytes * CONST_TWO));
                            sum = sum + x0;
                            for (d2 = 0; d2 < ((Dim2 - CONST_ONE) >> CONST_ONE);
                                    d2++)
                            {
                                /* Align load priming of input */
                                align_src2 = PDX_LA_MXF32_PP(p_src2);
                                align_src1 = PDX_LA_MXF32_PP(p_src1);

                                /* Load input elements with stride "inner_stride" */
                                PDX_LA_MXF32_XP(x1, align_src2, p_src2,
                                        (inner_stride_bytes * CONST_TWO));
                                PDX_LA_MXF32_XP(x2, align_src1, p_src1,
                                        (inner_stride_bytes * CONST_TWO));

                                /* Calculate sum across each lane of vector */
                                sum = sum + PDX_ADD_MXF32(x1, x2);
                            }
                            if (rem_axis)
                            {
                                /* Align load priming of input */
                                align_src2 = PDX_LA_MXF32_PP(p_src2);

                                /* Load input elements with stride "inner_stride" */
                                PDX_LA_MXF32_XP(x1, align_src2, p_src2,
                                        (inner_stride_bytes * CONST_TWO));

                                /* Calculate maximum across each lane of vector */
                                sum = PDX_ADD_MXF32(sum, x1);
                            }
                        }
                        /* Store output */
                        PDX_SA_MXF32_IP(sum, align_dst, p_dst);
                    }
                    if(rem_elm)
                    {
                         /* Process remaining elements */
                        WORD32 k;
                        xb_vecMxf32 rem_sum = 0, x0 = 0;
                        for(d0 = 0; d0 < Dim0; d0++)
                        {
                            FLOAT32 *p_inp1 = (FLOAT32*) ((p_inp + d3
                                    + base_offset) + (d0 * Dim1 * Dim2 * Dim3));
                            p_in_mxf32 = (const xb_vecMxf32 *)(p_inp1);
                            align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);
                            PDX_LAV_MXF32_XP(x0, align_src1, p_in_mxf32,
                                    rem_elm);
                            rem_sum = rem_sum + x0;
                            for (k = 0; k < (Dim2 - CONST_ONE); k++)
                            {
                                p_inp1 += Dim3;
                                p_in_mxf32 = (const xb_vecMxf32 *)(p_inp1);

                                /* Align load priming of input */
                                align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);

                                /* Load input elements with stride "inner_stride" */
                                PDX_LAV_MXF32_XP(x1, align_src1, p_in_mxf32,
                                        rem_elm);

                                rem_sum = PDX_ADD_MXF32(rem_sum, x1);
                            }
                        }
                        /* Store output */
                        PDX_SAV_MXF32_XP(rem_sum, align_dst, p_dst, rem_elm);
                    }
                    PDX_SAPOS_MXF32_FP(align_dst, p_dst);
                }
            }
        }
            break;
        case 5:
        {
            if(last_dim)
            {
                FLOAT32 out;
                const xb_vecMxf32 *__restrict__ p_x = (xb_vecMxf32 *)p_inp;
                valign a_x = PDX_LA_MXF32_PP(p_x);
                xtfloat *p_z = (xtfloat *)p_out;

                WORD32 Dim0 = new_input_shape[0];
                WORD32 Dim1 = new_input_shape[CONST_ONE];
                WORD32 Dim2 = new_input_shape[CONST_TWO];
                WORD32 Dim3 = new_input_shape[CONST_THREE];
                WORD32 Dim4 = new_input_shape[CONST_FOUR];

                WORD32 d0_offset = 0, d1_offset = 0;
                WORD32 d2_offset = 0;
                WORD32 d3_offset = 0;

                rem_elm = (Dim4 & (PDX_M - CONST_ONE)) * sizeof(float);
                WORD32 d0, d2, d4, d1, d3;
                for (d1 = 0; d1 < Dim1; d1++)
                {
                    d1_offset = d1 * Dim2 * Dim3 * Dim4;
                    for(d3 = 0; d3 < Dim3; d3++)
                    {
                        d3_offset = d3 * Dim4;
                        sum = PDX_ZERO_MXF32();

                        for (d0 = 0; d0 < Dim0; d0++)
                        {
                            d0_offset = d0 * Dim1 * Dim2 * Dim3 * Dim4;

                            for (d2 = 0; d2 < Dim2; d2++)
                            {
                                d2_offset = d2 * Dim3 * Dim4;

                                FLOAT32 *p_inp1 = (FLOAT32*) (p_inp + d0_offset
                                        + d1_offset + d2_offset + d3_offset);
                                p_x = (xb_vecMxf32 *)p_inp1;
                                a_x = PDX_LA_MXF32_PP(p_x);

                                for (d4 = 0; d4 < (Dim4 >> CONST_TWO); d4++)
                                {
                                    PDX_LA_MXF32_IP(x0, a_x, p_x);
                                    sum = PDX_ADD_MXF32(sum, x0);
                                }
                                PDX_LAV_MXF32_XP(x0, a_x, p_x, rem_elm);
                                sum = PDX_ADD_MXF32(sum, x0);
                            }
                        }

                        // Reduce and store result
                        out = PDX_RADD_MXF32(sum);
                        xtfloat_storeip(out, p_z, CONST_FOUR);
                    }
                }
            }
            else
            {
                p_src1 = (const xb_vecMxf32 *)(p_inp);
                p_src2 = (const xb_vecMxf32 *)(p_inp);
                p_in_mxf32 = (const xb_vecMxf32 *)(p_inp);
                align_src1 = PDX_LA_MXF32_PP(p_src1);
                align_src2 = PDX_LA_MXF32_PP(p_src2);
                p_dst = (xb_vecMxf32 *)p_out;
                align_dst = PDX_Z_ALIGN();

                WORD32 Dim0 = new_input_shape[0];
                WORD32 Dim1 = new_input_shape[CONST_ONE];
                WORD32 Dim2 = new_input_shape[CONST_TWO];
                WORD32 Dim3 = new_input_shape[CONST_THREE];
                WORD32 Dim4 = new_input_shape[CONST_FOUR];

                inner_stride_bytes = Dim4 << CONST_TWO;
                rem_elm = ((Dim4 & (PDX_M - CONST_ONE)) * sizeof(float));
                rem_axis = ((Dim3 - CONST_ONE) & (CONST_ONE));
                WORD32 d4 = 0, d0d2, d1, d3;
                for(d0d2 = 0; d0d2 < (Dim0 * Dim2); d0d2++)
                {
                    WORD32 base_offset = (d0d2 * Dim3 * Dim4);
                    sum = PDX_ZERO_MXF32();
                    for (d4 = 0; d4 < (Dim4 - CONST_THREE) ; d4 += CONST_FOUR)
                    {
                        sum = PDX_ZERO_MXF32();
                        for(d1 = 0; d1 < Dim1; d1++)
                        {
                            p_src1 = (const xb_vecMxf32*) ((p_inp + d4
                                    + base_offset) + (d1 * Dim2 * Dim3 * Dim4));
                            p_src2 = (const xb_vecMxf32*) ((p_inp + d4
                                    + base_offset) + (d1 * Dim2 * Dim3 * Dim4)
                                    + Dim4);
                            align_src1 = PDX_LA_MXF32_PP(p_src1);
                            PDX_LA_MXF32_XP(x0, align_src1, p_src1,
                                    (inner_stride_bytes * CONST_TWO));
                            sum = sum + x0;
                            for (d3 = 0; d3 < ((Dim3 - CONST_ONE) >> CONST_ONE);
                                    d3++)
                            {
                                /* Align load priming of input */
                                align_src2 = PDX_LA_MXF32_PP(p_src2);
                                align_src1 = PDX_LA_MXF32_PP(p_src1);

                                /* Load input elements with stride "inner_stride" */
                                PDX_LA_MXF32_XP(x1, align_src2, p_src2,
                                        (inner_stride_bytes * CONST_TWO));
                                PDX_LA_MXF32_XP(x2, align_src1, p_src1,
                                        (inner_stride_bytes * CONST_TWO));

                                /* Calculate sum across each lane of vector */
                                sum = sum + PDX_ADD_MXF32(x1, x2);
                            }
                            if (rem_axis)
                            {
                                /* Align load priming of input */
                                align_src2 = PDX_LA_MXF32_PP(p_src2);

                                /* Load input elements with stride "inner_stride" */
                                PDX_LA_MXF32_XP(x1, align_src2, p_src2,
                                        (inner_stride_bytes * CONST_TWO));

                                /* Calculate maximum across each lane of vector */
                                sum = PDX_ADD_MXF32(sum, x1);
                            }
                        }
                        /* Store output */
                        PDX_SA_MXF32_IP(sum, align_dst, p_dst);
                    }
                    if(rem_elm)
                    {
                         /* Process remaining elements */
                        xb_vecMxf32 rem_sum = 0, x0 = 0;
                        WORD32 k;
                        for(d1 = 0; d1 < Dim1; d1++)
                        {
                            FLOAT32 *p_inp1 = (FLOAT32*) ((p_inp + d4
                                    + base_offset) + (d1 * Dim2 * Dim3 * Dim4));
                            p_in_mxf32 = (const xb_vecMxf32 *)(p_inp1);
                            align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);
                            PDX_LAV_MXF32_XP(x0, align_src1, p_in_mxf32,
                                    rem_elm);
                            rem_sum = rem_sum + x0;
                            for (k = 0; k < (Dim3 - CONST_ONE); k++)
                            {
                                p_inp1 += Dim4;
                                p_in_mxf32 = (const xb_vecMxf32 *)(p_inp1);

                                /* Align load priming of input */
                                align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);

                                /* Load input elements with stride "inner_stride" */
                                PDX_LAV_MXF32_XP(x1, align_src1, p_in_mxf32,
                                        rem_elm);

                                rem_sum = PDX_ADD_MXF32(rem_sum, x1);
                            }
                        }
                        /* Store output */
                        PDX_SAV_MXF32_XP(rem_sum, align_dst, p_dst, rem_elm);
                    }
                    PDX_SAPOS_MXF32_FP(align_dst, p_dst);
                }
            }
        }
            break;
        default:
            break;

    }
    if (num_elm_in_axis > CONST_ONE)
    {
        WORD32 itr = 0;
        xb_vecMxf32 *__restrict__ p_z = (xb_vecMxf32*) (p_out);
        xb_vecMxf32 *__restrict__ p_x = (xb_vecMxf32*) (p_out);
        valign ax = PDX_LA_MXF32_PP(p_x);
        valign az = PDX_Z_ALIGN();

        WORD32 rem_elm = (out_length & (PDX_M - CONST_ONE)) * sizeof(float);
        xb_vecMxf32 multiplier = PDX_DIV_MXF32(CONST_ONE,
                (FLOAT32) num_elm_in_axis);
        xb_vecMxf32 x1 = 0, z0 = 0;
        for (itr = 0; itr < (out_length >> CONST_TWO); itr++)
        {
            PDX_LA_MXF32_IP(x1, ax, p_x);
            z0 = PDX_MUL_MXF32(x1, multiplier);
            PDX_SA_MXF32_IP(z0, az, p_z);
        }

        PDX_LAV_MXF32_XP(x1, ax, p_x, rem_elm);
        z0 = PDX_MUL_MXF32(x1, multiplier);
        PDX_SAV_MXF32_XP(z0, az, p_z, rem_elm);
        PDX_SAPOS_MXF32_FP(az, p_z);
    }
    return 0;
}