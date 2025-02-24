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
#include <math.h>

WORD32 xa_nn_elm_quantize_f32_sym4(WORD8 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp,
        const WORD32 *const p_inp_shape,
        WORD32 num_inp_dims,
        WORD32 *p_axis,
        FLOAT32 *p_out_scale,
        WORD32 quant_min,
        WORD32 quant_max)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_scale, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_scale, sizeof(FLOAT32), UNSUPPORTED_PARAM);

    /* Invalid input checks
     * quant_min should be >= -8,
     * quant_max should be <= 7,
     * num_inp_dims should be greater than 0 and less than or equal to 5
     * p_inp_shape values should be positive
     */
    XA_NNLIB_ARG_CHK_COND((quant_min < INT4_LOWER_LIMIT), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND((quant_max > INT4_UPPER_LIMIT), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND((quant_max < quant_min), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND((num_inp_dims <= 0), UNSUPPORTED_PARAM);

    WORD32 i;

    for (i = 0; i < num_inp_dims; i++)
    {
        XA_NNLIB_ARG_CHK_COND((p_inp_shape[i] <= 0), UNSUPPORTED_PARAM);
    }

    /* Number of elements to be processed with a stride of 1 */
    WORD32 num_elm = CONST_ONE;
    /* Number of leading dimensions of axis */
    WORD32 leading_dims = CONST_ONE;
    /* Number of trailing dimensions of axis */
    WORD32 trailing_dims = CONST_ONE;
    WORD32 length_per_step = 0;
    WORD32 axis_count = CONST_ONE;

    WORD32 has_axis = 0;

    if (p_axis == NULL)
    {
        /* out_scale should not be equal to zero, nan and infinity */
        XA_NNLIB_ARG_CHK_COND(
                ((0 == *p_out_scale) || (isnan(*p_out_scale)) ||
                        (isinf(*p_out_scale))), UNSUPPORTED_PARAM);

        for (i = 0; i < num_inp_dims; i++)
        {
            num_elm *= p_inp_shape[i];
        }
    }
    else
    {
        WORD32 axis = *p_axis;
        /* Invalid input checks
         * axis should be in the range [0,num_inp_dims-1]
         * out_scale should not be equal to zero, nan and infinity
         */
        XA_NNLIB_ARG_CHK_COND(((axis < 0) || (axis >= num_inp_dims)),
                UNSUPPORTED_PARAM);

        for (i = 0; i < p_inp_shape[axis]; i++)
        {
            XA_NNLIB_ARG_CHK_COND(
                    ((0 == p_out_scale[i]) || (isnan(p_out_scale[i])) ||
                            (isinf(p_out_scale[i]))), UNSUPPORTED_PARAM);
        }

        /* Calculating leading dims */
        for (i = 0; i < axis; i++)
        {
            leading_dims *= p_inp_shape[i];
        }

        /* Calculating trailing dims */
        for (i = axis + CONST_ONE; i < num_inp_dims; i++)
        {
            trailing_dims *= p_inp_shape[i];
        }

        num_elm = trailing_dims;

        /* Number of elements to be skipped after trailing number of
         * elements quantized with scale values to get
         * the next base addresses.
         */
        length_per_step = p_inp_shape[axis] * trailing_dims;

        /* Length of the dimension along axis */
        axis_count = p_inp_shape[axis];

        has_axis = CONST_ONE;
    }

    xb_vecMxf32 x, s1;

    xb_vecMx32 z;
    xb_vecMx32 min, max;

    xb_vecMxf32 one_over_scale1, one = PDX_CONST_MXF32(CONST_ONE);

    xb_vecMx8 *__restrict__ p_z1;
    valign az1 = PDX_Z_ALIGN();

    xb_vecMx8 *__restrict__ p_z2;
    valign az2 = PDX_Z_ALIGN();

    const FLOAT32 *__restrict__ p_inp_base;
    xb_vecMxf32 *__restrict__ p_x1;
    xb_vecMxf32 *__restrict__ p_x2;

    const WORD8 *__restrict__ p_out_base;

    valign ax1, ax2;

    /* Setting rounding mode to zero - rounding to nearest integer */
    xb_int32 actual_scf = PDX_MOV32_SCF();
    xb_int32 converted_scf = PDX_AND_32(actual_scf, 0xFFFFFCFF);
    PDX_MOVSCF_32(converted_scf);

    WORD32 axis_index, leading_dims_index;

#ifndef ENABLE_4BIT_PACK  /* Code for bytes store (unpacked) */
    min = quant_min << SHIFT_FACTOR_4_BIT;
    max = quant_max << SHIFT_FACTOR_4_BIT;

    WORD32 two_times_lps = CONST_TWO * length_per_step;

    /* If the axis is given and is equal to last dimension */
    if (has_axis && (*p_axis == (num_inp_dims - CONST_ONE)))
    {
        valign as;

        xb_vecMxf32 * __restrict__ p_s = (xb_vecMxf32*) (p_out_scale);
        as = PDX_LA_MXF32_PP(p_s);

        WORD32 num_scalar_ops = (axis_count & (PDX_M - CONST_ONE));

        WORD32 rem_inp_elms = (num_scalar_ops * SIZE_OF_FLOAT);

        for (axis_index = 0; axis_index < (axis_count - CONST_THREE);
                axis_index += CONST_FOUR)
        {
            PDX_LA_MXF32_IP(s1, as, p_s);
            one_over_scale1 = PDX_DIV_MXF32(one, s1);

            for (leading_dims_index = 0;
                    leading_dims_index < leading_dims;
                    leading_dims_index++)
            {
                p_x1 = (xb_vecMxf32*) (p_inp
                        + (leading_dims_index * length_per_step)
                        + axis_index);
                ax1 = PDX_LA_MXF32_PP(p_x1);

                p_z1 = (xb_vecMx8*) (p_out
                        + (leading_dims_index * length_per_step)
                        + axis_index);

                PDX_LA_MXF32_IP(x, ax1, p_x1);

                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, SHIFT_FACTOR_4_BIT);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                PDX_SA32_MX8_IP(z, az1, p_z1);
                PDX_SAPOS_MX8_FP(az1, p_z1);
            }

        }
        if((axis_count & CONST_THREE) != 0)
        {
            PDX_LAV_MXF32_XP(s1, as, p_s, rem_inp_elms);
            one_over_scale1 = PDX_DIV_MXF32(one, s1);

            for (leading_dims_index = 0;
                    leading_dims_index < leading_dims;
                    leading_dims_index++)
            {
                p_x1 = (xb_vecMxf32*) (p_inp
                        + (leading_dims_index * length_per_step)
                        + axis_index);
                ax1 = PDX_LA_MXF32_PP(p_x1);

                p_z1 = (xb_vecMx8*) (p_out
                        + (leading_dims_index * length_per_step)
                        + axis_index);


                PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp_elms);

                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, SHIFT_FACTOR_4_BIT);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                PDX_SAV32_MX8_XP(z, az1, p_z1, num_scalar_ops);
                PDX_SAPOS_MX8_FP(az1, p_z1);
            }
        }
    }
    /* When axis is not given or axis is not last dim */
    else
    {
        WORD32 num_simd4_ops = (num_elm >> LOG2_PDX_M);
        WORD32 num_scalar_ops = (num_elm & (PDX_M - CONST_ONE));
        WORD32 rem_inp_elms = num_scalar_ops * SIZE_OF_FLOAT;

        /* Outermost loop iterates over the channels */
        for (axis_index = 0; axis_index < axis_count; axis_index++)
        {
            s1 = p_out_scale[axis_index];
            p_inp_base = p_inp + (axis_index * trailing_dims);
            p_out_base = p_out + (axis_index * trailing_dims);

            one_over_scale1 = PDX_DIV_MXF32(one, s1);

            /* This loop iterates over the leading dims.
             * All the elements are quantized at a time for
             * single scale once loaded
             */
            for (leading_dims_index = 0;
                    leading_dims_index < (leading_dims - CONST_ONE);
                    leading_dims_index += CONST_TWO)
            {
                p_x1 = (xb_vecMxf32*) p_inp_base;
                ax1 = PDX_LA_MXF32_PP(p_x1);
                p_z1 = (xb_vecMx8*) p_out_base;

                p_x2 = (xb_vecMxf32*) (p_inp_base + length_per_step);
                ax2 = PDX_LA_MXF32_PP(p_x2);
                p_z2 = (xb_vecMx8*) (p_out_base + length_per_step);

                for (i = 0; i < num_simd4_ops; i++)
                {
                    PDX_LA_MXF32_IP(x, ax1, p_x1);
                    x = PDX_MUL_MXF32(x, one_over_scale1);
                    x = PDX_FIRINT_MXF32(x);
                    z = PDX_TRUNC32_MXF32(x, SHIFT_FACTOR_4_BIT);
                    z = PDX_MIN_MX32(z, max);
                    z = PDX_MAX_MX32(z, min);

                    PDX_SA32_MX8_IP(z, az1, p_z1);

                    PDX_LA_MXF32_IP(x, ax2, p_x2);
                    x = PDX_MUL_MXF32(x, one_over_scale1);
                    x = PDX_FIRINT_MXF32(x);
                    z = PDX_TRUNC32_MXF32(x, SHIFT_FACTOR_4_BIT);
                    z = PDX_MIN_MX32(z, max);
                    z = PDX_MAX_MX32(z, min);

                    PDX_SA32_MX8_IP(z, az2, p_z2);

                }
                PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp_elms);
                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, SHIFT_FACTOR_4_BIT);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                PDX_SAV32_MX8_XP(z, az1, p_z1, num_scalar_ops);
                PDX_SAPOS_MX8_FP(az1, p_z1);

                PDX_LAV_MXF32_XP(x, ax2, p_x2, rem_inp_elms);
                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, SHIFT_FACTOR_4_BIT);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                PDX_SAV32_MX8_XP(z, az2, p_z2, num_scalar_ops);
                PDX_SAPOS_MX8_FP(az2, p_z2);

                p_inp_base = p_inp_base + two_times_lps;
                p_out_base = p_out_base + two_times_lps;
            }
            if ((leading_dims & CONST_ONE) != 0)
            {
                p_x1 = (xb_vecMxf32*) p_inp_base;
                ax1 = PDX_LA_MXF32_PP(p_x1);
                p_z1 = (xb_vecMx8*) p_out_base;

                for (i = 0; i < num_simd4_ops; i++)
                {
                    PDX_LA_MXF32_IP(x, ax1, p_x1);
                    x = PDX_MUL_MXF32(x, one_over_scale1);
                    x = PDX_FIRINT_MXF32(x);
                    z = PDX_TRUNC32_MXF32(x, SHIFT_FACTOR_4_BIT);
                    z = PDX_MIN_MX32(z, max);
                    z = PDX_MAX_MX32(z, min);

                    PDX_SA32_MX8_IP(z, az1, p_z1);
                }
                PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp_elms);
                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, SHIFT_FACTOR_4_BIT);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                PDX_SAV32_MX8_XP(z, az1, p_z1, num_scalar_ops);
                PDX_SAPOS_MX8_FP(az1, p_z1);
            }
        }
    }
#else /* Code for bits4x2 store */

    xb_vecMx32 odd_nibbles, even_nibbles, or;
    WORD8 *__restrict__ p_z_nibble;
    WORD32 rem_inp, rem_out;

    min = quant_min;
    max = quant_max;

    /* If the axis is given and is equal to last dimension */
    if (has_axis && (*p_axis == (num_inp_dims - CONST_ONE)))
    {
        valign as;

        xb_vecMxf32 * __restrict__ p_s;

        WORD32 rem_inp_elms1, rem_inp_elms2;
        WORD32 rem_out_elms1, rem_out_elms2;
        WORD32 increment, leading_dims_limit;
        WORD32 axis_count1, axis_count2, num_scalar_ops;

        /* If axis_count is even all the elements
         * are packed into bytes without leaving empty nibble.
         */
        if ((axis_count & CONST_ONE) == 0)
        {
            axis_count1 = axis_count;
            axis_count2 = 0;

            num_scalar_ops = (axis_count & (PDX_M - CONST_ONE));

            rem_inp_elms1 = num_scalar_ops * SIZE_OF_FLOAT;
            rem_out_elms1 = (num_scalar_ops >> CONST_ONE)
                    + (num_scalar_ops & CONST_ONE);

            leading_dims_limit = leading_dims;
            increment = CONST_ONE;
        }
        /* If axis_count is odd, number of elements in the
         * even rows is reduced by one and in the odd rows increased by one.
         */
        else
        {
            axis_count1 = axis_count - CONST_ONE;
            num_scalar_ops = (axis_count1 & (PDX_M - CONST_ONE));

            rem_inp_elms1 = num_scalar_ops * SIZE_OF_FLOAT;
            rem_out_elms1 = (num_scalar_ops >> CONST_ONE)
                    + (num_scalar_ops & CONST_ONE);

            axis_count2 = axis_count + CONST_ONE;
            num_scalar_ops = (axis_count2 & (PDX_M - CONST_ONE));

            rem_inp_elms2 = num_scalar_ops * SIZE_OF_FLOAT;
            rem_out_elms2 = (num_scalar_ops >> CONST_ONE)
                    + (num_scalar_ops & CONST_ONE);

            leading_dims_limit = (leading_dims >> CONST_ONE) +
                    (leading_dims & CONST_ONE);
            increment = CONST_TWO;
        }

        p_s = (xb_vecMxf32*) (p_out_scale);
        as = PDX_LA_MXF32_PP(p_s);


        /* If axis_count is even each row is stored sequentially.
         * If axis_count is odd all even rows are handled first and
         * then odd rows.
         */
        for (axis_index = 0; axis_index < (axis_count1 - CONST_THREE);
                axis_index += CONST_FOUR)
        {
            PDX_LA_MXF32_IP(s1, as, p_s);
            one_over_scale1 = PDX_DIV_MXF32(one, s1);

            for (leading_dims_index = 0;
                    leading_dims_index < leading_dims_limit;
                    leading_dims_index++)
            {
                p_x1 = (xb_vecMxf32*) (p_inp
                        + (leading_dims_index * length_per_step * increment)
                        + axis_index);
                ax1 = PDX_LA_MXF32_PP(p_x1);

                p_z1 = (xb_vecMx8*) (p_out
                        + (((leading_dims_index * length_per_step * increment)
                                + axis_index) >> CONST_ONE));

                PDX_LA_MXF32_IP(x, ax1, p_x1);

                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or,
                        PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az1, p_z1, CONST_TWO);
                PDX_SAPOS_MX8_FP(az1, p_z1);
            }

        }
        if((axis_count1 & CONST_THREE) != 0)
        {
            PDX_LAV_MXF32_XP(s1, as, p_s, rem_inp_elms1);
            one_over_scale1 = PDX_DIV_MXF32(one, s1);

            for (leading_dims_index = 0;
                    leading_dims_index < leading_dims_limit;
                    leading_dims_index++)
            {
                p_x1 = (xb_vecMxf32*) (p_inp
                        + (leading_dims_index * length_per_step * increment)
                        + axis_index);
                ax1 = PDX_LA_MXF32_PP(p_x1);

                p_z1 = (xb_vecMx8*) (p_out
                        + (((leading_dims_index * length_per_step * increment)
                                + axis_index) >> CONST_ONE));

                PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp_elms1);

                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or,
                        PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az1, p_z1, rem_out_elms1);
                PDX_SAPOS_MX8_FP(az1, p_z1);
            }
        }

        if (axis_count2 != 0)
        {
            /* Getting last scale value and three scale values from
             * base addresses into a register.
             */
            p_s = (xb_vecMxf32 *) (p_out_scale);
            as = PDX_LA_MXF32_PP(p_s);
            PDX_LAV_MXF32_XP(s1, as, p_s, CONST_THREE * SIZE_OF_FLOAT);

            FLOAT32 *p_s_end = p_out_scale + axis_count - CONST_ONE;
            xb_vecMxf32 s_end = p_s_end[0];
            s1 = PDX_SELI_MXF32(s1, s_end, PDX_SELI_32B_ROTATE_LEFT_1);
            one_over_scale1 = PDX_DIV_MXF32(one, s1);
         }

        /* When axis_count is odd below code handles the odd rows */
        for (axis_index = 0; axis_index < (axis_count2 - CONST_THREE);
                axis_index += CONST_FOUR)
        {
            for (leading_dims_index = CONST_ONE;
                    leading_dims_index < leading_dims;
                    leading_dims_index += CONST_TWO)
            {
                p_x1 = (xb_vecMxf32*) (p_inp
                        + (leading_dims_index * length_per_step) + axis_index
                        - CONST_ONE);
                ax1 = PDX_LA_MXF32_PP(p_x1);

                p_z1 = (xb_vecMx8*) (p_out
                        + (((leading_dims_index * length_per_step) + axis_index)
                                >> CONST_ONE));

                PDX_LA_MXF32_IP(x, ax1, p_x1);

                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or, PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az1, p_z1, CONST_TWO);
                PDX_SAPOS_MX8_FP(az1, p_z1);
            }

            PDX_LA_MXF32_IP(s1, as, p_s);
            one_over_scale1 = PDX_DIV_MXF32(one, s1);
        }
        if ((axis_count2 & CONST_THREE) != 0)
        {
            for (leading_dims_index = CONST_ONE;
                    leading_dims_index < leading_dims;
                    leading_dims_index += CONST_TWO)
            {
                p_x1 = (xb_vecMxf32*) (p_inp
                        + (leading_dims_index * length_per_step) + axis_index
                        - CONST_ONE);
                ax1 = PDX_LA_MXF32_PP(p_x1);

                p_z1 = (xb_vecMx8*) (p_out
                        + (((leading_dims_index * length_per_step) + axis_index)
                                >> CONST_ONE));

                PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp_elms2);

                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or,
                        PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az1, p_z1, rem_out_elms2);
                PDX_SAPOS_MX8_FP(az1, p_z1);
            }
        }
        if ((axis_count2 != 0) &&
                ((leading_dims & CONST_ONE) != 0))
        {

            p_z1 = (xb_vecMx8*) (p_out
                    + ((leading_dims * axis_count) >> CONST_ONE));

            FLOAT32 x_tmp = p_inp[(leading_dims * axis_count) - CONST_ONE];

            x_tmp = x_tmp / p_out_scale[axis_count - CONST_ONE];
            x_tmp = XT_FIRINT_S(x_tmp);
            WORD32 z_tmp = XT_TRUNC_S(x_tmp, 0);
            WORD32 clamped_tmp = XT_MIN(z_tmp, quant_max);
            clamped_tmp = XT_MAX(clamped_tmp, quant_min);

            even_nibbles = clamped_tmp << SHIFT_FACTOR_4_BIT;

            PDX_SAV32_MX8_XP(even_nibbles, az1, p_z1, CONST_ONE);
            PDX_SAPOS_MX8_FP(az1, p_z1);
        }
    }
    /* If axis is given and trailing_dims is even */
    else if (has_axis && ((trailing_dims & CONST_ONE) == 0))
    {
        WORD32 num_simd4_ops = (num_elm >> LOG2_PDX_M);
        WORD32 num_scalar_ops = (num_elm & (PDX_M - CONST_ONE));
        WORD32 rem_inp_elms = (num_scalar_ops * SIZE_OF_FLOAT);
        WORD32 rem_out_elms = (num_scalar_ops >> CONST_ONE)
                + (num_scalar_ops & CONST_ONE);

        WORD32 two_times_lps = CONST_TWO * length_per_step;
        vboolM bool = PDX_LTRSAT_BM(num_scalar_ops);

        for (axis_index = 0; axis_index < axis_count; axis_index++)
        {
            p_inp_base = p_inp + (axis_index * trailing_dims);
            p_out_base = p_out
                    + (axis_index * (trailing_dims >> CONST_ONE));

            s1 = p_out_scale[axis_index];

            one_over_scale1 = PDX_DIV_MXF32(one, s1);

            for (leading_dims_index = 0;
                    leading_dims_index < (leading_dims - CONST_ONE);
                    leading_dims_index += CONST_TWO)
            {

                p_x1 = (xb_vecMxf32*) (p_inp_base);
                ax1 = PDX_LA_MXF32_PP(p_x1);

                p_z1 = (xb_vecMx8 *) p_out_base;

                p_x2 = (xb_vecMxf32*) (p_inp_base + length_per_step);
                ax2 = PDX_LA_MXF32_PP(p_x2);

                p_z2 = (xb_vecMx8*) (p_out_base +
                        (length_per_step >> CONST_ONE));

                for (i = 0; i < num_simd4_ops; i++)
                {
                    PDX_LA_MXF32_IP(x, ax1, p_x1);
                    x = PDX_MUL_MXF32(x, one_over_scale1);
                    x = PDX_FIRINT_MXF32(x);
                    z = PDX_TRUNC32_MXF32(x, 0);
                    z = PDX_MIN_MX32(z, max);
                    z = PDX_MAX_MX32(z, min);

                    z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                    even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                    odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                    or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                    z = PDX_SELI_MX32(or, or,
                            PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                    PDX_SAV32_MX8_XP(z, az1, p_z1, CONST_TWO);

                    PDX_LA_MXF32_IP(x, ax2, p_x2);
                    x = PDX_MUL_MXF32(x, one_over_scale1);
                    x = PDX_FIRINT_MXF32(x);
                    z = PDX_TRUNC32_MXF32(x, 0);
                    z = PDX_MIN_MX32(z, max);
                    z = PDX_MAX_MX32(z, min);

                    z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                    even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                    odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                    or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                    z = PDX_SELI_MX32(or, or,
                            PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                    PDX_SAV32_MX8_XP(z, az2, p_z2, CONST_TWO);
                }
                PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp_elms);
                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                PDX_MIN_MX32_T(z, z, max, bool);
                PDX_MAX_MX32_T(z, z, min, bool);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or, PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az1, p_z1, rem_out_elms);

                PDX_SAPOS_MX8_FP(az1, p_z1);

                PDX_LAV_MXF32_XP(x, ax2, p_x2, rem_inp_elms);
                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                PDX_MIN_MX32_T(z, z, max, bool);
                PDX_MAX_MX32_T(z, z, min, bool);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or, PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az2, p_z2, rem_out_elms);

                PDX_SAPOS_MX8_FP(az2, p_z2);

                p_inp_base = p_inp_base + two_times_lps;
                p_out_base = p_out_base + (two_times_lps >> CONST_ONE);
            }

            if ((leading_dims & CONST_ONE) != 0)
            {
                p_x1 = (xb_vecMxf32*) (p_inp_base);
                ax1 = PDX_LA_MXF32_PP(p_x1);
                p_z1 = (xb_vecMx8 *) p_out_base;

                for (i = 0; i < num_simd4_ops; i++)
                {
                    PDX_LA_MXF32_IP(x, ax1, p_x1);
                    x = PDX_MUL_MXF32(x, one_over_scale1);
                    x = PDX_FIRINT_MXF32(x);
                    z = PDX_TRUNC32_MXF32(x, 0);
                    z = PDX_MIN_MX32(z, max);
                    z = PDX_MAX_MX32(z, min);

                    z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                    even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                    odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                    or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                    z = PDX_SELI_MX32(or, or,
                            PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                    PDX_SAV32_MX8_XP(z, az1, p_z1, CONST_TWO);
                }
                PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp_elms);
                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                PDX_MIN_MX32_T(z, z, max, bool);
                PDX_MAX_MX32_T(z, z, min, bool);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or, PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az1, p_z1, rem_out_elms);

                PDX_SAPOS_MX8_FP(az1, p_z1);
            }
        }
    }
    /* If axis is not given or trailing_dims is odd */
    else
    {
        FLOAT32 *p_scale = p_out_scale;

        circb cs = PDX_MOVC_AU32D((UWORD32) (p_scale + axis_count),
                (UWORD32) p_scale);

        WORD32 num_simd4_ops = (num_elm >> LOG2_PDX_M);
        WORD32 num_scalar_ops = (num_elm & (PDX_M - CONST_ONE));
        WORD32 rem_inp_elms = (num_scalar_ops * SIZE_OF_FLOAT);
        WORD32 rem_out_elms = (num_scalar_ops >> CONST_ONE)
                + (num_scalar_ops & CONST_ONE);

        vboolM bool = PDX_LTRSAT_BM(num_scalar_ops);

        WORD32 step_by_n_bytes, pointer_step = 0;
        if (axis_count == CONST_ONE)
        {
            step_by_n_bytes = CONST_FOUR;
        }
        else
        {
            step_by_n_bytes = STEP_BY_8_BYTES;
            pointer_step = CONST_ONE;
        }

        xb_vecMxf32 s2, one_over_scale2;

        WORD32 chunks = (leading_dims * axis_count);

        for (leading_dims_index = 0;
                leading_dims_index < (chunks - CONST_TWO);
                leading_dims_index += CONST_FOUR)
        {
            PDX_LSR_F32_XC(s1, p_scale, step_by_n_bytes, cs);
            p_x1 = (xb_vecMxf32*) (p_inp +
                    (leading_dims_index * trailing_dims));
            ax1 = PDX_LA_MXF32_PP(p_x1);

            one_over_scale1 = PDX_DIV_MXF32(one, s1);

            p_z1 = (xb_vecMx8*) (p_out
                    + ((leading_dims_index * trailing_dims) >> CONST_ONE)
                    + ((leading_dims_index * trailing_dims) & CONST_ONE));

            PDX_LSR_F32_XC(s2, p_scale, step_by_n_bytes, cs);
            p_x2 = (xb_vecMxf32*) (p_inp +
                    ((leading_dims_index + CONST_TWO) * trailing_dims));
            ax2 = PDX_LA_MXF32_PP(p_x2);

            one_over_scale2 = PDX_DIV_MXF32(one, s2);

            p_z2 = (xb_vecMx8*) (p_out
                    + (((leading_dims_index + CONST_TWO) * trailing_dims)
                            >> CONST_ONE)
                    + (((leading_dims_index + CONST_TWO) * trailing_dims)
                            & CONST_ONE));

            for (i = 0; i < num_simd4_ops; i++)
            {
                PDX_LA_MXF32_IP(x, ax1, p_x1);
                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or,
                        PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az1, p_z1, CONST_TWO);

                PDX_LA_MXF32_IP(x, ax2, p_x2);
                x = PDX_MUL_MXF32(x, one_over_scale2);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or,
                        PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az2, p_z2, CONST_TWO);

            }
            PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp_elms);
            x = PDX_MUL_MXF32(x, one_over_scale1);
            x = PDX_FIRINT_MXF32(x);
            z = PDX_TRUNC32_MXF32(x, 0);
            PDX_MIN_MX32_T(z, z, max, bool);
            PDX_MAX_MX32_T(z, z, min, bool);

            z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
            even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
            odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
            or = PDX_OR_MX32(even_nibbles, odd_nibbles);

            z = PDX_SELI_MX32(or, or, PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

            PDX_SAV32_MX8_XP(z, az1, p_z1, rem_out_elms);

            PDX_SAPOS_MX8_FP(az1, p_z1);

            PDX_LAV_MXF32_XP(x, ax2, p_x2, rem_inp_elms);
            x = PDX_MUL_MXF32(x, one_over_scale2);
            x = PDX_FIRINT_MXF32(x);
            z = PDX_TRUNC32_MXF32(x, 0);
            PDX_MIN_MX32_T(z, z, max, bool);
            PDX_MAX_MX32_T(z, z, min, bool);

            z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
            even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
            odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
            or = PDX_OR_MX32(even_nibbles, odd_nibbles);

            z = PDX_SELI_MX32(or, or, PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

            PDX_SAV32_MX8_XP(z, az2, p_z2, rem_out_elms);

            PDX_SAPOS_MX8_FP(az2, p_z2);

        }
        if (((chunks & CONST_THREE) == CONST_ONE)
                || ((chunks & CONST_THREE) == CONST_TWO))
        {
            PDX_LSR_F32_XC(s1, p_scale, step_by_n_bytes, cs);
            p_x1 = (xb_vecMxf32*) (p_inp +
                    (leading_dims_index * trailing_dims));
            ax1 = PDX_LA_MXF32_PP(p_x1);

            one_over_scale1 = PDX_DIV_MXF32(one, s1);

            p_z1 = (xb_vecMx8*) (p_out
                    + ((leading_dims_index * trailing_dims) >> CONST_ONE)
                    + ((leading_dims_index * trailing_dims) & CONST_ONE));

            for (i = 0; i < num_simd4_ops; i++)
            {
                PDX_LA_MXF32_IP(x, ax1, p_x1);
                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or,
                        PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az1, p_z1, CONST_TWO);
            }
            PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp_elms);
            x = PDX_MUL_MXF32(x, one_over_scale1);
            x = PDX_FIRINT_MXF32(x);
            z = PDX_TRUNC32_MXF32(x, 0);
            PDX_MIN_MX32_T(z, z, max, bool);
            PDX_MAX_MX32_T(z, z, min, bool);

            z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
            even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
            odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
            or = PDX_OR_MX32(even_nibbles, odd_nibbles);

            z = PDX_SELI_MX32(or, or, PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

            PDX_SAV32_MX8_XP(z, az1, p_z1, rem_out_elms);

            PDX_SAPOS_MX8_FP(az1, p_z1);
        }

        rem_inp = rem_inp_elms - CONST_FOUR;

        rem_out = rem_out_elms - CONST_ONE;

        bool = PDX_LTRSAT_BM(num_scalar_ops - CONST_ONE);

        p_scale = (p_out_scale + pointer_step);

        for (leading_dims_index = CONST_ONE;
                leading_dims_index < (chunks - CONST_TWO);
                leading_dims_index += CONST_FOUR)
        {
            PDX_LSR_F32_XC(s1, p_scale, step_by_n_bytes, cs);
            p_x1 = (xb_vecMxf32*) (p_inp
                    + (leading_dims_index * trailing_dims));
            ax1 = PDX_LA_MXF32_PP(p_x1);

            one_over_scale1 = PDX_DIV_MXF32(one, s1);

            p_z1 = (xb_vecMx8*) (p_out
                    + ((leading_dims_index * trailing_dims) >> CONST_ONE)
                    + ((leading_dims_index * trailing_dims) & CONST_ONE));

            p_z_nibble = (WORD8*) p_z1 - CONST_ONE;
            PDX_LAV_MXF32_XP(x, ax1, p_x1, CONST_FOUR);
            x = PDX_MUL_MXF32(x, one_over_scale1);
            x = PDX_FIRINT_MXF32(x);
            z = PDX_TRUNC32_MXF32(x, 0);
            z = PDX_MIN_MX32(z, max);
            z = PDX_MAX_MX32(z, min);

            z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
            odd_nibbles = p_z_nibble[0];
            z = PDX_OR_MX32(z, odd_nibbles);

            PDX_SS32_8_IP(z, p_z_nibble, CONST_ONE);

            PDX_LSR_F32_XC(s2, p_scale, step_by_n_bytes, cs);
            p_x2 = (xb_vecMxf32*) (p_inp +
                    ((leading_dims_index + CONST_TWO) * trailing_dims));
            ax2 = PDX_LA_MXF32_PP(p_x2);

            one_over_scale2 = PDX_DIV_MXF32(one, s2);

            p_z2 = (xb_vecMx8*) (p_out
                    + (((leading_dims_index + CONST_TWO) * trailing_dims)
                            >> CONST_ONE)
                    + (((leading_dims_index + CONST_TWO) * trailing_dims)
                            & CONST_ONE));

            p_z_nibble = (WORD8*) p_z2 - CONST_ONE;
            PDX_LAV_MXF32_XP(x, ax2, p_x2, CONST_FOUR);
            x = PDX_MUL_MXF32(x, one_over_scale2);
            x = PDX_FIRINT_MXF32(x);
            z = PDX_TRUNC32_MXF32(x, 0);
            z = PDX_MIN_MX32(z, max);
            z = PDX_MAX_MX32(z, min);

            z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
            odd_nibbles = p_z_nibble[0];
            z = PDX_OR_MX32(z, odd_nibbles);

            PDX_SS32_8_IP(z, p_z_nibble, CONST_ONE);

            for (i = 0; i < num_simd4_ops; i++)
            {
                PDX_LA_MXF32_IP(x, ax1, p_x1);
                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or,
                        PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az1, p_z1, CONST_TWO);

                PDX_LA_MXF32_IP(x, ax2, p_x2);
                x = PDX_MUL_MXF32(x, one_over_scale2);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or,
                        PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az2, p_z2, CONST_TWO);
            }
            PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp);
            x = PDX_MUL_MXF32(x, one_over_scale1);
            x = PDX_FIRINT_MXF32(x);
            z = PDX_TRUNC32_MXF32(x, 0);
            PDX_MIN_MX32_T(z, z, max, bool);
            PDX_MAX_MX32_T(z, z, min, bool);

            z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
            even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
            odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
            or = PDX_OR_MX32(even_nibbles, odd_nibbles);

            z = PDX_SELI_MX32(or, or, PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

            PDX_SAV32_MX8_XP(z, az1, p_z1, rem_out);

            PDX_SAPOS_MX8_FP(az1, p_z1);

            PDX_LAV_MXF32_XP(x, ax2, p_x2, rem_inp);
            x = PDX_MUL_MXF32(x, one_over_scale2);
            x = PDX_FIRINT_MXF32(x);
            z = PDX_TRUNC32_MXF32(x, 0);
            PDX_MIN_MX32_T(z, z, max, bool);
            PDX_MAX_MX32_T(z, z, min, bool);

            z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
            even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
            odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
            or = PDX_OR_MX32(even_nibbles, odd_nibbles);

            z = PDX_SELI_MX32(or, or, PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

            PDX_SAV32_MX8_XP(z, az2, p_z2, rem_out);

            PDX_SAPOS_MX8_FP(az2, p_z2);
        }
        if (((chunks & CONST_THREE) == CONST_THREE)
                || ((chunks & CONST_THREE) == CONST_TWO))
        {
            PDX_LSR_F32_XC(s1, p_scale, step_by_n_bytes, cs);
            p_x1 = (xb_vecMxf32*) (p_inp
                    + (leading_dims_index * trailing_dims));
            ax1 = PDX_LA_MXF32_PP(p_x1);

            one_over_scale1 = PDX_DIV_MXF32(one, s1);

            p_z1 = (xb_vecMx8*) (p_out
                    + ((leading_dims_index * trailing_dims) >> CONST_ONE)
                    + ((leading_dims_index * trailing_dims) & CONST_ONE));

            p_z_nibble = (WORD8*) p_z1 - CONST_ONE;
            PDX_LAV_MXF32_XP(x, ax1, p_x1, CONST_FOUR);
            x = PDX_MUL_MXF32(x, one_over_scale1);
            x = PDX_FIRINT_MXF32(x);
            z = PDX_TRUNC32_MXF32(x, 0);
            z = PDX_MIN_MX32(z, max);
            z = PDX_MAX_MX32(z, min);

            z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
            odd_nibbles = p_z_nibble[0];
            z = PDX_OR_MX32(z, odd_nibbles);

            PDX_SS32_8_IP(z, p_z_nibble, CONST_ONE);

            for (i = 0; i < num_simd4_ops; i++)
            {
                PDX_LA_MXF32_IP(x, ax1, p_x1);
                x = PDX_MUL_MXF32(x, one_over_scale1);
                x = PDX_FIRINT_MXF32(x);
                z = PDX_TRUNC32_MXF32(x, 0);
                z = PDX_MIN_MX32(z, max);
                z = PDX_MAX_MX32(z, min);

                z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
                even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
                odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
                or = PDX_OR_MX32(even_nibbles, odd_nibbles);

                z = PDX_SELI_MX32(or, or,
                        PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

                PDX_SAV32_MX8_XP(z, az1, p_z1, CONST_TWO);
            }
            PDX_LAV_MXF32_XP(x, ax1, p_x1, rem_inp);
            x = PDX_MUL_MXF32(x, one_over_scale1);
            x = PDX_FIRINT_MXF32(x);
            z = PDX_TRUNC32_MXF32(x, 0);
            PDX_MIN_MX32_T(z, z, max, bool);
            PDX_MAX_MX32_T(z, z, min, bool);

            z = PDX_AND_MX32(z, MASK_HIGHER_NIBBLE);
            even_nibbles = PDX_SLLI_MX32(z, SHIFT_FACTOR_4_BIT);
            odd_nibbles = PDX_SHFLI_MX32(z, PDX_SHFLI_32B_SWAP_1);
            or = PDX_OR_MX32(even_nibbles, odd_nibbles);

            z = PDX_SELI_MX32(or, or, PDX_SELI_32B_EXTRACT_1_OF_2_OFF_0);

            PDX_SAV32_MX8_XP(z, az1, p_z1, rem_out);

            PDX_SAPOS_MX8_FP(az1, p_z1);
        }
    }
#endif

    /* Resetting the original scf */
    PDX_MOVSCF_32(actual_scf);

    return 0;
}
