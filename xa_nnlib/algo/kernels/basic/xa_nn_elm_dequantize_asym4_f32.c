/******************************************************************************
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

WORD32 xa_nn_elm_dequantize_asym4_f32(FLOAT32 *__restrict__ p_out,
        const WORD8 *__restrict__ p_inp,
        const WORD32 *const p_inp_shape,
        WORD32 num_inp_dims,
        WORD32 *p_axis,
        WORD32 *p_inp_zero_bias,
        FLOAT32 *p_inp_scale)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_zero_bias, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_scale, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_zero_bias, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_scale, sizeof(FLOAT32), UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_inp_dims <= 0), UNSUPPORTED_PARAM);

    WORD32 i, axis_index;
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

    WORD32 total_num_elm = CONST_ONE;

    /* Calculating total number of input elements */
    for (i = 0; i < num_inp_dims; i++)
    {
        total_num_elm *= p_inp_shape[i];
    }

    /* Flag to check whether axis is given or not */
    WORD32 has_axis = 0;

    if (p_axis == NULL)
    {
        XA_NNLIB_ARG_CHK_COND(((isnan(*p_inp_scale)) || (isinf(*p_inp_scale))),
                UNSUPPORTED_PARAM);

        num_elm = total_num_elm;
    }
    else
    {
        WORD32 axis = *p_axis;

        /* Invalid input checks */
        /* axis should be in the range [0,num_inp_dims-1] */
        XA_NNLIB_ARG_CHK_COND(((axis < 0) || (axis >= num_inp_dims)),
                UNSUPPORTED_PARAM);

        for (i = 0; i < p_inp_shape[*p_axis]; i++)
        {
            XA_NNLIB_ARG_CHK_COND(
                    ((isnan(p_inp_scale[i])) || (isinf(p_inp_scale[i]))),
                    UNSUPPORTED_PARAM);
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
         * elements dequantized with a scale and zero_bias values to get
         * the next base addresses.
         */
        length_per_step = p_inp_shape[axis] * trailing_dims;

        /* Length of the dimension along axis */
        axis_count = p_inp_shape[axis];

        has_axis = 1;
    }

    /* Pointers for base address */
    const WORD8 * p_inp_base;
    const FLOAT32 * p_out_base;

    /* Vector pointers for the base pointers */
    const xb_vecMx8 * p_x1;
    xb_vecMxf32 * __restrict__ p_z1;

    const xb_vecMx8 * p_x2;
    xb_vecMxf32 *__restrict__ p_z2;

    WORD32 leading_dim_idx, num_rem_inps, num_rem_outs;

    /* Align registers */
    valign ax1, az1, ax2, az2;
    az1 = PDX_Z_ALIGN();
    az2 = PDX_Z_ALIGN();

    /* Variables to hold input and zero_bias values */
    xb_vecMx32 x, zb1;
    /* Variables to hold output and scale values */
    xb_vecMxf32 z, s1;

#ifndef ENABLE_4BIT_PACK /* Code for bytes store (unpacked) */
    WORD32 two_times_lps = CONST_TWO * length_per_step;

    /* When axis is given and is also last dimension */
    if (has_axis && (*p_axis == (num_inp_dims - CONST_ONE)))
    {
        valign as, azb;

        xb_vecMxf32 *__restrict__ p_s = (xb_vecMxf32*) p_inp_scale;
        as = PDX_LA_MXF32_PP(p_s);

        xb_vecMx32 *__restrict__ p_zb = (xb_vecMx32*) p_inp_zero_bias;
        azb = PDX_LA_MX32_PP(p_zb);

        num_rem_inps = (length_per_step & (PDX_M - CONST_ONE));
        num_rem_outs = (num_rem_inps * SIZE_OF_FLOAT);

        for (axis_index = 0; axis_index < (axis_count - CONST_THREE);
                axis_index += CONST_FOUR)
        {
            PDX_LA_MXF32_IP(s1, as, p_s);
            PDX_LA_MX32_IP(zb1, azb, p_zb);
            for (leading_dim_idx = 0;
                    leading_dim_idx < (leading_dims);
                    leading_dim_idx++)
            {
                p_x1 = (xb_vecMx8*) (p_inp
                        + (leading_dim_idx * length_per_step) + axis_index);
                ax1 = PDX_LA_MX8_PP(p_x1);

                p_z1 = (xb_vecMxf32*) (p_out
                        + (leading_dim_idx * length_per_step) + axis_index);

                PDX_LA32_MX8_IP(x, ax1, p_x1);
                x = PDX_SRAI_MX32(x, SHIFT_FACTOR_4_BIT);

                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SA_MXF32_IP(z, az1, p_z1);
                PDX_SAPOS_MXF32_FP(az1, p_z1);
            }
        }
        /* Remaining scale and zero_bias values are loaded */
        if((axis_count & CONST_THREE) != 0)
        {
            PDX_LAV_MXF32_XP(s1, as, p_s, num_rem_outs);
            PDX_LAV_MX32_XP(zb1, azb, p_zb, num_rem_outs);

            for (leading_dim_idx = 0;
                    leading_dim_idx < leading_dims;
                    leading_dim_idx++)
            {
                p_x1 = (xb_vecMx8*) (p_inp
                        + (leading_dim_idx * length_per_step) + axis_index);
                ax1 = PDX_LA_MX8_PP(p_x1);

                p_z1 = (xb_vecMxf32*) (p_out
                        + (leading_dim_idx * length_per_step) + axis_index);

                PDX_LAV32_MX8_XP(x, ax1, p_x1, num_rem_inps);
                x = PDX_SRAI_MX32(x, SHIFT_FACTOR_4_BIT);

                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SAV_MXF32_XP(z, az1, p_z1, num_rem_outs);
                PDX_SAPOS_MXF32_FP(az1, p_z1);
            }
        }
    }
    /* When axis is not given or axis is not last dim */
    else
    {
        num_rem_inps = (num_elm & (PDX_M - CONST_ONE));
        num_rem_outs = (num_rem_inps * SIZE_OF_FLOAT);
        for (axis_index = 0; axis_index < axis_count; axis_index++)
        {
            s1 = (p_inp_scale[axis_index] / SCALE_FACTOR_4_BIT);
            zb1 = (p_inp_zero_bias[axis_index] << SHIFT_FACTOR_4_BIT);
            p_inp_base = p_inp + (axis_index * trailing_dims);
            p_out_base = p_out + (axis_index * trailing_dims);

            /* This loop iterates over the leading dims.
             * All the elements are quantized at a time for
             * single scale and zero_bias once loaded
             */
            for (leading_dim_idx = 0;
                    leading_dim_idx < (leading_dims - CONST_ONE);
                    leading_dim_idx += CONST_TWO)
            {
                p_x1 = (const xb_vecMx8*) p_inp_base;
                ax1 = PDX_LA_MX8_PP(p_x1);
                p_z1 = (xb_vecMxf32*) p_out_base;

                p_x2 = (const xb_vecMx8*) (p_inp_base + length_per_step);
                ax2 = PDX_LA_MX8_PP(p_x2);
                p_z2 = (xb_vecMxf32*) (p_out_base + length_per_step);

                for (i = 0; i < (num_elm >> LOG2_PDX_M); i++)
                {
                    PDX_LA32_MX8_IP(x, ax1, p_x1);
                    x = PDX_SUB_MX32(x, zb1);
                    z = PDX_MUL_MXF32(x, s1);
                    PDX_SA_MXF32_IP(z, az1, p_z1);

                    PDX_LA32_MX8_IP(x, ax2, p_x2);
                    x = PDX_SUB_MX32(x, zb1);
                    z = PDX_MUL_MXF32(x, s1);
                    PDX_SA_MXF32_IP(z, az2, p_z2);
                }
                PDX_LAV32_MX8_XP(x, ax1, p_x1, num_rem_inps);
                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SAV_MXF32_XP(z, az1, p_z1, num_rem_outs);
                PDX_SAPOS_MXF32_FP(az1, p_z1);

                PDX_LAV32_MX8_XP(x, ax2, p_x2, num_rem_inps);
                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SAV_MXF32_XP(z, az2, p_z2, num_rem_outs);
                PDX_SAPOS_MXF32_FP(az2, p_z2);

                p_inp_base = p_inp_base + two_times_lps;
                p_out_base = p_out_base + two_times_lps;
            }
            if ((leading_dims & CONST_ONE) != 0)
            {
                p_x1 = (const xb_vecMx8*) p_inp_base;
                ax1 = PDX_LA_MX8_PP(p_x1);
                p_z1 = (xb_vecMxf32*) p_out_base;

                for (i = 0; i < (num_elm >> LOG2_PDX_M); i++)
                {
                    PDX_LA32_MX8_IP(x, ax1, p_x1);
                    x = PDX_SUB_MX32(x, zb1);
                    z = PDX_MUL_MXF32(x, s1);
                    PDX_SA_MXF32_IP(z, az1, p_z1);
                }
                PDX_LAV32_MX8_XP(x, ax1, p_x1, num_rem_inps);
                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SAV_MXF32_XP(z, az1, p_z1, num_rem_outs);
                PDX_SAPOS_MXF32_FP(az1, p_z1);
            }
        }
    }
/* Code for Bits4x2 */
#else
    /* Number of chunks of size 16 */
    WORD32 num_simd_nibbles = (total_num_elm >> PDX_M);
    /* Remaining number of elements */
    WORD32 num_rem_nibbles = (total_num_elm & (PDX_4M - CONST_ONE));

    /* Number of bytes required to pack remaining elements */
    WORD32 offset = (num_rem_nibbles >> CONST_ONE)
            + (num_rem_nibbles & CONST_ONE);

    /* Pointer pointing to starting address of input in output buffer */
    WORD8 * p_scratch_buff = (WORD8*) p_out + (total_num_elm * SIZE_OF_FLOAT)
            - total_num_elm;

    xb_vec4Mx8 x0, odd_nibble, even_nibble, z0;
    xb_vec4Mx8 *__restrict__ p_x0 = (xb_vec4Mx8*) p_inp;
    valign ax0 = PDX_LA_4MX8_PP(p_x0);
    xb_vec4Mx8 * p_z0 = (xb_vec4Mx8*) p_scratch_buff;
    valign az0 = PDX_Z_ALIGN();

    /* Code for unpacking nibbles to bytes.
     * Unpacked nibbles are left justified by 4 bits
     * before storing to a byte.
     */
    for (i = 0; i < num_simd_nibbles; i++)
    {
        PDX_LAV_4MX8_XP(x0, ax0, p_x0, CONST_EIGHT);
        odd_nibble = PDX_SLLI_4MX8(x0, SHIFT_FACTOR_4_BIT);
        even_nibble = PDX_AND_4MX8(x0, MASK_LOWER_NIBBLE);
        z0 = PDX_SELI_4MX8(odd_nibble, even_nibble,
                PDX_SELI_8B_INTERLEAVE_1_LO);
        PDX_SA_4MX8_IP(z0, az0, p_z0);
    }

    PDX_LAV_4MX8_XP(x0, ax0, p_x0, offset);

    odd_nibble = PDX_SLLI_4MX8(x0, SHIFT_FACTOR_4_BIT);

    even_nibble = PDX_AND_4MX8(x0, MASK_LOWER_NIBBLE);

    z0 = PDX_SELI_4MX8(odd_nibble, even_nibble,
            PDX_SELI_8B_INTERLEAVE_1_LO);

    PDX_SAV_4MX8_XP(z0, az0, p_z0, num_rem_nibbles);
    PDX_SAPOS_4MX8_FP(az0, p_z0);

    xb_vecMxf32 s2;
    xb_vecMx32 zb2;

    /* When axis is given and is also last dimension */
    if (has_axis && (*p_axis == (num_inp_dims - CONST_ONE)))
    {
        valign as, azb;

        xb_vecMxf32 *__restrict__ p_s = (xb_vecMxf32*) p_inp_scale;
        as = PDX_LA_MXF32_PP(p_s);

        xb_vecMx32 *__restrict__ p_zb = (xb_vecMx32*) p_inp_zero_bias;
        azb = PDX_LA_MX32_PP(p_zb);

        WORD32 num_simd4_ops = (length_per_step >> LOG2_PDX_M);
        num_rem_inps = (length_per_step & (PDX_M - CONST_ONE));
        num_rem_outs = (num_rem_inps * SIZE_OF_FLOAT);

        /* Number of times leading_dim_idx loop should be
         * iterated to avoid overwriting of inputs in
         * scratch buffer by outputs.
         */
        WORD32 initial_lead_dims = (total_num_elm * CONST_THREE)
                / (length_per_step * SIZE_OF_FLOAT);

        /* Four scale values and zero_bias values are
         *  loaded at a time column wise.
         */
        for (axis_index = 0; axis_index < (axis_count - CONST_THREE);
                axis_index += CONST_FOUR)
        {
            PDX_LA_MXF32_IP(s1, as, p_s);
            PDX_LA_MX32_IP(zb1, azb, p_zb);

            /* Loaded scale and zero_bias values are used for
             * initial_lead_dims number of leading_dims out of total
             * number of leading_dims.
             */
            for (leading_dim_idx = 0;
                    leading_dim_idx < initial_lead_dims;
                    leading_dim_idx++)
            {
                p_x1 = (xb_vecMx8*) (p_scratch_buff
                        + (leading_dim_idx * length_per_step) + axis_index);
                ax1 = PDX_LA_MX8_PP(p_x1);

                p_z1 = (xb_vecMxf32*) (p_out
                        + (leading_dim_idx * length_per_step) + axis_index);

                PDX_LA32_MX8_IP(x, ax1, p_x1);
                x = PDX_SRAI_MX32(x, SHIFT_FACTOR_4_BIT);

                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SA_MXF32_IP(z, az1, p_z1);
                PDX_SAPOS_MXF32_FP(az1, p_z1);
            }
        }
        /* Remaining scale and zero_bias values are loaded */
        if((axis_count & CONST_THREE) != 0)
        {
            PDX_LAV_MXF32_XP(s1, as, p_s, num_rem_outs);
            PDX_LAV_MX32_XP(zb1, azb, p_zb, num_rem_outs);

            /* Loaded scale and zero_bias values are used for
             * initial_lead_dims number of leading_dims out of total
             * number of leading_dims.
             */
            for (leading_dim_idx = 0;
                    leading_dim_idx < initial_lead_dims;
                    leading_dim_idx++)
            {
                p_x1 = (xb_vecMx8*) (p_scratch_buff
                        + (leading_dim_idx * length_per_step) + axis_index);
                ax1 = PDX_LA_MX8_PP(p_x1);

                p_z1 = (xb_vecMxf32*) (p_out
                        + (leading_dim_idx * length_per_step) + axis_index);

                PDX_LAV32_MX8_XP(x, ax1, p_x1, num_rem_inps);
                x = PDX_SRAI_MX32(x, SHIFT_FACTOR_4_BIT);

                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SAV_MXF32_XP(z, az1, p_z1, num_rem_outs);
                PDX_SAPOS_MXF32_FP(az1, p_z1);
            }
        }

        p_x1 = (xb_vecMx8*) (p_scratch_buff
                + (leading_dim_idx * length_per_step));
        ax1 = PDX_LA_MX8_PP(p_x1);

        p_z1 = (xb_vecMxf32*) (p_out
                + (leading_dim_idx * length_per_step));
        /* Handling remaining leading_dims row wise. */
        for (; leading_dim_idx < leading_dims;
                leading_dim_idx++)
        {
            /* For every iteration, scale and zero_bias are
             * loaded from starting of base addresses.
             */
            p_s = (xb_vecMxf32*) p_inp_scale;
            as = PDX_LA_MXF32_PP(p_s);

            p_zb = (xb_vecMx32*) p_inp_zero_bias;
            azb = PDX_LA_MX32_PP(p_zb);

            PDX_LAV32_MX8_XP(x, ax1, p_x1, num_rem_inps);
            PDX_LAV_MXF32_XP(s1, as, p_s, num_rem_outs);
            PDX_LAV_MX32_XP(zb1, azb, p_zb, num_rem_outs);

            x = PDX_SRAI_MX32(x, SHIFT_FACTOR_4_BIT);

            x = PDX_SUB_MX32(x, zb1);
            z = PDX_MUL_MXF32(x, s1);
            PDX_SAV_MXF32_XP(z, az1, p_z1, num_rem_outs);

            for (i = 0; i < num_simd4_ops; i++)
            {
                PDX_LA32_MX8_IP(x, ax1, p_x1);
                PDX_LA_MXF32_IP(s1, as, p_s);
                PDX_LA_MX32_IP(zb1, azb, p_zb);

                x = PDX_SRAI_MX32(x, SHIFT_FACTOR_4_BIT);

                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SA_MXF32_IP(z, az1, p_z1);
            }
        }
        PDX_SAPOS_MXF32_FP(az1, p_z1);
    }
    /* When axis is not given or axis is not last dim */
    else
    {
        num_rem_inps = (num_elm & (PDX_M - CONST_ONE));
        num_rem_outs = (num_rem_inps * SIZE_OF_FLOAT);

        for (leading_dim_idx = 0; leading_dim_idx < (leading_dims - CONST_ONE);
                leading_dim_idx++)
        {
            p_inp_base = p_scratch_buff + (leading_dim_idx * length_per_step);
            p_out_base = p_out + (leading_dim_idx * length_per_step);

            for (axis_index = 0; axis_index < (axis_count - CONST_ONE);
                    axis_index += CONST_TWO)
            {
                s1 = (p_inp_scale[axis_index] / SCALE_FACTOR_4_BIT);
                zb1 = (p_inp_zero_bias[axis_index] << SHIFT_FACTOR_4_BIT);

                p_x1 = (const xb_vecMx8*) (p_inp_base);
                ax1 = PDX_LA_MX8_PP(p_x1);
                p_z1 = (xb_vecMxf32*) p_out_base;

                s2 = (p_inp_scale[axis_index + CONST_ONE] / SCALE_FACTOR_4_BIT);
                zb2 = (p_inp_zero_bias[axis_index + CONST_ONE]
                            << SHIFT_FACTOR_4_BIT);

                p_x2 = (const xb_vecMx8*) (p_inp_base + trailing_dims);
                ax2 = PDX_LA_MX8_PP(p_x2);
                p_z2 = (xb_vecMxf32*) (p_out_base + trailing_dims);

                for (i = 0; i < (num_elm >> LOG2_PDX_M); i++)
                {
                    PDX_LA32_MX8_IP(x, ax1, p_x1);
                    x = PDX_SUB_MX32(x, zb1);
                    z = PDX_MUL_MXF32(x, s1);
                    PDX_SA_MXF32_IP(z, az1, p_z1);

                    PDX_LA32_MX8_IP(x, ax2, p_x2);
                    x = PDX_SUB_MX32(x, zb2);
                    z = PDX_MUL_MXF32(x, s2);
                    PDX_SA_MXF32_IP(z, az2, p_z2);
                }
                PDX_LAV32_MX8_XP(x, ax1, p_x1, num_rem_inps);
                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SAV_MXF32_XP(z, az1, p_z1, num_rem_outs);
                PDX_SAPOS_MXF32_FP(az1, p_z1);

                PDX_LAV32_MX8_XP(x, ax2, p_x2, num_rem_inps);
                x = PDX_SUB_MX32(x, zb2);
                z = PDX_MUL_MXF32(x, s2);
                PDX_SAV_MXF32_XP(z, az2, p_z2, num_rem_outs);
                PDX_SAPOS_MXF32_FP(az2, p_z2);

                p_inp_base = p_inp_base + (CONST_TWO * trailing_dims);
                p_out_base = p_out_base + (CONST_TWO * trailing_dims);
            }
            if ((axis_count & CONST_ONE) != 0)
            {
                s1 = (p_inp_scale[axis_index] / SCALE_FACTOR_4_BIT);
                zb1 = (p_inp_zero_bias[axis_index] << SHIFT_FACTOR_4_BIT);

                p_x1 = (const xb_vecMx8*) (p_inp_base);
                ax1 = PDX_LA_MX8_PP(p_x1);
                p_z1 = (xb_vecMxf32*) p_out_base;

                for (i = 0; i < (num_elm >> LOG2_PDX_M); i++)
                {
                    PDX_LA32_MX8_IP(x, ax1, p_x1);
                    x = PDX_SUB_MX32(x, zb1);
                    z = PDX_MUL_MXF32(x, s1);
                    PDX_SA_MXF32_IP(z, az1, p_z1);
                }
                PDX_LAV32_MX8_XP(x, ax1, p_x1, num_rem_inps);
                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SAV_MXF32_XP(z, az1, p_z1, num_rem_outs);
                PDX_SAPOS_MXF32_FP(az1, p_z1);

            }
        } /* leading_dim_idx */
        p_inp_base = p_scratch_buff + (leading_dim_idx * length_per_step);
        p_out_base = p_out + (leading_dim_idx * length_per_step);

        for (axis_index = 0; axis_index < (axis_count - CONST_THREE);
                axis_index += CONST_TWO)
        {
            s1 = (p_inp_scale[axis_index] / SCALE_FACTOR_4_BIT);
            zb1 = (p_inp_zero_bias[axis_index] << SHIFT_FACTOR_4_BIT);

            p_x1 = (const xb_vecMx8*) (p_inp_base);
            ax1 = PDX_LA_MX8_PP(p_x1);
            p_z1 = (xb_vecMxf32*) p_out_base;

            s2 = (p_inp_scale[axis_index + CONST_ONE] / SCALE_FACTOR_4_BIT);
            zb2 = (p_inp_zero_bias[axis_index + CONST_ONE]
                            << SHIFT_FACTOR_4_BIT);

            p_x2 = (const xb_vecMx8*) (p_inp_base + trailing_dims);
            ax2 = PDX_LA_MX8_PP(p_x2);
            p_z2 = (xb_vecMxf32*) (p_out_base + trailing_dims);

            for (i = 0; i < (num_elm >> LOG2_PDX_M); i++)
            {
                PDX_LA32_MX8_IP(x, ax1, p_x1);
                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SA_MXF32_IP(z, az1, p_z1);

                PDX_LA32_MX8_IP(x, ax2, p_x2);
                x = PDX_SUB_MX32(x, zb2);
                z = PDX_MUL_MXF32(x, s2);
                PDX_SA_MXF32_IP(z, az2, p_z2);
            }
            PDX_LAV32_MX8_XP(x, ax1, p_x1, num_rem_inps);
            x = PDX_SUB_MX32(x, zb1);
            z = PDX_MUL_MXF32(x, s1);
            PDX_SAV_MXF32_XP(z, az1, p_z1, num_rem_outs);
            PDX_SAPOS_MXF32_FP(az1, p_z1);

            PDX_LAV32_MX8_XP(x, ax2, p_x2, num_rem_inps);
            x = PDX_SUB_MX32(x, zb2);
            z = PDX_MUL_MXF32(x, s2);
            PDX_SAV_MXF32_XP(z, az2, p_z2, num_rem_outs);
            PDX_SAPOS_MXF32_FP(az2, p_z2);

            p_inp_base = p_inp_base + (CONST_TWO * trailing_dims);
            p_out_base = p_out_base + (CONST_TWO * trailing_dims);
        }
        for (; axis_index < axis_count; axis_index++)
        {
            s1 = (p_inp_scale[axis_index] / SCALE_FACTOR_4_BIT);
            zb1 = (p_inp_zero_bias[axis_index] << SHIFT_FACTOR_4_BIT);

            p_x1 = (const xb_vecMx8*) (p_inp_base);
            ax1 = PDX_LA_MX8_PP(p_x1);
            p_z1 = (xb_vecMxf32*) p_out_base;

            for (i = 0; i < (num_elm >> LOG2_PDX_M); i++)
            {
                PDX_LA32_MX8_IP(x, ax1, p_x1);
                x = PDX_SUB_MX32(x, zb1);
                z = PDX_MUL_MXF32(x, s1);
                PDX_SA_MXF32_IP(z, az1, p_z1);
            }
            PDX_LAV32_MX8_XP(x, ax1, p_x1, num_rem_inps);
            x = PDX_SUB_MX32(x, zb1);
            z = PDX_MUL_MXF32(x, s1);
            PDX_SAV_MXF32_XP(z, az1, p_z1, num_rem_outs);
            PDX_SAPOS_MXF32_FP(az1, p_z1);

            p_inp_base = p_inp_base + trailing_dims;
            p_out_base = p_out_base + trailing_dims;
        }
    }
#endif
    return 0;
}
