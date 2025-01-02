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
#include <math.h>

WORD32 xa_nn_elm_quantize_f32_sym8(WORD8 *__restrict__ p_out,
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
     * quant_min should be >= -128
     * quant_max should be <= 127
     * num_inp_dims should be greater than 0 and less than or equal to 5
     * p_inp_shape values should be positive
     */
    XA_NNLIB_ARG_CHK_COND((quant_min < INT8_LOWER_LIMIT), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND((quant_max > INT8_UPPER_LIMIT), UNSUPPORTED_PARAM);
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
         * out_scale should not be equal to zero, nan and infinite
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
         * elements quantized with a scale value to get
         * the next base addresses.
         */
        length_per_step = p_inp_shape[axis] * trailing_dims;

        /* Length of the dimension along axis */
        axis_count = p_inp_shape[axis];
    }

    xb_vecMxf32 d_inp, d_out_scale;

    /* Base pointers that points to the first element in the channel */
    const FLOAT32 *__restrict__ inp_base;
    WORD8 *__restrict__ out_base;

    /* Vector pointers for the base pointers */
    xb_vecMxf32 *__restrict__ inp_base_p1;
    xb_vecMx8 *__restrict__ out_base_p1;

    xb_vecMxf32 *__restrict__ inp_base_p2;
    xb_vecMx8 *__restrict__ out_base_p2;

    /* Calculating number of simd and scalar operations */
    WORD32 num_simd4_ops = (num_elm >> LOG2_PDX_M);
    WORD32 num_scalar_ops = (num_elm & (PDX_M - CONST_ONE));

    /* Calculating multiples of 32-bits and 16-bits */
    WORD32 m_32 = num_scalar_ops * SIZE_OF_FLOAT;
    WORD32 m_8 = num_scalar_ops * SIZE_OF_INT8;

    valign align_inp1, align_out1;
    align_out1 = PDX_Z_ALIGN();

    valign align_inp2, align_out2;
    align_out2 = PDX_Z_ALIGN();

    WORD32 two_times_lps = CONST_TWO * length_per_step;

    xb_vecMxf32 d_inp_t;

    xb_vecMx32 d_out32, clamped;
    xb_vecMx32 min = quant_min;
    xb_vecMx32 max = quant_max;

    xb_vecMxf32 d_one_over_out_scale, d_one = PDX_CONST_MXF32(CONST_ONE);

    /* Setting rounding mode to zero - rounding to nearest integer */
    xb_int32 actual_scf = PDX_MOV32_SCF();
    xb_int32 converted_scf = PDX_AND_32(actual_scf, 0xFFFFFCFF);
    PDX_MOVSCF_32(converted_scf);

    WORD32 axis_index, leading_dims_index;

    /* Outermost loop iterates over the channels */
    for (axis_index = 0; axis_index < axis_count; axis_index++)
    {
        d_out_scale = p_out_scale[axis_index];
        inp_base = p_inp + (axis_index * trailing_dims);
        out_base = p_out + (axis_index * trailing_dims);

        d_one_over_out_scale = PDX_DIV_MXF32(d_one, d_out_scale);

        /* This loop iterates over the leading dims.
         * All the elements are quantized at a time for
         * single scale once loaded
         */
        for (leading_dims_index = 0;
                leading_dims_index < leading_dims - CONST_ONE;
                leading_dims_index += CONST_TWO)
        {
            inp_base_p1 = (xb_vecMxf32*) inp_base;
            align_inp1 = PDX_LA_MXF32_PP(inp_base_p1);
            out_base_p1 = (xb_vecMx8*) out_base;

            inp_base_p2 = (xb_vecMxf32*) (inp_base + length_per_step);
            align_inp2 = PDX_LA_MXF32_PP(inp_base_p2);
            out_base_p2 = (xb_vecMx8*) (out_base + length_per_step);
            for (i = 0; i < num_simd4_ops; i++)
            {
                PDX_LA_MXF32_IP(d_inp, align_inp1, inp_base_p1);
                d_inp_t = PDX_MUL_MXF32(d_inp, d_one_over_out_scale);
                d_inp_t = PDX_FIRINT_MXF32(d_inp_t);
                d_out32 = PDX_TRUNC32_MXF32(d_inp_t, 0);
                clamped = PDX_MIN_MX32(d_out32, max);
                clamped = PDX_MAX_MX32(clamped, min);

                PDX_SA32_MX8_IP(clamped, align_out1, out_base_p1);

                PDX_LA_MXF32_IP(d_inp, align_inp2, inp_base_p2);
                d_inp_t = PDX_MUL_MXF32(d_inp, d_one_over_out_scale);
                d_inp_t = PDX_FIRINT_MXF32(d_inp_t);
                d_out32 = PDX_TRUNC32_MXF32(d_inp_t, 0);
                clamped = PDX_MIN_MX32(d_out32, max);
                clamped = PDX_MAX_MX32(clamped, min);

                PDX_SA32_MX8_IP(clamped, align_out2, out_base_p2);
            }
            PDX_LAV_MXF32_XP(d_inp, align_inp1, inp_base_p1, m_32);
            d_inp_t = PDX_MUL_MXF32(d_inp, d_one_over_out_scale);
            d_inp_t = PDX_FIRINT_MXF32(d_inp_t);
            d_out32 = PDX_TRUNC32_MXF32(d_inp_t, 0);
            clamped = PDX_MIN_MX32(d_out32, max);
            clamped = PDX_MAX_MX32(clamped, min);

            PDX_SAV32_MX8_XP(clamped, align_out1, out_base_p1, m_8);
            PDX_SAPOS_MX8_FP(align_out1, out_base_p1);

            PDX_LAV_MXF32_XP(d_inp, align_inp2, inp_base_p2, m_32);
            d_inp_t = PDX_MUL_MXF32(d_inp, d_one_over_out_scale);
            d_inp_t = PDX_FIRINT_MXF32(d_inp_t);
            d_out32 = PDX_TRUNC32_MXF32(d_inp_t, 0);
            clamped = PDX_MIN_MX32(d_out32, max);
            clamped = PDX_MAX_MX32(clamped, min);

            PDX_SAV32_MX8_XP(clamped, align_out2, out_base_p2, m_8);
            PDX_SAPOS_MX8_FP(align_out2, out_base_p2);

            inp_base = inp_base + two_times_lps;
            out_base = out_base + two_times_lps;
        }
        if ((leading_dims % CONST_TWO) != 0)
        {
            inp_base_p1 = (xb_vecMxf32*) inp_base;
            align_inp1 = PDX_LA_MXF32_PP(inp_base_p1);
            out_base_p1 = (xb_vecMx8*) out_base;

            for (i = 0; i < num_simd4_ops; i++)
            {
                PDX_LA_MXF32_IP(d_inp, align_inp1, inp_base_p1);
                d_inp_t = PDX_MUL_MXF32(d_inp, d_one_over_out_scale);
                d_inp_t = PDX_FIRINT_MXF32(d_inp_t);
                d_out32 = PDX_TRUNC32_MXF32(d_inp_t, 0);
                clamped = PDX_MIN_MX32(d_out32, max);
                clamped = PDX_MAX_MX32(clamped, min);

                PDX_SA32_MX8_IP(clamped, align_out1, out_base_p1);
            }
            PDX_LAV_MXF32_XP(d_inp, align_inp1, inp_base_p1, m_32);
            d_inp_t = PDX_MUL_MXF32(d_inp, d_one_over_out_scale);
            d_inp_t = PDX_FIRINT_MXF32(d_inp_t);
            d_out32 = PDX_TRUNC32_MXF32(d_inp_t, 0);
            clamped = PDX_MIN_MX32(d_out32, max);
            clamped = PDX_MAX_MX32(clamped, min);

            PDX_SAV32_MX8_XP(clamped, align_out1, out_base_p1, m_8);
            PDX_SAPOS_MX8_FP(align_out1, out_base_p1);
        }
    }

    /* Resetting the original scf */
    PDX_MOVSCF_32(actual_scf);

    return 0;
}
