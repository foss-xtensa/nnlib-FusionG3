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

WORD32 xa_nn_elm_less_f32xf32_bool(WORD8 *p_out,
        const FLOAT32 *p_inp1,
        const FLOAT32 *p_inp2,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    WORD32 n, rem_inp_elms, rem_out_elms;

    /* Variables to hold input1 and input2 */
    xb_vecMxf32 x0, y0;
    /* Variable to hold output */
    xb_vecMx32 z0;
    /* Vectors holding ones and zeros */
    xb_vecMx32 one_vec = CONST_ONE;
    xb_vecMx32 zero_vec = 0;

    vboolM bool_data;
    /* Align registers */
    valign ax, ay, az;

    /* Pointers to input1, input2 and output */
    const xb_vecMxf32 *restrict p_x = (const xb_vecMxf32*) p_inp1;
    const xb_vecMxf32 *restrict p_y = (const xb_vecMxf32*) p_inp2;
    xb_vecMx8 *restrict p_z = (xb_vecMx8*) p_out;

    ax = PDX_LA_MXF32_PP(p_x);
    ay = PDX_LA_MXF32_PP(p_y);
    az = PDX_Z_ALIGN();

    for (n = 0; n < (num_elm >> LOG2_PDX_M); n++)
    {
        PDX_LA_MXF32_IP(x0, ax, p_x);
        PDX_LA_MXF32_IP(y0, ay, p_y);
        /* Bit is set if input1 < input2, else unset */
        bool_data = PDX_OLT_MXF32(x0, y0);

        /* One is stored if bool bit is set,
         * zero is stored if bool bit is unset
         */
        z0 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
        PDX_SA32_MX8_IP(z0, az, p_z);
    }
    /* Remaining elements after processing the loop */
    rem_inp_elms = (num_elm & (PDX_M - CONST_ONE)) << LOG2_SIZE_FLOAT;
    rem_out_elms = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_INT8;

    /* Variable aligning load */
    PDX_LAV_MXF32_XP(x0, ax, p_x, rem_inp_elms);
    PDX_LAV_MXF32_XP(y0, ay, p_y, rem_inp_elms);
    /* Bit is set if input1 < input2, else unset */
    bool_data = PDX_OLT_MXF32(x0, y0);

    /* One is stored if bool bit is set,
     * zero is stored if bool bit is unset
     */
    z0 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
    PDX_SAV32_MX8_XP(z0, az, p_z, rem_out_elms);
    PDX_SAPOS_MX8_FP(az, p_z);

    return 0;
}

WORD32 xa_nn_elm_less_scalar_f32xf32_bool(WORD8 *p_out,
        const FLOAT32 *p_inp1,
        const FLOAT32 inp2,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    WORD32 n, rem_inp_elms, rem_out_elms;

    /* Variables to hold input1 and input2 */
    xb_vecMxf32 x, y;
    /* Variable to hold output */
    xb_vecMx32 z;
    /* Vectors holding ones and zeros */
    xb_vecMx32 one_vec = CONST_ONE;
    xb_vecMx32 zero_vec = 0;

    vboolM bool_data;
    y = inp2;

    /* Align registers */
    valign ax, az;

    /* Pointers to input1 and output */
    const xb_vecMxf32 *restrict p_x = (const xb_vecMxf32*) p_inp1;
    xb_vecMx8 *restrict p_z = (xb_vecMx8*) p_out;

    /* Align load priming */
    ax = PDX_LA_MXF32_PP(p_x);

    /* Zeroing align register */
    az = PDX_Z_ALIGN();

    /* Loop iterates for multiple of LOG2_PDX_M */
    for (n = 0; n < (num_elm >> LOG2_PDX_M); n++)
    {
        PDX_LA_MXF32_IP(x, ax, p_x);
        /* Bit is set if input1 < input2, else unset */
        bool_data = PDX_OLT_MXF32(x, y);

        /* One is stored if bool bit is set,
         * zero is stored if bool bit is unset
         */
        z = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);

        PDX_SA32_MX8_IP(z, az, p_z);
    }

    /* Remaining elements after processing the loop */
    rem_inp_elms = (num_elm & (PDX_M - CONST_ONE)) << LOG2_SIZE_FLOAT;
    rem_out_elms = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_INT8;

    /* Variable aligning load */
    PDX_LAV_MXF32_XP(x, ax, p_x, rem_inp_elms);
    /* Bit is set if input1 < input2, else unset */
    bool_data = PDX_OLT_MXF32(x, y);

    /* One is stored if bool bit is set,
     * zero is stored if bool bit is unset
     */
    z = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
    /* Variable aligning store and flush */
    PDX_SAV32_MX8_XP(z, az, p_z, rem_out_elms);
    PDX_SAPOS_MX8_FP(az, p_z);

    return 0;
}

static inline void shapes_convert_5D(WORD32 *const __restrict__ p_5d_out_shape,
        WORD32 *const __restrict__ p_5d_inp1_shape, /* new input1 shapes */
        WORD32 *const __restrict__ p_5d_inp2_shape, /* new input2 shapes */
        const WORD32 *const __restrict__ p_out_shape,
        const WORD32 *const __restrict__ p_inp1_shape, /* original input1 shapes */
        const WORD32 *const __restrict__ p_inp2_shape, /* original input1 shapes */
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

static inline void strides_calculation(const WORD32 *const p_inp1_shape,
        const WORD32 *const p_inp2_shape,
        WORD32 *const p_inp1_strides,
        WORD32 *const p_inp2_strides)
{
    WORD32 i;

    p_inp1_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    p_inp2_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    for (i = MAX_DIMS - CONST_TWO; i >= 0; i--)
    {
        p_inp1_strides[i] = p_inp1_strides[i + CONST_ONE]
                * p_inp1_shape[i + CONST_ONE];
        p_inp2_strides[i] = p_inp2_strides[i + CONST_ONE]
                * p_inp2_shape[i + CONST_ONE];
    }
}

static inline void internal_elm_less_broadcast_2D_f32xf32_bool(
        WORD8 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp1,
        const FLOAT32 *__restrict__ p_inp2,
        WORD32 out_lc,
        WORD32 in_lc,
        const WORD32 *p_input1_shapes,
        const WORD32 *p_input2_shapes)
{
    WORD32 n, rem_inp_elms, rem_out_elms;

    /* Variables to hold input1 and input2 */
    xb_vecMxf32 x0, x1, y0, y1;

    /* Variables to hold output */
    xb_vecMx32 z0, z1;
    /* Vectors holding ones and zeros */
    xb_vecMx32 one_vec = CONST_ONE;
    xb_vecMx32 zero_vec = 0;

    vboolM bool_data;

    /* Pointer for base address for input1 */
    const xb_vecMxf32 *__restrict__ p_x = (const xb_vecMxf32*) &p_inp1[0];
    /* Pointer for base address for input2 */
    const xb_vecMxf32 *__restrict__ p_y = (const xb_vecMxf32*) &p_inp2[0];

    valign ax, ax0, ax1, ay, ay0, ay1, az, az0, az1;
    ax = PDX_LA_MXF32_PP(p_x);
    ay = PDX_LA_MXF32_PP(p_y);
    az = PDX_Z_ALIGN();

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
    xb_vecMx8 *__restrict__ p_z0 = (xb_vecMx8*) p_out;
    /* Pointer for middle address for output */
    xb_vecMx8 *__restrict__ p_z1 = (xb_vecMx8*) (p_out
            + ((out_lc / CONST_TWO) * in_lc));

    /* Priming for output stores */
    az0 = PDX_Z_ALIGN();
    az1 = PDX_Z_ALIGN();

    rem_inp_elms = (in_lc & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
    rem_out_elms = (in_lc & (PDX_M - CONST_ONE)) * SIZE_OF_INT8;

    WORD32 i;

    if (p_input1_shapes[3] == CONST_ONE)
    {
        /* p_input1_shapes[3] is 1 */
        for (i = 0; i < out_lc - CONST_ONE; i += CONST_TWO)
        {
            /* Unroll the loop by x4 for SIMD */
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                /* Load 4 elements from input1 */
                PDX_LA_MXF32_IP(x0, ax, p_x);

                /* Load 4 elements from input2 base address */
                PDX_LA_MXF32_IP(y0, ay0, p_y0);
                /* Load 4 elements from input2 middle address */
                PDX_LA_MXF32_IP(y1, ay1, p_y1);

                /* Bit is set if input1 < input2, else unset */
                bool_data = PDX_OLT_MXF32(x0, y0);

                /* One is stored if bool bit is set,
                 * zero is stored if bool bit is unset
                 */
                z0 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
                PDX_SA32_MX8_IP(z0, az0, p_z0);

                /* Bit is set if input1 < input2, else unset */
                bool_data = PDX_OLT_MXF32(x0, y1);

                /* One is stored if bool bit is set,
                 * zero is stored if bool bit is unset
                 */
                z1 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
                PDX_SA32_MX8_IP(z1, az1, p_z1);
            } /* Inner loop */

            /* Remaining iterations of inner loop */
            PDX_LAV_MXF32_XP(x0, ax, p_x, rem_inp_elms);

            PDX_LAV_MXF32_XP(y0, ay0, p_y0, rem_inp_elms);
            PDX_LAV_MXF32_XP(y1, ay1, p_y1, rem_inp_elms);

            bool_data = PDX_OLT_MXF32(x0, y0);
            z0 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
            PDX_SAV32_MX8_XP(z0, az0, p_z0, rem_out_elms);

            bool_data = PDX_OLT_MXF32(x0, y1);
            z1 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
            PDX_SAV32_MX8_XP(z1, az1, p_z1, rem_out_elms);

            PDX_SAPOS_MX8_FP(az0, p_z0);
            PDX_SAPOS_MX8_FP(az1, p_z1);

            /* input1 Pointer updates to base address
             * as input1 is broadcasted
             */
            p_x = (const xb_vecMxf32*) &p_inp1[0];
            ax = PDX_LA_MXF32_PP(p_x);
        } /* Outer loop */

        /* Loop through remaining iterations of outer loop */
        if (out_lc % CONST_TWO != 0)
        {
            /* Unroll the loop by x4 for SIMD */
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                /* Load 4 elements from input2 middle address */
                PDX_LA_MXF32_IP(y1, ay1, p_y1);
                /* Load 4 elements from input1 */
                PDX_LA_MXF32_IP(x0, ax, p_x);
                /* Bit is set if input1 < input2, else unset */
                bool_data = PDX_OLT_MXF32(x0, y1);

                /* One is stored if bool bit is set,
                 * zero is stored if bool bit is unset
                 */
                z0 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
                PDX_SA32_MX8_IP(z0, az1, p_z1);
            }
            /* Remaining iterations */
            PDX_LAV_MXF32_XP(y1, ay1, p_y1, rem_inp_elms);
            PDX_LAV_MXF32_XP(x0, ax, p_x, rem_inp_elms);
            bool_data = PDX_OLT_MXF32(x0, y1);
            z0 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
            PDX_SAV32_MX8_XP(z0, az1, p_z1, rem_out_elms);
            PDX_SAPOS_MX8_FP(az1, p_z1);
        }
    }
    else
    {
        /* p_input2_shapes[3] is 1 */
        for (i = 0; i < out_lc - CONST_ONE; i += CONST_TWO)
        {
            /* Unroll the loop by x4 for SIMD */
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                /* Load 4 elements from input2 */
                PDX_LA_MXF32_IP(y0, ay, p_y);

                /* Load 4 elements from input1 base address */
                PDX_LA_MXF32_IP(x0, ax0, p_x0);
                /* Load 4 elements from input1 middle address */
                PDX_LA_MXF32_IP(x1, ax1, p_x1);

                /* Bit is set if input1 < input2, else unset */
                bool_data = PDX_OLT_MXF32(x0, y0);

                /* One is stored if bool bit is set,
                 * zero is stored if bool bit is unset
                 */
                z0 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);

                /* Bit is set if input1 < input2, else unset */
                bool_data = PDX_OLT_MXF32(x1, y0);

                /* One is stored if bool bit is set,
                 * zero is stored if bool bit is unset
                 */
                z1 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);

                PDX_SA32_MX8_IP(z0, az0, p_z0);
                PDX_SA32_MX8_IP(z1, az1, p_z1);
            } /* Inner loop */

            /* Remaining iterations of inner loop */
            PDX_LAV_MXF32_XP(y0, ay, p_y, rem_inp_elms);
            PDX_LAV_MXF32_XP(x0, ax0, p_x0, rem_inp_elms);
            PDX_LAV_MXF32_XP(x1, ax1, p_x1, rem_inp_elms);

            bool_data = PDX_OLT_MXF32(x0, y0);
            z0 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
            bool_data = PDX_OLT_MXF32(x1, y0);
            z1 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);

            PDX_SAV32_MX8_XP(z0, az0, p_z0, rem_out_elms);
            PDX_SAV32_MX8_XP(z1, az1, p_z1, rem_out_elms);
            PDX_SAPOS_MX8_FP(az0, p_z0);
            PDX_SAPOS_MX8_FP(az1, p_z1);

            /* input2 Pointer updates to base address
             * as input1 is broadcasted
             */
            p_y = (const xb_vecMxf32*) &p_inp2[0];
            ay = PDX_LA_MXF32_PP(p_y);
        }

        /* Loop through remaining iterations of outer loop */
        if (out_lc % CONST_TWO != 0)
        {
            /* Unroll the loop by x4 for SIMD */
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                /* Load 4 elements from input2 */
                PDX_LA_MXF32_IP(y0, ay, p_y);
                /* Load 4 elements from input1 base address */
                PDX_LA_MXF32_IP(x1, ax1, p_x1);
                /* Bit is set if input1 < input2, else unset */
                bool_data = PDX_OLT_MXF32(x1, y0);

                /* One is stored if bool bit is set,
                 * zero is stored if bool bit is unset
                 */
                z1 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
                PDX_SA32_MX8_IP(z1, az1, p_z1);
            }
            /* Remaining iterations */
            PDX_LAV_MXF32_XP(y0, ay, p_y, rem_inp_elms);
            PDX_LAV_MXF32_XP(x1, ax1, p_x1, rem_inp_elms);
            bool_data = PDX_OLT_MXF32(x1, y0);
            z0 = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
            PDX_SAV32_MX8_XP(z0, az1, p_z1, rem_out_elms);
            PDX_SAPOS_MX8_FP(az1, p_z1);
        }
    }
}

static inline void internal_elm_less_broadcast_1D_scalar_f32xf32_bool(
        WORD8 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp1,
        const FLOAT32 *__restrict__ p_inp2,
        WORD32 num_elm,
        const WORD32 *__restrict__ p_input1_shapes,
        const WORD32 inp1_const,
        const WORD32 inp2_const)
{

    /* Variables to hold input1 and input2 */
    xb_vecMxf32 x, y;
    /* Variable to hold output */
    xb_vecMx32 z;
    /* Vectors holding ones and zeros */
    xb_vecMx32 one_vec = CONST_ONE;
    xb_vecMx32 zero_vec = 0;

    vboolM bool_data;

    /* Pointers for input1, input2 and output */
    xb_vecMxf32 *restrict p_x = (xb_vecMxf32*) p_inp1;
    xb_vecMxf32 *restrict p_y = (xb_vecMxf32*) p_inp2;
    xb_vecMx8 *restrict p_z = (xb_vecMx8*) p_out;

    valign ax, az;
    az = PDX_Z_ALIGN();

    WORD32 rem_inp_elms, rem_out_elms;
    WORD32 i;

    rem_inp_elms = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
    rem_out_elms = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_INT8;

    /* If input1 is to be broadcasted */
    if (((p_input1_shapes[4] == CONST_ONE) && (inp2_const != CONST_ONE))
            || (inp1_const == CONST_ONE))
    {
        x = PDX_LSR_F32_I(p_inp1, 0);
        ax = PDX_LA_MXF32_PP(p_y);
        for (i = 0; i < (num_elm >> LOG2_PDX_M); i++)
        {
            PDX_LA_MXF32_IP(y, ax, p_y);
            bool_data = PDX_OLT_MXF32(x, y);
            z = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
            PDX_SA32_MX8_IP(z, az, p_z);
        }
        PDX_LAV_MXF32_XP(y, ax, p_y, rem_inp_elms);
        bool_data = PDX_OLT_MXF32(x, y);
        z = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
        PDX_SAV32_MX8_XP(z, az, p_z, rem_out_elms);
    }
    /* If input2 is to be broadcasted */
    else
    {
        y = PDX_LSR_F32_I(p_inp2, 0);
        ax = PDX_LA_MXF32_PP(p_x);
        for (i = 0; i < (num_elm >> LOG2_PDX_M); i++)
        {
            PDX_LA_MXF32_IP(x, ax, p_x);
            bool_data = PDX_OLT_MXF32(x, y);
            z = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
            PDX_SA32_MX8_IP(z, az, p_z);
        }
        PDX_LAV_MXF32_XP(x, ax, p_x, rem_inp_elms);
        bool_data = PDX_OLT_MXF32(x, y);
        z = PDX_MOV_MX32_T(one_vec, zero_vec, bool_data);
        PDX_SAV32_MX8_XP(z, az, p_z, rem_out_elms);
    }
    PDX_SAPOS_MX8_FP(az, p_z);
}

WORD32 xa_nn_elm_less_broadcast_5D_f32xf32_bool(WORD8 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        const FLOAT32 *__restrict__ p_inp1,
        const WORD32 *const p_inp1_shape,
        const FLOAT32 *__restrict__ p_inp2,
        const WORD32 *const p_inp2_shape,
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
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), UNSUPPORTED_PARAM);

    /* UNSUPPORTED_PARAM input checks */
    XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);

    WORD32 i;

    /* Shape checks */
    /* Shapes should be greater than zero */
    for (i = 0; i < num_inp_dims; i++)
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
    WORD32 p_inp1_strides[MAX_DIMS], p_inp2_strides[MAX_DIMS];
    strides_calculation(p_5d_inp1_shape, p_5d_inp2_shape, p_inp1_strides,
            p_inp2_strides);

    WORD32 itr0, itr1, itr2, itr3;

    /* Check for broadcast need */
    WORD32 need_broadcast = 0;
    WORD32 inp1_const = CONST_ONE, inp2_const = CONST_ONE;
    for (i = 0; i < MAX_DIMS; i++)
    {
        if (p_5d_inp1_shape[i] != p_5d_inp2_shape[i])
        {
            if (p_5d_inp1_shape[i] == CONST_ONE)
            {
                p_inp1_strides[i] = 0;
            }
            else
            {
                p_inp2_strides[i] = 0;
            }
            need_broadcast = CONST_ONE;
        }

        if (p_5d_inp1_shape[i] != CONST_ONE)
        {
            inp1_const &= 0;
        }
        if (p_5d_inp2_shape[i] != CONST_ONE)
        {
            inp2_const &= 0;
        }
    }

    const FLOAT32 *__restrict__ p_inp1_base = p_inp1;
    const FLOAT32 *__restrict__ p_inp2_base = p_inp2;
    WORD8 *__restrict__ p_out_base = p_out;

    /* If broadcast is not needed */
    if (need_broadcast == 0)
    {
        xa_nn_elm_less_f32xf32_bool(
                p_out_base,
                p_inp1_base,
                p_inp2_base,
                p_5d_out_shape[0] * p_inp1_strides[0]);
    }

    /* If broadcast is needed */
    else if (inp1_const == CONST_ONE || inp2_const == CONST_ONE)
    {
        internal_elm_less_broadcast_1D_scalar_f32xf32_bool(
                p_out_base,
                p_inp1_base,
                p_inp2_base,
                p_5d_out_shape[0] * p_5d_out_shape[1] * p_5d_out_shape[2]
                        * p_5d_out_shape[3] * p_5d_out_shape[4],
                p_5d_inp1_shape,
                inp1_const,
                inp2_const);
    }
    /* Check if 4th dim in both inputs is the same */
    else if (p_inp1_strides[4] == p_inp2_strides[4])
    {
        WORD32 in_lc, out_lc;
        /* Check if 3rd dim needs to be broadcasted */
        if (p_inp1_strides[3] == 0 || p_inp2_strides[3] == 0)
        {
            /* Repeat the 4th dimension as the
             * 3rd dimension needs to be broadcasted */
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
                        internal_elm_less_broadcast_2D_f32xf32_bool(
                                p_out_base,
                                p_inp1_itr1,
                                p_inp2_itr1,
                                out_lc,
                                in_lc,
                                p_5d_inp1_shape,
                                p_5d_inp2_shape);
                        p_out_base += in_lc * out_lc;
                        p_inp1_itr1 += p_inp1_strides[2];
                        p_inp2_itr1 += p_inp2_strides[2];
                    }
                    p_inp1_itr0 += p_inp1_strides[1];
                    p_inp2_itr0 += p_inp2_strides[1];
                }
                p_inp1_base += p_inp1_strides[0];
                p_inp2_base += p_inp2_strides[0];
            }

        }
        else
        {
            /* 3rd and 4th dimensions need not be broadcasted. The lower
             * dimension broadcasting (0th, 1st, 2nd) will be taken care
             * while calculating the input addresses */
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
                        xa_nn_elm_less_f32xf32_bool(
                                p_out_base,
                                p_inp1_itr1,
                                p_inp2_itr1,
                                in_lc);
                        p_out_base += in_lc;
                        p_inp1_itr1 += p_inp1_strides[2];
                        p_inp2_itr1 += p_inp2_strides[2];
                    }
                    p_inp1_itr0 += p_inp1_strides[1];
                    p_inp2_itr0 += p_inp2_strides[1];
                }
                p_inp1_base += p_inp1_strides[0];
                p_inp2_base += p_inp2_strides[0];
            }
        }
    }
    else
    {
        /* If the last dim itself is broadcastable */
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
                        internal_elm_less_broadcast_1D_scalar_f32xf32_bool(
                                p_out_base,
                                p_inp1_itr2,
                                p_inp2_itr2,
                                p_5d_out_shape[4],
                                p_5d_inp1_shape,
                                inp1_const,
                                inp2_const);

                        p_out_base += p_5d_out_shape[4];
                        p_inp1_itr2 += p_inp1_strides[3];
                        p_inp2_itr2 += p_inp2_strides[3];
                    }
                    p_inp1_itr1 += p_inp1_strides[2];
                    p_inp2_itr1 += p_inp2_strides[2];
                }
                p_inp1_itr0 += p_inp1_strides[1];
                p_inp2_itr0 += p_inp2_strides[1];
            }
            p_inp1_base += p_inp1_strides[0];
            p_inp2_base += p_inp2_strides[0];
        }

    }

    return 0;
}
