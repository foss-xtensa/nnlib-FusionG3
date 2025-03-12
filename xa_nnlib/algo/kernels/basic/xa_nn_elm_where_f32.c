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

WORD32 xa_nn_elm_where_f32xf32_f32(FLOAT32 *p_out,
        const FLOAT32 *p_inp1,
        const FLOAT32 *p_inp2,
        const UWORD8 *p_cond,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_cond, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_cond, SIZE_OF_INT8, UNSUPPORTED_PARAM);

    /* UNSUPPORTED_PARAM input checks */
    /* num_elm should be greater than 0 */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    /* Pointer to conditional input */
    const xb_vecMx8 *__restrict__ p_c = (const xb_vecMx8*) p_cond;

    /* Pointers to input1, input2 and output */
    const xb_vecMxf32 *__restrict__ p_x = (const xb_vecMxf32*) p_inp1;
    const xb_vecMxf32 *__restrict__ p_y = (const xb_vecMxf32*) p_inp2;
    xb_vecMxf32 *__restrict__ p_z = (xb_vecMxf32*) p_out;

    /* Variables to store input1, input2 and output */
    xb_vecMxf32 x, y, z;

    /* Variable to store condition */
    xb_vecMx32 c;

    /* Align registers for input1, input2, condition, output */
    valign ax, ay, ac, az;
    ax = PDX_LA_MXF32_PP(p_x);
    ay = PDX_LA_MXF32_PP(p_y);
    ac = PDX_LA_MX8_PP(p_c);

    az = PDX_Z_ALIGN();

    /* Boolean register, values are selected from input1 if the value
     * is 1, else values are selected from input2
     */
    vboolM bool_data;

    /* Calculating number of simd4 ops */
    WORD32 num_simd4_ops = (num_elm >> LOG2_PDX_M);
    WORD32 rem_inp_elms = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
    WORD32 rem_cond_elms = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_INT8;

    WORD32 i;

    for (i = 0; i < num_simd4_ops; i++)
    {
        PDX_LA_MXF32_IP(x, ax, p_x);

        PDX_LA_MXF32_IP(y, ay, p_y);

        PDX_LA32_MX8_IP(c, ac, p_c);

        /* Converting 8-bit data to boolean data */
        bool_data = PDX_EQ_MX32(c, CONST_ONE);

        z = PDX_MOV_MXF32_T(x, y, bool_data);

        PDX_SA_MXF32_IP(z, az, p_z);

    }

    PDX_LAV_MXF32_XP(x, ax, p_x, rem_inp_elms);

    PDX_LAV_MXF32_XP(y, ay, p_y, rem_inp_elms);

    PDX_LAV32_MX8_XP(c, ac, p_c, rem_cond_elms);

    /* Converting 8-bit data to boolean data */
    bool_data = PDX_EQ_MX32(c, CONST_ONE);

    z = PDX_MOV_MXF32_T(x, y, bool_data);

    PDX_SAV_MXF32_XP(z, az, p_z, rem_inp_elms);

    PDX_SAPOS_MXF32_FP(az, p_z);

    return 0;
} /* xa_nn_elm_where_f32xf32_f32 */

/* Code for broadcasting */
static inline void shapes_convert_5D(WORD32 *const __restrict__ p_5d_out_shape,
        WORD32 *const __restrict__ p_5d_inp1_shape, /* new input1 shapes */
        WORD32 *const __restrict__ p_5d_inp2_shape, /* new input2 shapes */
        WORD32 *const __restrict__ p_5d_cond_shape, /* new cond shapes */
        const WORD32 *const __restrict__ p_out_shape, /* original output shapes */
        const WORD32 *const __restrict__ p_inp1_shape, /* original input1 shapes */
        const WORD32 *const __restrict__ p_inp2_shape, /* original input2 shapes */
        const WORD32 *const __restrict__ p_cond_shape, /* original cond shape */
        const WORD32 num_inp_dims)
{
    WORD32 i;

    for (i = 0; i < num_inp_dims; i++)
    {
        p_5d_inp1_shape[i + MAX_DIMS - num_inp_dims] = p_inp1_shape[i];
        p_5d_inp2_shape[i + MAX_DIMS - num_inp_dims] = p_inp2_shape[i];
        p_5d_cond_shape[i + MAX_DIMS - num_inp_dims] = p_cond_shape[i];
        p_5d_out_shape[i + MAX_DIMS - num_inp_dims] = p_out_shape[i];
    }
} /* shapes_convert_5D */

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
} /* check_shapes */

static inline void strides_calculation(const WORD32 *const p_5d_inp1_shape,
        const WORD32 *const p_5d_inp2_shape,
        const WORD32 *const p_5d_cond_shape,
        WORD32 *const p_inp1_strides,
        WORD32 *const p_inp2_strides,
        WORD32 *const p_cond_strides)
{
    WORD32 i;

    p_inp1_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    p_inp2_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    p_cond_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    for (i = MAX_DIMS - CONST_TWO; i >= 0; i--)
    {
        p_inp1_strides[i] = p_inp1_strides[i + CONST_ONE]
                * p_5d_inp1_shape[i + CONST_ONE];
        p_inp2_strides[i] = p_inp2_strides[i + CONST_ONE]
                * p_5d_inp2_shape[i + CONST_ONE];
        p_cond_strides[i] = p_cond_strides[i + CONST_ONE]
                * p_5d_cond_shape[i + CONST_ONE];
    }
} /* strides_calculation */

static inline void internal_elm_where_broadcast_2D_f32xf32_f32(
        FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp1,
        const FLOAT32 *__restrict__ p_inp2,
        const UWORD8 *__restrict__ p_cond,
        WORD32 out_lc,
        WORD32 in_lc,
        const WORD32 *p_input1_shapes,
        const WORD32 *p_input2_shapes,
        const WORD32 *p_cond_shapes)
{
    xb_vecMxf32 x0, x1, y0, y1, z0, z1;
    xb_vecMx32 c0, c1;

    /* Align registers for input1, input2 and cond */
    valign ax0, ax1, ay0, ay1, ac0, ac1;
    /* align registers for output */
    valign az0, az1;

    vboolM bool_data0, bool_data1;

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

    /* Pointer for base address for cond */
    const xb_vecMx8 *__restrict__ p_c0 = (const xb_vecMx8*) p_cond;
    /* Pointer for middle address for input2 */
    const xb_vecMx8 *__restrict__ p_c1 = (const xb_vecMx8*) (p_cond
            + ((out_lc / CONST_TWO) * in_lc));

    /* Priming for input2 loads */
    ac0 = PDX_LA_MX8_PP(p_c0);
    ac1 = PDX_LA_MX8_PP(p_c1);

    /* Pointer for base address for output */
    xb_vecMxf32 *__restrict__ p_z0 = (xb_vecMxf32*) p_out;
    /* Pointer for middle address for output */
    xb_vecMxf32 *__restrict__ p_z1 = (xb_vecMxf32*) (p_out
            + ((out_lc / CONST_TWO) * in_lc));

    /* Priming for output stores */
    az0 = PDX_Z_ALIGN();
    az1 = PDX_Z_ALIGN();

    WORD32 num_simd4_ops = in_lc >> LOG2_PDX_M;
    WORD32 rem_inp_elms = (in_lc & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
    WORD32 rem_cond_elms = (in_lc & (PDX_M - CONST_ONE)) * SIZE_OF_INT8;

    WORD32 i, j;

    /* If shape of input1 and input2 are same but
     * different from cond at dimension 3
     */
    if ((p_input1_shapes[3] == p_input2_shapes[3])
            && (p_input1_shapes[3] != p_cond_shapes[3]))
    {
        /* If dimension 3 of cond is broadcastable */
        if (p_cond_shapes[3] == CONST_ONE)
        {
            /* Unroll the loop by x2 for SIMD */
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    /* Load 4 elements from cond */
                    PDX_LA32_MX8_IP(c0, ac0, p_c0);

                    /* Converting 8-bit data to boolean data */
                    bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);

                    /* Load the 4 elements from input1 base address */
                    PDX_LA_MXF32_IP(x0, ax0, p_x0);
                    /* Load the 4 elements from input1 Middle address */
                    PDX_LA_MXF32_IP(x1, ax1, p_x1);

                    /* Load the 4 elements from input2 base address */
                    PDX_LA_MXF32_IP(y0, ay0, p_y0);
                    /* Load the 4 elements from input2 Middle address */
                    PDX_LA_MXF32_IP(y1, ay1, p_y1);

                    /* Selecting the values from vectors
                     * based on boolean values
                     */
                    z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                    z1 = PDX_MOV_MXF32_T(x1, y1, bool_data0);

                    /* Store the output */
                    PDX_SA_MXF32_IP(z0, az0, p_z0);
                    PDX_SA_MXF32_IP(z1, az1, p_z1);
                } /* Inner loop */

                /* Remaining iterations of inner loop */
                PDX_LAV32_MX8_XP(c0, ac0, p_c0, rem_cond_elms);

                bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);

                PDX_LAV_MXF32_XP(x0, ax0, p_x0, rem_inp_elms);
                PDX_LAV_MXF32_XP(x1, ax1, p_x1, rem_inp_elms);

                PDX_LAV_MXF32_XP(y0, ay0, p_y0, rem_inp_elms);
                PDX_LAV_MXF32_XP(y1, ay1, p_y1, rem_inp_elms);

                z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                z1 = PDX_MOV_MXF32_T(x1, y1, bool_data0);

                PDX_SAV_MXF32_XP(z0, az0, p_z0, rem_inp_elms);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

                /* cond Pointer updates to base address
                 * as cond is broadcasted
                 */
                p_c0 = (const xb_vecMx8*) p_cond;
                ac0 = PDX_LA_MX8_PP(p_c0);

            } /* Outer loop */

            /* Loop through remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    /* Load 4 elements from cond */
                    PDX_LA32_MX8_IP(c0, ac0, p_c0);

                    /* Converting 8-bit data to boolean data */
                    bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);

                    /* Load the 4 elements from input1 Middle address */
                    PDX_LA_MXF32_IP(x1, ax1, p_x1);

                    /* Load the 4 elements from input2 Middle address */
                    PDX_LA_MXF32_IP(y1, ay1, p_y1);

                    z1 = PDX_MOV_MXF32_T(x1, y1, bool_data0);

                    PDX_SA_MXF32_IP(z1, az1, p_z1);
                }

                /* Remaining iterations */
                PDX_LAV32_MX8_XP(c0, ac0, p_c0, rem_cond_elms);
                bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                PDX_LAV_MXF32_XP(x1, ax1, p_x1, rem_inp_elms);
                PDX_LAV_MXF32_XP(y1, ay1, p_y1, rem_inp_elms);
                z1 = PDX_MOV_MXF32_T(x1, y1, bool_data0);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

            } /* (out_lc % CONST_TWO) != 0 */

        } /* p_cond_shapes[3] == CONST_ONE */

        /* If dimension 3 of input1 and input2 are broadcastable */
        else
        {
            /* Unroll the loop by x2 for SIMD */
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    /* Load 4 elements of input1 */
                    PDX_LA_MXF32_IP(x0, ax0, p_x0);

                    /* Load 4 elements of input2 */
                    PDX_LA_MXF32_IP(y0, ay0, p_y0);

                    /* Load the 4 elements from cond base address */
                    PDX_LA32_MX8_IP(c0, ac0, p_c0);
                    /* Load the 4 elements from cond Middle address */
                    PDX_LA32_MX8_IP(c1, ac1, p_c1);

                    /* Converting 8-bit data to boolean data */
                    bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                    bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);

                    /* Selecting the values from vectors
                     * based on boolean values
                     */
                    z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                    z1 = PDX_MOV_MXF32_T(x0, y0, bool_data1);

                    /* Store the output */
                    PDX_SA_MXF32_IP(z0, az0, p_z0);
                    PDX_SA_MXF32_IP(z1, az1, p_z1);
                } /* Inner loop */

                /* Remaining iterations of inner loop */
                PDX_LAV_MXF32_XP(x0, ax0, p_x0, rem_inp_elms);
                PDX_LAV_MXF32_XP(y0, ay0, p_y0, rem_inp_elms);
                PDX_LAV32_MX8_XP(c0, ac0, p_c0, rem_cond_elms);
                PDX_LAV32_MX8_XP(c1, ac1, p_c1, rem_cond_elms);
                bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);
                z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                z1 = PDX_MOV_MXF32_T(x0, y0, bool_data1);
                PDX_SAV_MXF32_XP(z0, az0, p_z0, rem_inp_elms);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

                /* input1 and input2 Pointers update to base address
                 * as input1 and input2 are broadcasted
                 */
                p_x0 = (const xb_vecMxf32*) p_inp1;
                p_y0 = (const xb_vecMxf32*) p_inp2;

                ax0 = PDX_LA_MXF32_PP(p_x0);
                ay0 = PDX_LA_MXF32_PP(p_y0);

            } /* Outer loop */

            /* Loop through remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    /* Load 4 elements of input1 */
                    PDX_LA_MXF32_IP(x0, ax0, p_x0);

                    /* Load 4 elements of input2 */
                    PDX_LA_MXF32_IP(y0, ay0, p_y0);

                    /* Load the 4 elements from cond Middle address */
                    PDX_LA32_MX8_IP(c1, ac1, p_c1);

                    bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);

                    z1 = PDX_MOV_MXF32_T(x0, y0, bool_data1);

                    PDX_SA_MXF32_IP(z1, az1, p_z1);
                }

                /* Remaining iterations */
                PDX_LAV_MXF32_XP(x0, ax0, p_x0, rem_inp_elms);
                PDX_LAV_MXF32_XP(y0, ay0, p_y0, rem_inp_elms);
                PDX_LAV32_MX8_XP(c1, ac1, p_c1, rem_cond_elms);
                bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);
                z1 = PDX_MOV_MXF32_T(x0, y0, bool_data1);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

            } /* (out_lc % CONST_TWO) != 0 */

        } /* p_cond_shapes[3] != CONST_ONE */

    } /* (p_input1_shapes[3] == p_input2_shapes[3])
     && (p_input1_shapes[3] != p_cond_shapes[3]) */

    /* If shape of input1 and cond are same but
     * different from input2 at dimension 3
     */
    else if ((p_input1_shapes[3] == p_cond_shapes[3])
            && (p_input1_shapes[3] != p_input2_shapes[3]))
    {
        if (p_input2_shapes[3] == CONST_ONE)
        {
            /* Unroll the loop by x2 for SIMD */
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    /* Load 4 elements from input2 */
                    PDX_LA_MXF32_IP(y0, ay0, p_y0);

                    /* Load 4 elements from input1 base address */
                    PDX_LA_MXF32_IP(x0, ax0, p_x0);
                    /* Load 4 elements from input1 middle address */
                    PDX_LA_MXF32_IP(x1, ax1, p_x1);

                    /* Load 4 elements from cond base address */
                    PDX_LA32_MX8_IP(c0, ac0, p_c0);
                    /* Load 4 elements from cond middle address */
                    PDX_LA32_MX8_IP(c1, ac1, p_c1);

                    /* Converting 8-bit data to boolean data */
                    bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                    bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);

                    /* Selecting the values from vectors
                     * based on boolean values
                     */
                    z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                    z1 = PDX_MOV_MXF32_T(x1, y0, bool_data1);

                    /* Store the output */
                    PDX_SA_MXF32_IP(z0, az0, p_z0);
                    PDX_SA_MXF32_IP(z1, az1, p_z1);

                } /* Inner loop */

                /* Remaining iterations of inner loop */
                PDX_LAV_MXF32_XP(y0, ay0, p_y0, rem_inp_elms);
                PDX_LAV_MXF32_XP(x0, ax0, p_x0, rem_inp_elms);
                PDX_LAV_MXF32_XP(x1, ax1, p_x1, rem_inp_elms);
                PDX_LAV32_MX8_XP(c0, ac0, p_c0, rem_cond_elms);
                PDX_LAV32_MX8_XP(c1, ac1, p_c1, rem_cond_elms);
                bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);
                z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                z1 = PDX_MOV_MXF32_T(x1, y0, bool_data1);
                PDX_SAV_MXF32_XP(z0, az0, p_z0, rem_inp_elms);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

                /* input2 Pointer updates to base address
                 * as input2 is broadcasted
                 */
                p_y0 = (const xb_vecMxf32*) p_inp2;
                ay0 = PDX_LA_MXF32_PP(p_y0);

            } /* Outer loop */

            /* Remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    PDX_LA_MXF32_IP(y0, ay0, p_y0);
                    PDX_LA_MXF32_IP(x1, ax1, p_x1);
                    PDX_LA32_MX8_IP(c1, ac1, p_c1);
                    bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);
                    z1 = PDX_MOV_MXF32_T(x1, y0, bool_data1);
                    PDX_SA_MXF32_IP(z1, az1, p_z1);
                }
                PDX_LAV_MXF32_XP(y0, ay0, p_y0, rem_inp_elms);
                PDX_LAV_MXF32_XP(x1, ax1, p_x1, rem_inp_elms);
                PDX_LAV32_MX8_XP(c1, ac1, p_c1, rem_cond_elms);
                bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);
                z1 = PDX_MOV_MXF32_T(x1, y0, bool_data1);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

            } /* (out_lc % CONST_TWO) != 0 */

        } /* p_input2_shapes[3] == CONST_ONE */

        /* If dimension 3 of input1 and cond are broadcastable */
        else
        {
            /* Unroll the loop by x2 for SIMD */
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    /* Load 4 elements of input1 */
                    PDX_LA_MXF32_IP(x0, ax0, p_x0);

                    /* Load 4 elements of cond */
                    PDX_LA32_MX8_IP(c0, ac0, p_c0);

                    /* Converting 8-bit data to boolean data */
                    bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);

                    /* Load 4 elements from input2 base address */
                    PDX_LA_MXF32_IP(y0, ay0, p_y0);
                    /* Load 4 elements from input2 middle address */
                    PDX_LA_MXF32_IP(y1, ay1, p_y1);

                    /* Selecting the values from vectors
                     * based on boolean values
                     */
                    z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                    z1 = PDX_MOV_MXF32_T(x0, y1, bool_data0);

                    /* Store the output */
                    PDX_SA_MXF32_IP(z0, az0, p_z0);
                    PDX_SA_MXF32_IP(z1, az1, p_z1);

                } /* Inner loop */

                /* Remaining iterations of inner loop */
                PDX_LAV_MXF32_XP(x0, ax0, p_x0, rem_inp_elms);
                PDX_LAV32_MX8_XP(c0, ac0, p_c0, rem_cond_elms);
                bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                PDX_LAV_MXF32_XP(y0, ay0, p_y0, rem_inp_elms);
                PDX_LAV_MXF32_XP(y1, ay1, p_y1, rem_inp_elms);
                z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                z1 = PDX_MOV_MXF32_T(x0, y1, bool_data0);
                PDX_SAV_MXF32_XP(z0, az0, p_z0, rem_inp_elms);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

                /* input1 and cond Pointers update to base address
                 * as input1 and cond are broadcasted
                 */
                p_x0 = (const xb_vecMxf32*) p_inp1;
                p_c0 = (const xb_vecMx8*) p_cond;

                ax0 = PDX_LA_MXF32_PP(p_x0);
                ac0 = PDX_LA_MX8_PP(p_c0);

            } /* Outer loop */

            /* Remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    PDX_LA_MXF32_IP(x0, ax0, p_x0);
                    PDX_LA32_MX8_IP(c0, ac0, p_c0);
                    bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                    PDX_LA_MXF32_IP(y1, ay1, p_y1);
                    z1 = PDX_MOV_MXF32_T(x0, y1, bool_data0);
                    PDX_SA_MXF32_IP(z1, az1, p_z1);
                }
                PDX_LAV_MXF32_XP(x0, ax0, p_x0, rem_inp_elms);
                PDX_LAV32_MX8_XP(c0, ac0, p_c0, rem_cond_elms);
                bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                PDX_LAV_MXF32_XP(y1, ay1, p_y1, rem_inp_elms);
                z1 = PDX_MOV_MXF32_T(x0, y1, bool_data0);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

            } /* (out_lc % CONST_TWO) != 0 */

        } /* p_input2_shapes[3] != CONST_ONE */

    } /* (p_input1_shapes[3] == p_cond_shapes[3])
     && (p_input1_shapes[3] != p_input2_shapes[3]) */

    /* If shape of input2 and cond are same but
     * different from input1 at dimension 3
     */
    else if ((p_input2_shapes[3] == p_cond_shapes[3])
            && (p_input1_shapes[3] != p_input2_shapes[3]))
    {
        if (p_input1_shapes[3] == CONST_ONE)
        {
            /* Unroll the loop by x2 for SIMD */
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    /* Load 4 elements of input1 */
                    PDX_LA_MXF32_IP(x0, ax0, p_x0);

                    /* Load 4 elements from input2 base address */
                    PDX_LA_MXF32_IP(y0, ay0, p_y0);
                    /* Load 4 elements from input2 middle address */
                    PDX_LA_MXF32_IP(y1, ay1, p_y1);

                    /* Load 4 elements from cond base address */
                    PDX_LA32_MX8_IP(c0, ac0, p_c0);
                    /* Load 4 elements from cond middle address */
                    PDX_LA32_MX8_IP(c1, ac1, p_c1);

                    /* Converting 8-bit data to boolean data */
                    bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                    bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);

                    /* Selecting the values from vectors
                     * based on boolean values
                     */
                    z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                    z1 = PDX_MOV_MXF32_T(x0, y1, bool_data1);

                    /* Store the output */
                    PDX_SA_MXF32_IP(z0, az0, p_z0);
                    PDX_SA_MXF32_IP(z1, az1, p_z1);

                } /* Inner loop */

                /* Remaining iterations of inner loop */
                PDX_LAV_MXF32_XP(x0, ax0, p_x0, rem_inp_elms);
                PDX_LAV_MXF32_XP(y0, ay0, p_y0, rem_inp_elms);
                PDX_LAV_MXF32_XP(y1, ay1, p_y1, rem_inp_elms);
                PDX_LAV32_MX8_XP(c0, ac0, p_c0, rem_cond_elms);
                PDX_LAV32_MX8_XP(c1, ac1, p_c1, rem_cond_elms);
                bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);
                z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                z1 = PDX_MOV_MXF32_T(x0, y1, bool_data1);
                PDX_SAV_MXF32_XP(z0, az0, p_z0, rem_inp_elms);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

                /* input1 Pointer updates to base address
                 * as input1 is broadcasted
                 */
                p_x0 = (const xb_vecMxf32*) p_inp1;
                ax0 = PDX_LA_MXF32_PP(p_x0);

            }/* Outer loop */

            /* Remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    PDX_LA_MXF32_IP(x0, ax0, p_x0);
                    PDX_LA_MXF32_IP(y1, ay1, p_y1);
                    PDX_LA32_MX8_IP(c1, ac1, p_c1);
                    bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);
                    z1 = PDX_MOV_MXF32_T(x0, y1, bool_data1);
                    PDX_SA_MXF32_IP(z1, az1, p_z1);
                }
                PDX_LAV_MXF32_XP(x0, ax0, p_x0, rem_inp_elms);
                PDX_LAV_MXF32_XP(y1, ay1, p_y1, rem_inp_elms);
                PDX_LAV32_MX8_XP(c1, ac1, p_c1, rem_cond_elms);
                bool_data1 = PDX_EQ_MX32(c1, CONST_ONE);
                z1 = PDX_MOV_MXF32_T(x0, y1, bool_data1);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

            } /* (out_lc % CONST_TWO) != 0 */

        } /* p_input1_shapes[3] == CONST_ONE */

        /* If dimension 3 of input2 and cond are broadcastable */
        else
        {
            /* Unroll the loop by x2 for SIMD */
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    /* Load 4 elements from input1 base address */
                    PDX_LA_MXF32_IP(x0, ax0, p_x0);
                    /* Load 4 elements from input1 base address */
                    PDX_LA_MXF32_IP(x1, ax1, p_x1);

                    /* Load 4 elements of input2 */
                    PDX_LA_MXF32_IP(y0, ay0, p_y0);

                    /* Load 4 elements of cond */
                    PDX_LA32_MX8_IP(c0, ac0, p_c0);

                    /* Converting 8-bit data to boolean data */
                    bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);

                    /* Selecting the values from vectors
                     * based on boolean values
                     */
                    z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                    z1 = PDX_MOV_MXF32_T(x1, y0, bool_data0);

                    /* Store the output */
                    PDX_SA_MXF32_IP(z0, az0, p_z0);
                    PDX_SA_MXF32_IP(z1, az1, p_z1);

                } /* Inner loop */

                /* Remaining iterations of inner loop */
                PDX_LAV_MXF32_XP(x0, ax0, p_x0, rem_inp_elms);
                PDX_LAV_MXF32_XP(x1, ax1, p_x1, rem_inp_elms);
                PDX_LAV_MXF32_XP(y0, ay0, p_y0, rem_inp_elms);
                PDX_LAV32_MX8_XP(c0, ac0, p_c0, rem_cond_elms);
                bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                z0 = PDX_MOV_MXF32_T(x0, y0, bool_data0);
                z1 = PDX_MOV_MXF32_T(x1, y0, bool_data0);
                PDX_SAV_MXF32_XP(z0, az0, p_z0, rem_inp_elms);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

                /* input2 and cond Pointers update to base address
                 * as input2 and cond are broadcasted
                 */
                p_y0 = (const xb_vecMxf32*) p_inp2;
                p_c0 = (const xb_vecMx8*) p_cond;

                ay0 = PDX_LA_MXF32_PP(p_y0);
                ac0 = PDX_LA_MX8_PP(p_c0);

            } /* Outer loop */

            /* Remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                /* Unroll the loop by x4 for SIMD */
                for (j = 0; j < num_simd4_ops; j++)
                {
                    PDX_LA_MXF32_IP(x1, ax1, p_x1);
                    PDX_LA_MXF32_IP(y0, ay0, p_y0);
                    PDX_LA32_MX8_IP(c0, ac0, p_c0);
                    bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                    z1 = PDX_MOV_MXF32_T(x1, y0, bool_data0);
                    PDX_SA_MXF32_IP(z1, az1, p_z1);
                }
                PDX_LAV_MXF32_XP(x1, ax1, p_x1, rem_inp_elms);
                PDX_LAV_MXF32_XP(y0, ay0, p_y0, rem_inp_elms);
                PDX_LAV32_MX8_XP(c0, ac0, p_c0, rem_cond_elms);
                bool_data0 = PDX_EQ_MX32(c0, CONST_ONE);
                z1 = PDX_MOV_MXF32_T(x1, y0, bool_data0);
                PDX_SAV_MXF32_XP(z1, az1, p_z1, rem_inp_elms);

            } /* (out_lc % CONST_TWO) != 0 */

        } /* p_input1_shapes[3] != CONST_ONE */

    } /* (p_input2_shapes[3] == p_cond_shapes[3])
     && (p_input1_shapes[3] != p_input2_shapes[3]) */

    /* Flushing output align registers */
    PDX_SAPOS_MXF32_FP(az0, p_z0);
    PDX_SAPOS_MXF32_FP(az1, p_z1);

} /* internal_elm_where_broadcast_2D_f32xf32_f32 */

static inline void internal_elm_where_broadcast_1D_scalar_f32xf32_f32(
        FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp1,
        const FLOAT32 *__restrict__ p_inp2,
        const UWORD8 *__restrict__ p_cond,
        WORD32 num_elm,
        const WORD32 *p_input1_shapes,
        const WORD32 *p_input2_shapes,
        const WORD32 *p_cond_shapes)
{
    xb_vecMxf32 x, y, z;
    xb_vecMx32 c;

    /* Align registers for input1, input2 and cond */
    valign ax, ay, ac;
    /* align registers for output */
    valign az;

    vboolM bool_data;

    /* Pointer for base address for input1 */
    const xb_vecMxf32 *__restrict__ p_x = (const xb_vecMxf32*) p_inp1;

    /* Priming for input1 load */
    ax = PDX_LA_MXF32_PP(p_x);

    /* Pointer for base address for input2 */
    const xb_vecMxf32 *__restrict__ p_y = (const xb_vecMxf32*) p_inp2;

    /* Priming for input2 load */
    ay = PDX_LA_MXF32_PP(p_y);

    /* Pointer for base address for cond */
    const xb_vecMx8 *__restrict__ p_c = (const xb_vecMx8*) p_cond;

    /* Priming for cond load */
    ac = PDX_LA_MX8_PP(p_c);

    /* Pointer for base address for output */
    xb_vecMxf32 *__restrict__ p_z = (xb_vecMxf32*) p_out;

    /* Priming for ouput store */
    az = PDX_Z_ALIGN();

    WORD32 num_simd4_ops = num_elm >> LOG2_PDX_M;
    WORD32 rem_inp_elms = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
    WORD32 rem_cond_elms = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_INT8;

    WORD32 i;

    /* If shape of input1 and input2 are same but
     * different from cond at dimension 4
     */
    if ((p_input1_shapes[4] == p_input2_shapes[4])
            && (p_input1_shapes[4] != p_cond_shapes[4]))
    {
        /* If dimension 4 of cond is broadcastable */
        if (p_cond_shapes[4] == CONST_ONE)
        {
            c = p_cond[0];

            /* Converting 8-bit data to boolean data */
            bool_data = PDX_EQ_MX32(c, CONST_ONE);
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load 4 elements of input1 */
                PDX_LA_MXF32_IP(x, ax, p_x);
                /* Load 4 elements of input2 */
                PDX_LA_MXF32_IP(y, ay, p_y);

                /* Selecting the values from vectors
                 * based on boolean values
                 */
                z = PDX_MOV_MXF32_T(x, y, bool_data);

                /* Store the output */
                PDX_SA_MXF32_IP(z, az, p_z);
            }

            /* Remaining iterations */
            PDX_LAV_MXF32_XP(x, ax, p_x, rem_inp_elms);
            PDX_LAV_MXF32_XP(y, ay, p_y, rem_inp_elms);
            z = PDX_MOV_MXF32_T(x, y, bool_data);
            PDX_SAV_MXF32_XP(z, az, p_z, rem_inp_elms);

        } /* p_cond_shapes[4] == CONST_ONE */

        /* If dimension 4 of input1 and input2 are broadcastable */
        else
        {
            x = p_inp1[0];
            y = p_inp2[0];
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load 4 elemenst of cond */
                PDX_LA32_MX8_IP(c, ac, p_c);

                /* Converting 8-bit data to boolean data */
                bool_data = PDX_EQ_MX32(c, CONST_ONE);

                /* Selecting the values from vectors
                 * based on boolean values
                 */
                z = PDX_MOV_MXF32_T(x, y, bool_data);

                /* Store the output */
                PDX_SA_MXF32_IP(z, az, p_z);
            }

            /* Remaining iterations */
            PDX_LAV32_MX8_XP(c, ac, p_c, rem_cond_elms);
            bool_data = PDX_EQ_MX32(c, CONST_ONE);
            z = PDX_MOV_MXF32_T(x, y, bool_data);
            PDX_SAV_MXF32_XP(z, az, p_z, rem_inp_elms);

        } /* p_cond_shapes[4] != CONST_ONE */

    } /* (p_input1_shapes[4] == p_input2_shapes[4])
     && (p_input1_shapes[4] != p_cond_shapes[4]) */

    /* If shape of input1 and cond are same but
     * different from input2 at dimension 4
     */
    else if ((p_input1_shapes[4] == p_cond_shapes[4])
            && (p_input1_shapes[4] != p_input2_shapes[4]))
    {
        /* If dimension 4 of input2 is broadcastable */
        if (p_input2_shapes[4] == CONST_ONE)
        {
            y = p_inp2[0];
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load 4 elements of input1 */
                PDX_LA_MXF32_IP(x, ax, p_x);

                /* Load 4 elements of cond */
                PDX_LA32_MX8_IP(c, ac, p_c);

                /* Converting 8-bit data to boolean data */
                bool_data = PDX_EQ_MX32(c, CONST_ONE);

                /* Selecting the values from vectors
                 * based on boolean values
                 */
                z = PDX_MOV_MXF32_T(x, y, bool_data);

                /* Store the output */
                PDX_SA_MXF32_IP(z, az, p_z);
            }

            /* Remaining iterations */
            PDX_LAV_MXF32_XP(x, ax, p_x, rem_inp_elms);
            PDX_LAV32_MX8_XP(c, ac, p_c, rem_cond_elms);
            bool_data = PDX_EQ_MX32(c, CONST_ONE);
            z = PDX_MOV_MXF32_T(x, y, bool_data);
            PDX_SAV_MXF32_XP(z, az, p_z, rem_inp_elms);

        } /* p_input2_shapes[4] == CONST_ONE */

        /* If dimension 4 of input1 and cond are broadcastable */
        else
        {
            x = p_inp1[0];
            c = p_cond[0];
            /* Converting 8-bit data to boolean data */
            bool_data = PDX_EQ_MX32(c, CONST_ONE);
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load 4 elements of input2 */
                PDX_LA_MXF32_IP(y, ay, p_y);

                /* Selecting the values from vectors
                 * based on boolean values
                 */
                z = PDX_MOV_MXF32_T(x, y, bool_data);

                /* Store the output */
                PDX_SA_MXF32_IP(z, az, p_z);
            }

            /* Remaining iterations */
            PDX_LAV_MXF32_XP(y, ay, p_y, rem_inp_elms);
            z = PDX_MOV_MXF32_T(x, y, bool_data);
            PDX_SAV_MXF32_XP(z, az, p_z, rem_inp_elms);

        } /* p_input2_shapes[4] != CONST_ONE */

    } /* (p_input1_shapes[4] == p_cond_shapes[4])
     && (p_input1_shapes[4] != p_input2_shapes[4]) */

    /* If shape of input2 and cond are same but
     * different from input1 at dimension 4
     */
    else if ((p_input2_shapes[4] == p_cond_shapes[4])
            && (p_input1_shapes[4] != p_input2_shapes[4]))
    {
        /* If dimension 4 of input1 is broadcastable */
        if (p_input1_shapes[4] == CONST_ONE)
        {
            x = p_inp1[0];

            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load 4 elements of input2 */
                PDX_LA_MXF32_IP(y, ay, p_y);

                /* Load 4 elements of cond */
                PDX_LA32_MX8_IP(c, ac, p_c);

                /* Converting 8-bit data to boolean data */
                bool_data = PDX_EQ_MX32(c, CONST_ONE);

                /* Selecting the values from vectors
                 * based on boolean values
                 */
                z = PDX_MOV_MXF32_T(x, y, bool_data);

                /* Store the output */
                PDX_SA_MXF32_IP(z, az, p_z);
            }

            /* Remaining iterations */
            PDX_LAV_MXF32_XP(y, ay, p_y, rem_inp_elms);
            PDX_LAV32_MX8_XP(c, ac, p_c, rem_cond_elms);
            bool_data = PDX_EQ_MX32(c, CONST_ONE);
            z = PDX_MOV_MXF32_T(x, y, bool_data);
            PDX_SAV_MXF32_XP(z, az, p_z, rem_inp_elms);

        } /* p_input1_shapes[4] == CONST_ONE */

        /* If dimension 4 of input2 and cond are broadcastable */
        else
        {
            y = p_inp2[0];
            c = p_cond[0];
            /* Converting 8-bit data to boolean data */
            bool_data = PDX_EQ_MX32(c, CONST_ONE);
            for (i = 0; i < num_simd4_ops; i++)
            {
                /* Load 4 elements of input1 */
                PDX_LA_MXF32_IP(x, ax, p_x);

                /* Selecting the values from vectors
                 * based on boolean values
                 */
                z = PDX_MOV_MXF32_T(x, y, bool_data);

                /* Store the output */
                PDX_SA_MXF32_IP(z, az, p_z);
            }

            /* Remaining iterations */
            PDX_LAV_MXF32_XP(x, ax, p_x, rem_inp_elms);
            z = PDX_MOV_MXF32_T(x, y, bool_data);
            PDX_SAV_MXF32_XP(z, az, p_z, rem_inp_elms);

        } /* p_input1_shapes[4] != CONST_ONE */

    } /* (p_input2_shapes[4] == p_cond_shapes[4])
     && (p_input1_shapes[4] != p_input2_shapes[4]) */

    /* Flushing output align register */
    PDX_SAPOS_MXF32_FP(az, p_z);

} /* internal_elm_where_broadcast_1D_scalar_f32xf32_f32 */

static inline void internal_elm_where_two_vecs_const(
        FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp1,
        const FLOAT32 *__restrict__ p_inp2,
        const UWORD8 *__restrict__ p_cond,
        WORD32 num_elm,
        WORD32 inp1_const,
        WORD32 inp2_const,
        WORD32 cond_const)
{
    xb_vecMxf32 x, y, z;
    xb_vecMx32 c;

    /* Align registers for input1, input2 and cond */
    valign ax, ay, ac;
    /* align registers for output */
    valign az;

    vboolM bool_data;

    /* Pointer for base address for input1 */
    const xb_vecMxf32 *__restrict__ p_x = (const xb_vecMxf32*) p_inp1;

    /* Priming for input1 load */
    ax = PDX_LA_MXF32_PP(p_x);

    /* Pointer for base address for input2 */
    const xb_vecMxf32 *__restrict__ p_y = (const xb_vecMxf32*) p_inp2;

    /* Priming for input2 load */
    ay = PDX_LA_MXF32_PP(p_y);

    /* Pointer for base address for cond */
    const xb_vecMx8 *__restrict__ p_c = (const xb_vecMx8*) p_cond;

    /* Priming for cond load */
    ac = PDX_LA_MX8_PP(p_c);

    /* Pointer for base address for output */
    xb_vecMxf32 *__restrict__ p_z = (xb_vecMxf32*) p_out;

    /* Priming for output store */
    az = PDX_Z_ALIGN();

    WORD32 num_simd4_ops = num_elm >> LOG2_PDX_M;
    WORD32 rem_inp_elms = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
    WORD32 rem_cond_elms = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_INT8;

    WORD32 i;

    /* If both input1 and input2 are constants but cond is not constant */
    if (((inp1_const == CONST_ONE) && (inp2_const == CONST_ONE)
            && (cond_const == 0)))
    {
        x = p_inp1[0];
        y = p_inp2[0];
        for (i = 0; i < num_simd4_ops; i++)
        {
            /* Load 4 elements of cond */
            PDX_LA32_MX8_IP(c, ac, p_c);

            /* Converting 8-bit data to boolean data */
            bool_data = PDX_EQ_MX32(c, CONST_ONE);

            /* Selecting the values from vectors
             * based on boolean values
             */
            z = PDX_MOV_MXF32_T(x, y, bool_data);

            /* Store the output */
            PDX_SA_MXF32_IP(z, az, p_z);
        }

        /* Remaining iterations */
        PDX_LAV32_MX8_XP(c, ac, p_c, rem_cond_elms);
        bool_data = PDX_EQ_MX32(c, CONST_ONE);
        z = PDX_MOV_MXF32_T(x, y, bool_data);
        PDX_SAV_MXF32_XP(z, az, p_z, rem_inp_elms);

    } /* If both input1 and input2 are constants but cond is not constant */

    /* If both input1 and cond are constants but input2 is not constant */
    else if (((inp1_const == CONST_ONE) && (cond_const == CONST_ONE)
            && (inp2_const == 0)))
    {
        x = p_inp1[0];
        c = p_cond[0];
        /* Converting 8-bit data to boolean data */
        bool_data = PDX_EQ_MX32(c, CONST_ONE);
        for (i = 0; i < num_simd4_ops; i++)
        {
            /* Load 4 elements of input2 */
            PDX_LA_MXF32_IP(y, ay, p_y);

            /* Selecting the values from vectors
             * based on boolean values
             */
            z = PDX_MOV_MXF32_T(x, y, bool_data);

            /* Store the output */
            PDX_SA_MXF32_IP(z, az, p_z);
        }

        /* Remaining iterations */
        PDX_LAV_MXF32_XP(y, ay, p_y, rem_inp_elms);
        z = PDX_MOV_MXF32_T(x, y, bool_data);
        PDX_SAV_MXF32_XP(z, az, p_z, rem_inp_elms);

    } /* If both input1 and cond are constants but input2 is not constant */

    /* If both input2 and cond are constants but input1 is not constant */
    else if (((inp2_const == CONST_ONE) && (cond_const == CONST_ONE)
            && (inp1_const == 0)))
    {
        y = p_inp2[0];
        c = p_cond[0];
        /* Converting 8-bit data to boolean data */
        bool_data = PDX_EQ_MX32(c, CONST_ONE);
        for (i = 0; i < num_simd4_ops; i++)
        {
            /* Load 4 elements of input1 */
            PDX_LA_MXF32_IP(x, ax, p_x);

            /* Selecting the values from vectors
             * based on boolean values
             */
            z = PDX_MOV_MXF32_T(x, y, bool_data);

            /* Store the output */
            PDX_SA_MXF32_IP(z, az, p_z);
        }

        /* Remaining iterations */
        PDX_LAV_MXF32_XP(x, ax, p_x, rem_inp_elms);
        z = PDX_MOV_MXF32_T(x, y, bool_data);
        PDX_SAV_MXF32_XP(z, az, p_z, rem_inp_elms);

    } /* If both input2 and cond are constants but input1 is not constant */

    /* Flushing output align register */
    PDX_SAPOS_MXF32_FP(az, p_z);

} /* internal_elm_where_two_vecs_const */

WORD32 xa_nn_elm_where_broadcast_5D_f32xf32_f32(FLOAT32 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        const FLOAT32 *__restrict__ p_inp1,
        const WORD32 *const p_inp1_shape,
        const FLOAT32 *__restrict__ p_inp2,
        const WORD32 *const p_inp2_shape,
        const UWORD8 *p_cond,
        const WORD32 *const p_cond_shape,
        WORD32 num_inp_dims)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_cond, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_cond_shape, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, SIZE_OF_FLOAT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_cond, SIZE_OF_INT8, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_cond_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);

    /* num_inp_dims should be greater than 0 and less than or equal to 5 */
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
        XA_NNLIB_ARG_CHK_COND((p_cond_shape[i] <= 0), UNSUPPORTED_PARAM);
    }

    /* 5D shapes initialization */
    WORD32 p_5d_out_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE};
    WORD32 p_5d_inp1_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE};
    WORD32 p_5d_inp2_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE};
    WORD32 p_5d_cond_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE};

    shapes_convert_5D(p_5d_out_shape, p_5d_inp1_shape, p_5d_inp2_shape,
            p_5d_cond_shape, p_out_shape, p_inp1_shape, p_inp2_shape,
            p_cond_shape, num_inp_dims);

    /* Broadcast compatibility check for input1 and input2 */
    for (i = 0; i < MAX_DIMS; i++)
    {
        if ((p_5d_inp1_shape[i] != p_5d_inp2_shape[i])
                && (p_5d_inp1_shape[i] != CONST_ONE)
                && (p_5d_inp2_shape[i] != CONST_ONE))
        {
            return UNSUPPORTED_PARAM;
        }
    }

    /* Getting shapes of intermediate_out_shape */
    WORD32 intermediate_out_shape[MAX_DIMS];
    for (i = MAX_DIMS - CONST_ONE; i >= 0; i--)
    {
        intermediate_out_shape[i] =
                p_5d_inp2_shape[i] == 1 ? p_5d_inp1_shape[i] :
                                          p_5d_inp2_shape[i];
    }

    /* Check shapes for broadcast compatibility */
    WORD32 error = 0;
    error = check_shapes(intermediate_out_shape, p_5d_cond_shape,
            p_5d_out_shape);
    if (error)
    {
        return UNSUPPORTED_PARAM;
    }

    /* Strides calculation */
    WORD32 p_inp1_strides[MAX_DIMS], p_inp2_strides[MAX_DIMS],
            p_cond_strides[MAX_DIMS];
    strides_calculation(p_5d_inp1_shape, p_5d_inp2_shape, p_5d_cond_shape,
            p_inp1_strides, p_inp2_strides, p_cond_strides);

    /* Check for broadcast need */
    WORD32 need_broadcast = 0;
    WORD32 inp1_const = CONST_ONE;
    WORD32 inp2_const = CONST_ONE;
    WORD32 cond_const = CONST_ONE;
    for (i = 0; i < MAX_DIMS; i++)
    {
        /* If shape of input1 and input2 are different */
        if (p_5d_inp1_shape[i] != p_5d_inp2_shape[i])
        {
            /* If the shape at this dimension is one for input1,
             * then the stride at that dimension is made zero
             * for input1, else stride is made zero for input2
             */
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
        /* If the shape of cond and intermediate_out_shape are different */
        if (p_5d_cond_shape[i] != intermediate_out_shape[i])
        {
            /* If shape for cond is one at this dimension */
            if (p_5d_cond_shape[i] == CONST_ONE)
            {
                p_cond_strides[i] = 0;
            }
            /* If shape for cond is not one at this dimension
             * then for both input1 and input2 shape would be one.
             */
            else
            {
                p_inp1_strides[i] = 0;
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
        if (p_5d_cond_shape[i] != CONST_ONE)
        {
            cond_const &= 0;
        }
    }

    const FLOAT32 *__restrict__ p_inp1_base = p_inp1;
    const FLOAT32 *__restrict__ p_inp2_base = p_inp2;
    const UWORD8 *__restrict__ p_cond_base = p_cond;
    FLOAT32 *__restrict__ p_out_base = p_out;

    WORD32 itr0, itr1, itr2, itr3;

    /* If broadcast is not needed */
    if (need_broadcast == 0)
    {
        xa_nn_elm_where_f32xf32_f32(
                p_out_base,
                p_inp1_base,
                p_inp2_base,
                p_cond_base,
                p_5d_out_shape[0] * p_inp1_strides[0]);
    }
    /* If any two vectors are constants */
    else if (((inp1_const == CONST_ONE) && (inp2_const == CONST_ONE)
            && (cond_const == 0))
            || ((inp1_const == CONST_ONE) && (cond_const == CONST_ONE)
                    && (inp2_const == 0))
            || ((inp2_const == CONST_ONE) && (cond_const == CONST_ONE)
                    && (inp1_const == 0)))
    {
        internal_elm_where_two_vecs_const(
                p_out_base,
                p_inp1_base,
                p_inp2_base,
                p_cond_base,
                p_5d_out_shape[0] * p_5d_out_shape[1] * p_5d_out_shape[2]
                        * p_5d_out_shape[3] * p_5d_out_shape[4],
                inp1_const,
                inp2_const,
                cond_const);

    } /* If any two vectors are constants */

    /* If broadcast is needed and
     * the last dimensions of all three vectors are same
     */
    else if (p_inp1_strides[4] == p_inp2_strides[4]
            && p_inp2_strides[4] == p_cond_strides[4])
    {
        WORD32 in_lc, out_lc;
        /* Check if 3rd dim needs to be broadcasted */
        if (p_inp1_strides[3] == 0 || p_inp2_strides[3] == 0
                || p_cond_strides[3] == 0)
        {
            in_lc = p_5d_out_shape[4];
            out_lc = p_5d_out_shape[3];
            /* Repeat the 4th dimension as
             * the 3rd dimension needs to be broadcasted
             */
            for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
            {
                const FLOAT32 *__restrict__ p_inp1_itr0 = p_inp1_base;
                const FLOAT32 *__restrict__ p_inp2_itr0 = p_inp2_base;
                const UWORD8 *__restrict__ p_cond_itr0 = p_cond_base;
                for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
                {
                    const FLOAT32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                    const FLOAT32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                    const UWORD8 *__restrict__ p_cond_itr1 = p_cond_itr0;
                    for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                    {
                        internal_elm_where_broadcast_2D_f32xf32_f32(
                                p_out_base,
                                p_inp1_itr1,
                                p_inp2_itr1,
                                p_cond_itr1,
                                out_lc,
                                in_lc,
                                p_5d_inp1_shape,
                                p_5d_inp2_shape,
                                p_5d_cond_shape);

                        p_out_base += in_lc * out_lc;
                        p_inp1_itr1 += p_inp1_strides[2];
                        p_inp2_itr1 += p_inp2_strides[2];
                        p_cond_itr1 += p_cond_strides[2];
                    }
                    p_inp1_itr0 += p_inp1_strides[1];
                    p_inp2_itr0 += p_inp2_strides[1];
                    p_cond_itr0 += p_cond_strides[1];
                }
                p_inp1_base += p_inp1_strides[0];
                p_inp2_base += p_inp2_strides[0];
                p_cond_base += p_cond_strides[0];
            }

        } /* p_inp1_strides[3] == 0 || p_inp2_strides[3] == 0
         || p_cond_strides == 0 */
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
                const UWORD8 *__restrict__ p_cond_itr0 = p_cond_base;
                for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
                {
                    const FLOAT32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                    const FLOAT32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                    const UWORD8 *__restrict__ p_cond_itr1 = p_cond_itr0;
                    for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                    {
                        xa_nn_elm_where_f32xf32_f32(
                                p_out_base,
                                p_inp1_itr1,
                                p_inp2_itr1,
                                p_cond_itr1,
                                in_lc);
                        p_out_base += in_lc;
                        p_inp1_itr1 += p_inp1_strides[2];
                        p_inp2_itr1 += p_inp2_strides[2];
                        p_cond_itr1 += p_cond_strides[2];
                    }
                    p_inp1_itr0 += p_inp1_strides[1];
                    p_inp2_itr0 += p_inp2_strides[1];
                    p_cond_itr0 += p_cond_strides[1];
                }
                p_inp1_base += p_inp1_strides[0];
                p_inp2_base += p_inp2_strides[0];
                p_cond_base += p_cond_strides[0];
            }
        } /* 3rd and 4th dimensions need not be broadcasted. */

    } /* p_inp1_strides[4] == p_inp2_strides[4]
     && p_inp2_strides[4] == p_cond_strides[4] */
    else
    {
        for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
        {
            const FLOAT32 *__restrict__ p_inp1_itr0 = p_inp1_base;
            const FLOAT32 *__restrict__ p_inp2_itr0 = p_inp2_base;
            const UWORD8 *__restrict__ p_cond_itr0 = p_cond_base;
            for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
            {
                const FLOAT32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                const FLOAT32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                const UWORD8 *__restrict__ p_cond_itr1 = p_cond_itr0;
                for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                {
                    const FLOAT32 *__restrict__ p_inp1_itr2 = p_inp1_itr1;
                    const FLOAT32 *__restrict__ p_inp2_itr2 = p_inp2_itr1;
                    const UWORD8 *__restrict__ p_cond_itr2 = p_cond_itr1;
                    for (itr3 = 0; itr3 < p_5d_out_shape[3]; itr3++)
                    {
                        internal_elm_where_broadcast_1D_scalar_f32xf32_f32(
                                p_out_base,
                                p_inp1_itr2,
                                p_inp2_itr2,
                                p_cond_itr2,
                                p_5d_out_shape[4],
                                p_5d_inp1_shape,
                                p_5d_inp2_shape,
                                p_5d_cond_shape);
                        p_out_base += p_5d_out_shape[4];
                        p_inp1_itr2 += p_inp1_strides[3];
                        p_inp2_itr2 += p_inp2_strides[3];
                        p_cond_itr2 += p_cond_strides[3];
                    }
                    p_inp1_itr1 += p_inp1_strides[2];
                    p_inp2_itr1 += p_inp2_strides[2];
                    p_cond_itr1 += p_cond_strides[2];
                }
                p_inp1_itr0 += p_inp1_strides[1];
                p_inp2_itr0 += p_inp2_strides[1];
                p_cond_itr0 += p_cond_strides[1];
            }
            p_inp1_base += p_inp1_strides[0];
            p_inp2_base += p_inp2_strides[0];
            p_cond_base += p_cond_strides[0];
        }
    }

    return 0;
}
