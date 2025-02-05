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

WORD32 xa_nn_elm_clamp_16_16(WORD16 *p_out,
        const WORD16 *p_inp,
        const WORD16 *p_min,
        const WORD16 *p_max,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_min, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_max, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_INT16, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, SIZE_OF_INT16, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_min, SIZE_OF_INT16, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_max, SIZE_OF_INT16, UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    xb_vec2Mx16 *__restrict__ p_x = (xb_vec2Mx16*) p_inp;
    xb_vec2Mx16 *__restrict__ p_y = (xb_vec2Mx16*) p_min;
    xb_vec2Mx16 *__restrict__ p_z = (xb_vec2Mx16*) p_max;
    xb_vec2Mx16 *__restrict__ p_o = (xb_vec2Mx16*) p_out;

    xb_vec2Mx16 x0, y0, z0;
    valign ax, ay, az, ao;

    WORD32 i, rem;

    /* Load alignment pointers */
    ax = PDX_LA_2MX16_PP(p_x);
    ay = PDX_LA_2MX16_PP(p_y);
    az = PDX_LA_2MX16_PP(p_z);
    ao = PDX_Z_ALIGN();

    /* Main loop for processing 8 elements at a time */
    for (i = 0; i < num_elm >> LOG2_PDX_2M; i++)
    {
        /* Load 8 elements from each of input, min, and max vectors */
        PDX_LA_2MX16_IP(x0, ax, p_x);
        PDX_LA_2MX16_IP(y0, ay, p_y);
        PDX_LA_2MX16_IP(z0, az, p_z);

        /* Ensures the input value is not less than the minimum value */
        x0 = PDX_MAX_2MX16(x0, y0);

        /* Ensures the value from above is not greater than the maximum value */
        x0 = PDX_MIN_2MX16(x0, z0);

        PDX_SA_2MX16_IP(x0, ao, p_o);
    }

    /* Handle remaining elements */
    rem = (num_elm & (PDX_2M - CONST_ONE)) * SIZE_OF_INT16;

    PDX_LAV_2MX16_XP(x0, ax, p_x, rem);
    PDX_LAV_2MX16_XP(y0, ay, p_y, rem);
    PDX_LAV_2MX16_XP(z0, az, p_z, rem);

    x0 = PDX_MAX_2MX16(x0, y0);
    x0 = PDX_MIN_2MX16(x0, z0);

    PDX_SAV_2MX16_XP(x0, ao, p_o, rem);
    PDX_SAPOS_2MX16_FP(ao, p_o);

    return 0;
}
WORD32 xa_nn_elm_clamp_scalar_16_16(WORD16 *p_out,
        const WORD16 *p_inp,
        const WORD16 min,
        const WORD16 max,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_INT16, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, SIZE_OF_INT16, UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    xb_vec2Mx16 *__restrict__ p_x = (xb_vec2Mx16*) p_inp;
    xb_vec2Mx16 *__restrict__ p_o = (xb_vec2Mx16*) p_out;

    valign ax = PDX_LA_2MX16_PP(p_x);
    valign ao = PDX_Z_ALIGN();

    xb_vec2Mx16 min_vec = min;
    xb_vec2Mx16 max_vec = max;
    xb_vec2Mx16 x0;

    WORD32 i;
    /* Main loop for processing 8 elements at a time */
    for (i = 0; i < num_elm >> LOG2_PDX_2M; i++)
    {
        /* Load 8 input elements */
        PDX_LA_2MX16_IP(x0, ax, p_x);

        /* Ensures the input value is not less than the minimum value */
        x0 = PDX_MAX_2MX16(x0, min_vec);

        /* Ensures the value from above is not greater than the maximum value */
        x0 = PDX_MIN_2MX16(x0, max_vec);

        PDX_SA_2MX16_IP(x0, ao, p_o);
    }

    /* Handle remaining elements */
    WORD32 rem = (num_elm & (PDX_2M - CONST_ONE)) * SIZE_OF_INT16;

    PDX_LAV_2MX16_XP(x0, ax, p_x, rem);

    x0 = PDX_MAX_2MX16(x0, min_vec);
    x0 = PDX_MIN_2MX16(x0, max_vec);

    PDX_SAV_2MX16_XP(x0, ao, p_o, rem);
    PDX_SAPOS_2MX16_FP(ao, p_o);

    return 0;
}
static inline void shapes_convert_5D(WORD32 *const __restrict__ p_5d_out_shape,
        WORD32 *const __restrict__ p_5d_inp_shape, /* new inp shapes */
        WORD32 *const __restrict__ p_5d_min_shape, /* new min shapes */
        WORD32 *const __restrict__ p_5d_max_shape, /* new max shapes */
        const WORD32 *const __restrict__ p_out_shape, /* original output shapes */
        const WORD32 *const __restrict__ p_inp_shape, /* original inp shapes */
        const WORD32 *const __restrict__ p_min_shape, /* original min shapes */
        const WORD32 *const __restrict__ p_max_shape, /* original max shapes */
        const WORD32 num_inp_dims)
{
    WORD32 i;
    for (i = 0; i < num_inp_dims; i++)
    {
        p_5d_inp_shape[i + MAX_DIMS - num_inp_dims] = p_inp_shape[i];
        p_5d_min_shape[i + MAX_DIMS - num_inp_dims] = p_min_shape[i];
        p_5d_max_shape[i + MAX_DIMS - num_inp_dims] = p_max_shape[i];
        p_5d_out_shape[i + MAX_DIMS - num_inp_dims] = p_out_shape[i];
    }
} /* shapes_convert_5D */

static inline WORD32 check_shapes(const WORD32 *const p_inp_shape,
        const WORD32 *const p_min_shape,
        const WORD32 *const p_out_shape)
{
    WORD32 i;
    /* Check the shapes of input and output */
    for (i = 0; i < MAX_DIMS; i++)
    {
        if (((p_inp_shape[i] != p_min_shape[i]) && (p_inp_shape[i] != CONST_ONE)
                && (p_min_shape[i] != CONST_ONE))
                || (p_out_shape[i]
                        != (p_inp_shape[i] > p_min_shape[i] ?
                                p_inp_shape[i] : p_min_shape[i])))
        {
            return UNSUPPORTED_PARAM;
        }
    }
    return 0;
} /* check_shapes */

static inline void strides_calculation(const WORD32 *const p_5d_inp_shape,
        const WORD32 *const p_5d_min_shape,
        const WORD32 *const p_5d_max_shape,
        WORD32 *const p_inp_strides,
        WORD32 *const p_min_strides,
        WORD32 *const p_max_strides)
{
    WORD32 i;

    p_inp_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    p_min_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    p_max_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;

    for (i = MAX_DIMS - CONST_TWO; i >= 0; i--)
    {
        p_inp_strides[i] = p_inp_strides[i + CONST_ONE]
                * p_5d_inp_shape[i + CONST_ONE];
        p_min_strides[i] = p_min_strides[i + CONST_ONE]
                * p_5d_min_shape[i + CONST_ONE];
        p_max_strides[i] = p_max_strides[i + CONST_ONE]
                * p_5d_max_shape[i + CONST_ONE];
    }
} /* strides_calculation */

static inline void internal_elm_clamp_broadcast_2D_16_16(
        WORD16 *__restrict__ p_out,
        const WORD16 *__restrict__ p_inp,
        const WORD16 *__restrict__ p_min,
        const WORD16 *__restrict__ p_max,
        WORD32 out_lc,
        WORD32 in_lc,
        const WORD32 *p_input_shapes,
        const WORD32 *p_min_shapes,
        const WORD32 *p_max_shapes)
{
    xb_vec2Mx16 x0, x1, y0, y1, z0, z1, r0, r1, r;

    /* Align registers for input, min and max */
    valign ax0, ax1, ay0, ay1, az0, az1;
    /* align register for output */
    valign ar0, ar1;

    /* Pointer for base address for input */
    const xb_vec2Mx16 *__restrict__ p_x0 = (const xb_vec2Mx16*) p_inp;
    /* Pointer for middle address for input */
    const xb_vec2Mx16 *__restrict__ p_x1 = (const xb_vec2Mx16*) (p_inp
            + ((out_lc / CONST_TWO) * in_lc));

    /* Priming for input loads */
    ax0 = PDX_LA_2MX16_PP(p_x0);
    ax1 = PDX_LA_2MX16_PP(p_x1);

    /* Pointer for base address for min */
    const xb_vec2Mx16 *__restrict__ p_y0 = (const xb_vec2Mx16*) p_min;
    /* Pointer for middle address for min */
    const xb_vec2Mx16 *__restrict__ p_y1 = (const xb_vec2Mx16*) (p_min
            + ((out_lc / CONST_TWO) * in_lc));

    /* Priming for min loads */
    ay0 = PDX_LA_2MX16_PP(p_y0);
    ay1 = PDX_LA_2MX16_PP(p_y1);

    /* Pointer for base address for max */
    const xb_vec2Mx16 *__restrict__ p_z0 = (const xb_vec2Mx16*) p_max;
    /* Pointer for middle address for max */
    const xb_vec2Mx16 *__restrict__ p_z1 = (const xb_vec2Mx16*) (p_max
            + ((out_lc / CONST_TWO) * in_lc));

    /* Priming for max loads */
    az0 = PDX_LA_2MX16_PP(p_z0);
    az1 = PDX_LA_2MX16_PP(p_z1);

    /* Pointer for base address for output */
    xb_vec2Mx16 *__restrict__ p_r0 = (xb_vec2Mx16*) p_out;
    /* Pointer for middle address for output */
    xb_vec2Mx16 *__restrict__ p_r1 = (xb_vec2Mx16*) (p_out
            + ((out_lc / CONST_TWO) * in_lc));

    /* Priming for output stores */
    ar0 = PDX_Z_ALIGN();
    ar1 = PDX_Z_ALIGN();

    WORD32 i, j, rem;
    rem = (in_lc & (PDX_2M - CONST_ONE)) * SIZE_OF_INT16;

    /* If shape of inp and min are same but
     * different from max at dimension 3
     */
    if ((p_input_shapes[3] == p_min_shapes[3])
            && (p_input_shapes[3] != p_max_shapes[3]))
    {
        /* if dimension 3 of max is broadcastable */
        if (p_max_shapes[3] == CONST_ONE)
        {
            /* Unroll the loop by x2 for SIMD */
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    /* Load 8 elements from max */
                    PDX_LA_2MX16_IP(z0, az0, p_z0);

                    /* Load the 8 elements from inp base address */
                    PDX_LA_2MX16_IP(x0, ax0, p_x0);
                    /* Load the 8 elements from inp Middle address */
                    PDX_LA_2MX16_IP(x1, ax1, p_x1);

                    /* Load the 8 elements from min base address */
                    PDX_LA_2MX16_IP(y0, ay0, p_y0);
                    /* Load the 8 elements from min Middle address */
                    PDX_LA_2MX16_IP(y1, ay1, p_y1);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r0 = PDX_MAX_2MX16(x0, y0);
                    r0 = PDX_MIN_2MX16(r0, z0);

                    r1 = PDX_MAX_2MX16(x1, y1);
                    r1 = PDX_MIN_2MX16(r1, z0);

                    PDX_SA_2MX16_IP(r0, ar0, p_r0);
                    PDX_SA_2MX16_IP(r1, ar1, p_r1);
                }
                /* Remaining iterations of inner loop */
                PDX_LAV_2MX16_XP(z0, az0, p_z0, rem);

                PDX_LAV_2MX16_XP(x0, ax0, p_x0, rem);
                PDX_LAV_2MX16_XP(x1, ax1, p_x1, rem);

                PDX_LAV_2MX16_XP(y0, ay0, p_y0, rem);
                PDX_LAV_2MX16_XP(y1, ay1, p_y1, rem);

                r0 = PDX_MAX_2MX16(x0, y0);
                r0 = PDX_MIN_2MX16(r0, z0);

                r1 = PDX_MAX_2MX16(x1, y1);
                r1 = PDX_MIN_2MX16(r1, z0);

                PDX_SAV_2MX16_XP(r0, ar0, p_r0, rem);
                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);

                /* max Pointer updates to base address
                 * as max is broadcasted
                 */
                p_z0 = (const xb_vec2Mx16*) p_max;
                az0 = PDX_LA_2MX16_PP(p_z0);
            }

            /* Loop through remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    /* Load 8 elements from max */
                    PDX_LA_2MX16_IP(z0, az0, p_z0);
                    /* Load the 8 elements from inp Middle address */
                    PDX_LA_2MX16_IP(x1, ax1, p_x1);
                    /* Load the 8 elements from min Middle address */
                    PDX_LA_2MX16_IP(y1, ay1, p_y1);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r1 = PDX_MAX_2MX16(x1, y1);
                    r1 = PDX_MIN_2MX16(r1, z0);

                    PDX_SA_2MX16_IP(r1, ar1, p_r1);
                }
                /* Remaining iterations */
                PDX_LAV_2MX16_XP(z0, az0, p_z0, rem);

                PDX_LAV_2MX16_XP(x1, ax1, p_x1, rem);

                PDX_LAV_2MX16_XP(y1, ay1, p_y1, rem);

                r1 = PDX_MAX_2MX16(x1, y1);
                r1 = PDX_MIN_2MX16(r1, z0);

                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);
            }
        }/* p_max_shapes[3] == CONST_ONE */

        /* If dimension 3 of inp and min are broadcastable */
        else
        {
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    /* Load 8 elements of inp */
                    PDX_LA_2MX16_IP(x0, ax0, p_x0);

                    /* Load 8 elements of min */
                    PDX_LA_2MX16_IP(y0, ay0, p_y0);

                    /* Load the 8 elements from max base address */
                    PDX_LA_2MX16_IP(z0, az0, p_z0);
                    /* Load the 8 elements from max Middle address */
                    PDX_LA_2MX16_IP(z1, az1, p_z1);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r = PDX_MAX_2MX16(x0, y0);
                    r0 = PDX_MIN_2MX16(r, z0);

                    r1 = PDX_MIN_2MX16(r, z1);

                    PDX_SA_2MX16_IP(r0, ar0, p_r0);
                    PDX_SA_2MX16_IP(r1, ar1, p_r1);
                }
                /* Remaining iterations of inner loop */
                PDX_LAV_2MX16_XP(x0, ax0, p_x0, rem);

                PDX_LAV_2MX16_XP(y0, ay0, p_y0, rem);

                PDX_LAV_2MX16_XP(z0, az0, p_z0, rem);
                PDX_LAV_2MX16_XP(z1, az1, p_z1, rem);

                r = PDX_MAX_2MX16(x0, y0);
                r0 = PDX_MIN_2MX16(r, z0);

                r1 = PDX_MIN_2MX16(r, z1);

                PDX_SAV_2MX16_XP(r0, ar0, p_r0, rem);
                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);

                /* inp and min Pointers update to base address
                 * as inp and min are broadcasted
                 */
                p_x0 = (const xb_vec2Mx16*) p_inp;
                p_y0 = (const xb_vec2Mx16*) p_min;

                ax0 = PDX_LA_2MX16_PP(p_x0);
                ay0 = PDX_LA_2MX16_PP(p_y0);
            }
            /* Loop through remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    /* Load 8 elements of inp */
                    PDX_LA_2MX16_IP(x0, ax0, p_x0);
                    /* Load 8 elements of min */
                    PDX_LA_2MX16_IP(y0, ay0, p_y0);
                    /* Load the 8 elements from max Middle address */
                    PDX_LA_2MX16_IP(z1, az1, p_z1);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r1 = PDX_MAX_2MX16(x0, y0);
                    r1 = PDX_MIN_2MX16(r1, z1);

                    PDX_SA_2MX16_IP(r1, ar1, p_r1);
                }
                /* Remaining iterations */
                PDX_LAV_2MX16_XP(x0, ax0, p_x0, rem);

                PDX_LAV_2MX16_XP(y0, ay0, p_y0, rem);

                PDX_LAV_2MX16_XP(z1, az1, p_z1, rem);

                r1 = PDX_MAX_2MX16(x0, y0);
                r1 = PDX_MIN_2MX16(r1, z1);

                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);
            }

        }
    }/* (p_input_shapes[3] == p_min_shapes[3])
            && (inp_shapes[3] != p_max_shapes[3]) */

    /* If shape of inp and max are same but
     * different from min at dimension 3
     */
    else if ((p_input_shapes[3] == p_max_shapes[3])
            && (p_input_shapes[3] != p_min_shapes[3]))
    {
        if (p_min_shapes[3] == CONST_ONE)
        {
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    /* Load 8 elements from min */
                    PDX_LA_2MX16_IP(y0, ay0, p_y0);

                    /* Load 8 elements from inp base address */
                    PDX_LA_2MX16_IP(x0, ax0, p_x0);
                    /* Load 8 elements from inp middle address */
                    PDX_LA_2MX16_IP(x1, ax1, p_x1);

                    /* Load 8 elements from max base address */
                    PDX_LA_2MX16_IP(z0, az0, p_z0);
                    /* Load 8 elements from max middle address */
                    PDX_LA_2MX16_IP(z1, az1, p_z1);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r0 = PDX_MAX_2MX16(x0, y0);
                    r0 = PDX_MIN_2MX16(r0, z0);

                    r1 = PDX_MAX_2MX16(x1, y0);
                    r1 = PDX_MIN_2MX16(r1, z1);

                    PDX_SA_2MX16_IP(r0, ar0, p_r0);
                    PDX_SA_2MX16_IP(r1, ar1, p_r1);
                }
                /* Remaining iterations of inner loop */
                PDX_LAV_2MX16_XP(y0, ay0, p_y0, rem);

                PDX_LAV_2MX16_XP(x0, ax0, p_x0, rem);
                PDX_LAV_2MX16_XP(x1, ax1, p_x1, rem);

                PDX_LAV_2MX16_XP(z0, az0, p_z0, rem);
                PDX_LAV_2MX16_XP(z1, az1, p_z1, rem);

                r0 = PDX_MAX_2MX16(x0, y0);
                r0 = PDX_MIN_2MX16(r0, z0);

                r1 = PDX_MAX_2MX16(x1, y0);
                r1 = PDX_MIN_2MX16(r1, z1);

                PDX_SAV_2MX16_XP(r0, ar0, p_r0, rem);
                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);

                /* min Pointer updates to base address
                 * as min is broadcasted
                 */
                p_y0 = (const xb_vec2Mx16*) p_min;
                ay0 = PDX_LA_2MX16_PP(p_y0);
            }
            /* Remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    PDX_LA_2MX16_IP(y0, ay0, p_y0);
                    PDX_LA_2MX16_IP(x1, ax1, p_x1);
                    PDX_LA_2MX16_IP(z1, az1, p_z1);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r1 = PDX_MAX_2MX16(x1, y0);
                    r1 = PDX_MIN_2MX16(r1, z1);

                    PDX_SA_2MX16_IP(r1, ar1, p_r1);
                }
                PDX_LAV_2MX16_XP(y0, ay0, p_y0, rem);
                PDX_LAV_2MX16_XP(x1, ax1, p_x1, rem);
                PDX_LAV_2MX16_XP(z1, az1, p_z1, rem);

                r1 = PDX_MAX_2MX16(x1, y0);
                r1 = PDX_MIN_2MX16(r1, z1);

                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);

            }

        }/* p_min_shapes[3] == CONST_ONE */

        /* If dimension 3 of inp and max are broadcastable */
        else
        {
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    /* Load 8 elements of inp */
                    PDX_LA_2MX16_IP(x0, ax0, p_x0);

                    /* Load 8 elements of max */
                    PDX_LA_2MX16_IP(z0, az0, p_z0);

                    /* Load 8 elements from min base address */
                    PDX_LA_2MX16_IP(y0, ay0, p_y0);
                    /* Load 8 elements from min middle address */
                    PDX_LA_2MX16_IP(y1, ay1, p_y1);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r0 = PDX_MAX_2MX16(x0, y0);
                    r0 = PDX_MIN_2MX16(r0, z0);

                    r1 = PDX_MAX_2MX16(x0, y1);
                    r1 = PDX_MIN_2MX16(r1, z0);

                    PDX_SA_2MX16_IP(r0, ar0, p_r0);
                    PDX_SA_2MX16_IP(r1, ar1, p_r1);

                }
                /* Remaining iterations of inner loop */
                PDX_LAV_2MX16_XP(x0, ax0, p_x0, rem);
                PDX_LAV_2MX16_XP(z0, az0, p_z0, rem);

                PDX_LAV_2MX16_XP(y0, ay0, p_y0, rem);
                PDX_LAV_2MX16_XP(y1, ay1, p_y1, rem);

                r0 = PDX_MAX_2MX16(x0, y0);
                r0 = PDX_MIN_2MX16(r0, z0);

                r1 = PDX_MAX_2MX16(x0, y1);
                r1 = PDX_MIN_2MX16(r1, z0);

                PDX_SAV_2MX16_XP(r0, ar0, p_r0, rem);
                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);

                /* inp and max Pointers update to base address
                 * as inp and max are broadcasted
                 */
                p_x0 = (const xb_vec2Mx16*) p_inp;
                p_z0 = (const xb_vec2Mx16*) p_max;

                ax0 = PDX_LA_2MX16_PP(p_x0);
                az0 = PDX_LA_2MX16_PP(p_z0);

            }
            /* Remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    PDX_LA_2MX16_IP(x0, ax0, p_x0);
                    PDX_LA_2MX16_IP(z0, az0, p_z0);
                    PDX_LA_2MX16_IP(y1, ay1, p_y1);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r1 = PDX_MAX_2MX16(x0, y1);
                    r1 = PDX_MIN_2MX16(r1, z0);

                    PDX_SA_2MX16_IP(r1, ar1, p_r1);
                }
                PDX_LAV_2MX16_XP(x0, ax0, p_x0, rem);
                PDX_LAV_2MX16_XP(z0, az0, p_z0, rem);
                PDX_LAV_2MX16_XP(y1, ay1, p_y1, rem);

                r1 = PDX_MAX_2MX16(x0, y1);
                r1 = PDX_MIN_2MX16(r1, z0);

                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);
            }

        }

    }/* (p_input_shapes[3] == p_max_shapes[3])
            && (p_input_shapes[3] != p_min_shapes[3]) */


    /* If shape of min and max are same but
     * different from inp at dimension 3
     */
    else if ((p_min_shapes[3] == p_max_shapes[3])
            && (p_input_shapes[3] != p_min_shapes[3]))
    {
        if (p_input_shapes[3] == CONST_ONE)
        {
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    /* Load 8 elements of inp */
                    PDX_LA_2MX16_IP(x0, ax0, p_x0);

                    /* Load 8 elements from min base address */
                    PDX_LA_2MX16_IP(y0, ay0, p_y0);
                    /* Load 8 elements from min middle address */
                    PDX_LA_2MX16_IP(y1, ay1, p_y1);

                    /* Load 8 elements from max base address */
                    PDX_LA_2MX16_IP(z0, az0, p_z0);
                    /* Load 8 elements from max middle address */
                    PDX_LA_2MX16_IP(z1, az1, p_z1);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r0 = PDX_MAX_2MX16(x0, y0);
                    r0 = PDX_MIN_2MX16(r0, z0);

                    r1 = PDX_MAX_2MX16(x0, y1);
                    r1 = PDX_MIN_2MX16(r1, z1);

                    PDX_SA_2MX16_IP(r0, ar0, p_r0);
                    PDX_SA_2MX16_IP(r1, ar1, p_r1);

                }
                /* Remaining iterations of inner loop */
                PDX_LAV_2MX16_XP(x0, ax0, p_x0, rem);
                PDX_LAV_2MX16_XP(y0, ay0, p_y0, rem);
                PDX_LAV_2MX16_XP(y1, ay1, p_y1, rem);
                PDX_LAV_2MX16_XP(z0, az0, p_z0, rem);
                PDX_LAV_2MX16_XP(z1, az1, p_z1, rem);

                r0 = PDX_MAX_2MX16(x0, y0);
                r0 = PDX_MIN_2MX16(r0, z0);

                r1 = PDX_MAX_2MX16(x0, y1);
                r1 = PDX_MIN_2MX16(r1, z1);

                PDX_SAV_2MX16_XP(r0, ar0, p_r0, rem);
                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);

                /* inp Pointer updates to base address
                 * as inp is broadcasted
                 */
                p_x0 = (const xb_vec2Mx16*) p_inp;
                ax0 = PDX_LA_2MX16_PP(p_x0);

            }
            /* Remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    PDX_LA_2MX16_IP(x0, ax0, p_x0);
                    PDX_LA_2MX16_IP(y1, ay1, p_y1);
                    PDX_LA_2MX16_IP(z1, az1, p_z1);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r1 = PDX_MAX_2MX16(x0, y1);
                    r1 = PDX_MIN_2MX16(r1, z1);

                    PDX_SA_2MX16_IP(r1, ar1, p_r1);
                }
                PDX_LAV_2MX16_XP(x0, ax0, p_x0, rem);
                PDX_LAV_2MX16_XP(y1, ay1, p_y1, rem);
                PDX_LAV_2MX16_XP(z1, az1, p_z1, rem);

                r1 = PDX_MAX_2MX16(x0, y1);
                r1 = PDX_MIN_2MX16(r1, z1);

                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);

            }

        }/* p_input_shapes[3] == CONST_ONE */

        /* If dimension 3 of min and max are broadcastable */
        else
        {
            for (i = 0; i < (out_lc - CONST_ONE); i += CONST_TWO)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    /* Load 8 elements from inp base address */
                    PDX_LA_2MX16_IP(x0, ax0, p_x0);
                    /* Load 8 elements from inp base address */
                    PDX_LA_2MX16_IP(x1, ax1, p_x1);

                    /* Load 8 elements of min */
                    PDX_LA_2MX16_IP(y0, ay0, p_y0);
                    /* Load 8 elements of max */
                    PDX_LA_2MX16_IP(z0, az0, p_z0);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r0 = PDX_MAX_2MX16(x0, y0);
                    r0 = PDX_MIN_2MX16(r0, z0);

                    r1 = PDX_MAX_2MX16(x1, y0);
                    r1 = PDX_MIN_2MX16(r1, z0);

                    PDX_SA_2MX16_IP(r0, ar0, p_r0);
                    PDX_SA_2MX16_IP(r1, ar1, p_r1);

                }
                /* Remaining iterations of inner loop */
                PDX_LAV_2MX16_XP(x0, ax0, p_x0, rem);
                PDX_LAV_2MX16_XP(x1, ax1, p_x1, rem);
                PDX_LAV_2MX16_XP(y0, ay0, p_y0, rem);
                PDX_LAV_2MX16_XP(z0, az0, p_z0, rem);

                r0 = PDX_MAX_2MX16(x0, y0);
                r0 = PDX_MIN_2MX16(r0, z0);

                r1 = PDX_MAX_2MX16(x1, y0);
                r1 = PDX_MIN_2MX16(r1, z0);

                PDX_SAV_2MX16_XP(r0, ar0, p_r0, rem);
                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);

                /* min and max Pointers update to base address
                 * as min and max are broadcasted
                 */
                p_y0 = (const xb_vec2Mx16*) p_min;
                p_z0 = (const xb_vec2Mx16*) p_max;

                ay0 = PDX_LA_2MX16_PP(p_y0);
                az0 = PDX_LA_2MX16_PP(p_z0);
            }
            /* Remaining iterations of outer loop */
            if ((out_lc % CONST_TWO) != 0)
            {
                for (j = 0; j < in_lc >> LOG2_PDX_2M; j++)
                {
                    PDX_LA_2MX16_IP(x1, ax1, p_x1);
                    PDX_LA_2MX16_IP(y0, ay0, p_y0);
                    PDX_LA_2MX16_IP(z0, az0, p_z0);

                    /* Clamps the input value between the specified
                     *  minimum and maximum values
                     */
                    r1 = PDX_MAX_2MX16(x1, y0);
                    r1 = PDX_MIN_2MX16(r1, z0);

                    PDX_SA_2MX16_IP(r1, ar1, p_r1);
                }
                PDX_LAV_2MX16_XP(x1, ax1, p_x1, rem);
                PDX_LAV_2MX16_XP(y0, ay0, p_y0, rem);
                PDX_LAV_2MX16_XP(z0, az0, p_z0, rem);

                r1 = PDX_MAX_2MX16(x1, y0);
                r1 = PDX_MIN_2MX16(r1, z0);

                PDX_SAV_2MX16_XP(r1, ar1, p_r1, rem);

            }

        }
    }/* (p_min_shapes[3] == p_max_shapes[3])
            && (p_input_shapes[3] != p_min_shapes[3]) */

    /* Flushing output align registers */
    PDX_SAPOS_2MX16_FP(ar0, p_r0);
    PDX_SAPOS_2MX16_FP(ar1, p_r1);

}

static inline void internal_elm_clamp_broadcast_1D_scalar_16_16(
        WORD16 *__restrict__ p_out,
        const WORD16 *__restrict__ p_inp,
        const WORD16 *__restrict__ p_min,
        const WORD16 *__restrict__ p_max,
        WORD32 num_elm,
        const WORD32 *p_input_shapes,
        const WORD32 *p_min_shapes,
        const WORD32 *p_max_shapes)
{
    xb_vec2Mx16 x, y, z, r, r0;

    valign ax, ay, az;
    valign ar;

    /* Pointer for base address for inp */
    const xb_vec2Mx16 *__restrict__ p_x = (const xb_vec2Mx16*) p_inp;

    /* Priming for inp load */
    ax = PDX_LA_2MX16_PP(p_x);

    /* Pointer for base address for min */
    const xb_vec2Mx16 *__restrict__ p_y = (const xb_vec2Mx16*) p_min;

    /* Priming for min load */
    ay = PDX_LA_2MX16_PP(p_y);

    /* Pointer for base address for max */
    const xb_vec2Mx16 *__restrict__ p_z = (const xb_vec2Mx16*) p_max;

    /* Priming for max load */
    az = PDX_LA_2MX16_PP(p_z);

    /* Pointer for base address for output */
    xb_vec2Mx16 *__restrict__ p_r = (xb_vec2Mx16*) p_out;

    /* Priming for ouput store */
    ar = PDX_Z_ALIGN();

    WORD32 i, rem;
    rem = (num_elm & (PDX_2M - CONST_ONE)) * SIZE_OF_INT16;

    /* if shape of inp and min are same but
     * different from max at dimension 4
     */
    if ((p_input_shapes[4] == p_min_shapes[4])
            && (p_input_shapes[4] != p_max_shapes[4]))
    {
        /* If dimension 4 of max is broadcastable */
        if (p_max_shapes[4] == CONST_ONE)
        {
            z = p_max[0];
            for (i = 0; i < num_elm >> LOG2_PDX_2M; i++)
            {
                /* Load 8 elements of inp */
                PDX_LA_2MX16_IP(x, ax, p_x);
                /* Load 8 elements of min */
                PDX_LA_2MX16_IP(y, ay, p_y);
                /* Clamps the input value between the specified
                 *  minimum and maximum values
                 */
                r = PDX_MAX_2MX16(x, y);
                r = PDX_MIN_2MX16(r, z);
                PDX_SA_2MX16_IP(r, ar, p_r);
            }
            /* Remaining iterations */
            PDX_LAV_2MX16_XP(x, ax, p_x, rem);
            PDX_LAV_2MX16_XP(y, ay, p_y, rem);
            r = PDX_MAX_2MX16(x, y);
            r = PDX_MIN_2MX16(r, z);
            PDX_SAV_2MX16_XP(r, ar, p_r, rem);
        }/* p_max_shapes[4] == CONST_ONE */

        /* If dimension 4 of inp and min are broadcastable */
        else
        {
            x = p_inp[0];
            y = p_min[0];

            r0 = PDX_MAX_2MX16(x, y);
            for (i = 0; i < num_elm >> LOG2_PDX_2M; i++)
            {
                /* Load 8 elements of max */
                PDX_LA_2MX16_IP(z, az, p_z);
                /* Clamps the input value between the specified
                 *  minimum and maximum values
                 */
                r = PDX_MIN_2MX16(r0, z);
                PDX_SA_2MX16_IP(r, ar, p_r);
            }
            /* Remaining iterations */
            PDX_LAV_2MX16_XP(z, az, p_z, rem);
            r = PDX_MIN_2MX16(r0, z);
            PDX_SAV_2MX16_XP(r, ar, p_r, rem);
        }/* p_max_shapes[4] != CONST_ONE */
    }

    /* if shape of inp and max are same but
     * different from min at dimension 4
     */
    else if ((p_input_shapes[4] == p_max_shapes[4])
            && (p_input_shapes[4] != p_min_shapes[4]))
    {
        /* If dimension 4 of min is broadcastable */
        if (p_min_shapes[4] == CONST_ONE)
        {
            y = p_min[0];
            for (i = 0; i < num_elm >> LOG2_PDX_2M; i++)
            {
                /* Load 8 elements of inp */
                PDX_LA_2MX16_IP(x, ax, p_x);
                /* Load 8 elements of max */
                PDX_LA_2MX16_IP(z, az, p_z);
                /* Clamps the input value between the specified
                 *  minimum and maximum values
                 */
                r = PDX_MAX_2MX16(x, y);
                r = PDX_MIN_2MX16(r, z);
                PDX_SA_2MX16_IP(r, ar, p_r);
            }
            /* Remaining iterations */
            PDX_LAV_2MX16_XP(x, ax, p_x, rem);
            PDX_LAV_2MX16_XP(z, az, p_z, rem);
            r = PDX_MAX_2MX16(x, y);
            r = PDX_MIN_2MX16(r, z);
            PDX_SAV_2MX16_XP(r, ar, p_r, rem);
        }/* p_min_shapes[4] == CONST_ONE */

        /* If dimension 4 of inp and max are broadcastable */
        else
        {
            x = p_inp[0];
            z = p_max[0];

            for (i = 0; i < num_elm >> LOG2_PDX_2M; i++)
            {
                /* Load 8 elements of min */
                PDX_LA_2MX16_IP(y, ay, p_y);
                /* Clamps the input value between the specified
                 *  minimum and maximum values
                 */
                r = PDX_MAX_2MX16(x, y);
                r = PDX_MIN_2MX16(r, z);
                PDX_SA_2MX16_IP(r, ar, p_r);
            }
            /* Remaining iterations */
            PDX_LAV_2MX16_XP(y, ay, p_y, rem);
            r = PDX_MAX_2MX16(x, y);
            r = PDX_MIN_2MX16(r, z);
            PDX_SAV_2MX16_XP(r, ar, p_r, rem);
        }/* p_min_shapes[4] != CONST_ONE */
    }

    /* if shape of min and max are same but
     * different from inp at dimension 4
     */
    else if ((p_min_shapes[4] == p_max_shapes[4])
            && (p_input_shapes[4] != p_min_shapes[4]))
    {
        /* If dimension 4 of inp is broadcastable */
        if (p_input_shapes[4] == CONST_ONE)
        {
            x = p_inp[0];
            for (i = 0; i < num_elm >> LOG2_PDX_2M; i++)
            {
                /* Load 8 elements of min */
                PDX_LA_2MX16_IP(y, ay, p_y);
                PDX_LA_2MX16_IP(z, az, p_z);

                /* Clamps the input value between the specified
                 *  minimum and maximum values
                 */
                r = PDX_MAX_2MX16(x, y);
                r = PDX_MIN_2MX16(r, z);
                PDX_SA_2MX16_IP(r, ar, p_r);
            }
            /* Remaining iterations */
            PDX_LAV_2MX16_XP(y, ay, p_y, rem);
            PDX_LAV_2MX16_XP(z, az, p_z, rem);
            r = PDX_MAX_2MX16(x, y);
            r = PDX_MIN_2MX16(r, z);
            PDX_SAV_2MX16_XP(r, ar, p_r, rem);
        }/* p_input_shapes[4] == CONST_ONE */

        /* If dimension 4 of min and max are broadcastable */
        else
        {
            y = p_min[0];
            z = p_max[0];

            for (i = 0; i < num_elm >> LOG2_PDX_2M; i++)
            {
                /* Load 8 elements of inp */
                PDX_LA_2MX16_IP(x, ax, p_x);
                /* Clamps the input value between the specified
                 *  minimum and maximum values
                 */
                r = PDX_MAX_2MX16(x, y);
                r = PDX_MIN_2MX16(r, z);

                PDX_SA_2MX16_IP(r, ar, p_r);
            }
            /* Remaining iterations */
            PDX_LAV_2MX16_XP(x, ax, p_x, rem);
            r = PDX_MAX_2MX16(x, y);
            r = PDX_MIN_2MX16(r, z);
            PDX_SAV_2MX16_XP(r, ar, p_r, rem);
        }/* (p_min_shapes[4] == p_max_shapes[4])
            && (inp_shapes[4] != p_min_shapes[4]) */
    }
    /* Flushing output align register */
    PDX_SAPOS_2MX16_FP(ar, p_r);

}

static inline void internal_elm_clamp_two_vecs_const(
        WORD16 *__restrict__ p_out,
        const WORD16 *__restrict__ p_inp,
        const WORD16 *__restrict__ p_min,
        const WORD16 *__restrict__ p_max,
        WORD32 num_elm,
        WORD32 inp_const,
        WORD32 min_const,
        WORD32 max_const)
{
    xb_vec2Mx16 x, y, z, r, r0;

    valign ax, ay, az;
    valign ar;

    /* Pointer for base address for inp */
    const xb_vec2Mx16 *__restrict__ p_x = (const xb_vec2Mx16*) p_inp;

    /* Priming for inp load */
    ax = PDX_LA_2MX16_PP(p_x);

    /* Pointer for base address for min */
    const xb_vec2Mx16 *__restrict__ p_y = (const xb_vec2Mx16*) p_min;

    /* Priming for min load */
    ay = PDX_LA_2MX16_PP(p_y);

    /* Pointer for base address for max */
    const xb_vec2Mx16 *__restrict__ p_z = (const xb_vec2Mx16*) p_max;

    az = PDX_LA_2MX16_PP(p_z);

    /* Pointer for base address for output */
    xb_vec2Mx16 *__restrict__ p_r = (xb_vec2Mx16*) p_out;

    /* Priming for ouput store */
    ar = PDX_Z_ALIGN();

    WORD32 i, rem;
    rem = (num_elm & (PDX_2M - CONST_ONE)) * SIZE_OF_INT16;

    /* If both inp and min are constants but max is not constant */
    if (((inp_const == CONST_ONE) && (min_const == CONST_ONE)
            && (max_const == 0)))
    {
        x = p_inp[0];
        y = p_min[0];

        r0 = PDX_MAX_2MX16(x, y);
        for (i = 0; i < num_elm >> LOG2_PDX_2M; i++)
        {
            /* Load 8 elements of max */
            PDX_LA_2MX16_IP(z, az, p_z);
            /* Clamps the input value between the specified
             *  minimum and maximum values
             */
            r = PDX_MIN_2MX16(r0, z);
            PDX_SA_2MX16_IP(r, ar, p_r);
        }
        /* Remaining iterations */
        PDX_LAV_2MX16_XP(z, az, p_z, rem);
        r = PDX_MIN_2MX16(r0, z);
        PDX_SAV_2MX16_XP(r, ar, p_r, rem);

    }/* If both inp and min are constants but max is not constant */

    /* If both inp and max are constants but min is not constant */
    else if (((inp_const == CONST_ONE) && (max_const == CONST_ONE)
            && (min_const == 0)))
    {
        x = p_inp[0];
        z = p_max[0];

        for (i = 0; i < num_elm >> LOG2_PDX_2M; i++)
        {
            /* Load 8 elements of min */
            PDX_LA_2MX16_IP(y, ay, p_y);
            /* Clamps the input value between the specified
             *  minimum and maximum values
             */
            r = PDX_MAX_2MX16(x, y);
            r = PDX_MIN_2MX16(r, z);
            PDX_SA_2MX16_IP(r, ar, p_r);
        }
        /* Remaining iterations */
        PDX_LAV_2MX16_XP(y, ay, p_y, rem);
        r = PDX_MAX_2MX16(x, y);
        r = PDX_MIN_2MX16(r, z);
        PDX_SAV_2MX16_XP(r, ar, p_r, rem);

    } /* If both inp and max are constants but min is not constant */

    /* Flushing output align register */
    PDX_SAPOS_2MX16_FP(ar, p_r);

}

WORD32 xa_nn_elm_clamp_broadcast_5D_16_16(WORD16 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        const WORD16 *__restrict__ p_inp,
        const WORD32 *const p_inp_shape,
        const WORD16 *__restrict__ p_min,
        const WORD32 *const p_min_shape,
        const WORD16 *__restrict__ p_max,
        const WORD32 *const p_max_shape,
        WORD32 num_inp_dims)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_min, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_min_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_max, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_max_shape, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, SIZE_OF_INT16, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, SIZE_OF_INT16, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_min, SIZE_OF_INT16, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_min_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_max, SIZE_OF_INT16, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_max_shape, SIZE_OF_INT, UNSUPPORTED_PARAM);

    /* num_inp_dims should be greater than 0 and less than or equal to 5 */
    XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);

    WORD32 i, index;

    /* Shape Checks */
    /* Shapes should be greater than zero */
    for (i = 0; i < num_inp_dims; i++)
    {
        XA_NNLIB_ARG_CHK_COND((p_out_shape[i] <= 0), UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_COND((p_inp_shape[i] <= 0), UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_COND((p_min_shape[i] <= 0), UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_COND((p_max_shape[i] <= 0), UNSUPPORTED_PARAM);
    }

    /* 5D shapes initialization */
    WORD32 p_5d_out_shape[MAX_DIMS] = { CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE };
    WORD32 p_5d_inp_shape[MAX_DIMS] = { CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE };
    WORD32 p_5d_max_shape[MAX_DIMS] = { CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE };
    WORD32 p_5d_min_shape[MAX_DIMS] = { CONST_ONE, CONST_ONE, CONST_ONE,
    CONST_ONE, CONST_ONE };

    shapes_convert_5D(p_5d_out_shape, p_5d_inp_shape, p_5d_min_shape,
            p_5d_max_shape, p_out_shape, p_inp_shape, p_min_shape, p_max_shape,
            num_inp_dims);

    /* Broadcast compatibility check for inp and min */
    for (i = 0; i < MAX_DIMS; i++)
    {
        if ((p_5d_inp_shape[i] != p_5d_min_shape[i])
                && (p_5d_inp_shape[i] != CONST_ONE)
                && (p_5d_min_shape[i] != CONST_ONE))
        {
            return UNSUPPORTED_PARAM;
        }
    }

    /* Getting shapes of intermediate_out_shape */
    WORD32 intermediate_out_shape[MAX_DIMS];
    for (index = MAX_DIMS - CONST_ONE; index >= 0; index--)
    {
        intermediate_out_shape[index] =
                p_5d_min_shape[index] == 1 ?
                        p_5d_inp_shape[index] : p_5d_min_shape[index];
    }

    /* Check shapes for broadcast compatibility */
    WORD32 error = 0;
    error = check_shapes(intermediate_out_shape, p_5d_max_shape,
            p_5d_out_shape);
    if (error)
    {
        return UNSUPPORTED_PARAM;
    }

    /* Strides calculation */
    WORD32 p_inp_strides[MAX_DIMS], p_min_strides[MAX_DIMS], p_max_strides[MAX_DIMS];

    strides_calculation(p_5d_inp_shape, p_5d_min_shape, p_5d_max_shape,
            p_inp_strides, p_min_strides, p_max_strides);

    /* Check for broadcast need */
    WORD32 need_broadcast = 0;
    WORD32 inp_const = CONST_ONE;
    WORD32 min_const = CONST_ONE;
    WORD32 max_const = CONST_ONE;
    for (i = 0; i < MAX_DIMS; i++)
    {
        /* If shape of inp and min are different */
        if (p_5d_inp_shape[i] != p_5d_min_shape[i])
        {
            /* If the shape at this dimension is one for inp,
             * then the stride at that dimension is made zero
             * for inp, else stride is made zero for min
             */
            if (p_5d_inp_shape[i] == CONST_ONE)
            {
                p_inp_strides[i] = 0;
            }
            else
            {
                p_min_strides[i] = 0;
            }
            need_broadcast = CONST_ONE;
        }
        /* If the shape of max and intermediate_out_shape are different */
        if (p_5d_max_shape[i] != intermediate_out_shape[i])
        {
            /* If shape for max is one at this dimension */
            if (p_5d_max_shape[i] == CONST_ONE)
            {
                p_max_strides[i] = 0;
            }
            /* If shape for max is not one at this dimension
             * then for both inp and min shape would be one.
             */
            else
            {
                p_inp_strides[i] = 0;
                p_min_strides[i] = 0;
            }
            need_broadcast = CONST_ONE;
        }
        if (p_5d_inp_shape[i] != CONST_ONE)
        {
            inp_const &= 0;
        }
        if (p_5d_min_shape[i] != CONST_ONE)
        {
            min_const &= 0;
        }
        if (p_5d_max_shape[i] != CONST_ONE)
        {
            max_const &= 0;
        }
    }

    WORD32 itr0, itr1, itr2, itr3;
    const WORD16 *__restrict__ p_inp_base = p_inp;
    const WORD16 *__restrict__ p_min_base = p_min;
    const WORD16 *__restrict__ p_max_base = p_max;
    WORD16 *__restrict__ p_out_base = p_out;

    /* If broadcast is not needed */
    if (need_broadcast == 0)
    {
        xa_nn_elm_clamp_16_16(p_out_base,
                p_inp_base,
                p_min_base,
                p_max_base,
                p_5d_out_shape[0] * p_inp_strides[0]);
    }
    /* If any two inputs are constants */
    else if (((inp_const == CONST_ONE) && (min_const == CONST_ONE)
            && (max_const == 0))
            || ((inp_const == CONST_ONE) && (max_const == CONST_ONE)
                    && (min_const == 0)))
    {
        internal_elm_clamp_two_vecs_const(p_out_base,
                p_inp_base,
                p_min_base,
                p_max_base,
                p_5d_out_shape[0] * p_5d_out_shape[1] * p_5d_out_shape[2]
                        * p_5d_out_shape[3] * p_5d_out_shape[4],
                inp_const,
                min_const,
                max_const);

    } /* If max and min are constants */
    else if ((min_const == CONST_ONE) && (max_const == CONST_ONE)
            && (inp_const == 0))
    {
        WORD16 min = p_min[0];
        WORD16 max = p_max[0];
        xa_nn_elm_clamp_scalar_16_16(p_out_base,
                p_inp_base,
                min,
                max,
                p_5d_out_shape[0] * p_5d_out_shape[1] * p_5d_out_shape[2]
                        * p_5d_out_shape[3] * p_5d_out_shape[4]);

    }
    /* If broadcast is needed and
     * the last dimensions of all three inputs are same
     */
    else if (p_inp_strides[4] == p_min_strides[4]
            && p_min_strides[4] == p_max_strides[4])
    {
        WORD32 in_lc, out_lc;
        /* Check if 3rd dim needs to be broadcasted */
        if (p_inp_strides[3] == 0 || p_min_strides[3] == 0 || p_max_strides[3] == 0)
        {
            in_lc = p_5d_out_shape[4];
            out_lc = p_5d_out_shape[3];
            /* Repeat the 4th dimension as
             * the 3rd dimension needs to be broadcasted
             */
            for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
            {
                const WORD16 *__restrict__ p_inp_itr0 = p_inp_base;
                const WORD16 *__restrict__ p_min_itr0 = p_min_base;
                const WORD16 *__restrict__ p_max_itr0 = p_max_base;
                for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
                {
                    const WORD16 *__restrict__ p_inp_itr1 = p_inp_itr0;
                    const WORD16 *__restrict__ p_min_itr1 = p_min_itr0;
                    const WORD16 *__restrict__ p_max_itr1 = p_max_itr0;
                    for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                    {
                        internal_elm_clamp_broadcast_2D_16_16(p_out_base,
                                p_inp_itr1,
                                p_min_itr1,
                                p_max_itr1,
                                out_lc,
                                in_lc,
                                p_5d_inp_shape,
                                p_5d_min_shape,
                                p_5d_max_shape);

                        p_out_base += in_lc * out_lc;
                        p_inp_itr1 += p_inp_strides[2];
                        p_min_itr1 += p_min_strides[2];
                        p_max_itr1 += p_max_strides[2];
                    }
                    p_inp_itr0 += p_inp_strides[1];
                    p_min_itr0 += p_min_strides[1];
                    p_max_itr0 += p_max_strides[1];
                }
                p_inp_base += p_inp_strides[0];
                p_min_base += p_min_strides[0];
                p_max_base += p_max_strides[0];
            }

        } /* p_inp_strides[3] == 0 || p_max_strides[3] == 0
         || p_min_strides == 0 */
        else
        {
            /* 3rd and 4th dimensions need not be broadcasted. The lower
             * dimension broadcasting (0th, 1st, 2nd) will be taken care
             * while calculating the input addresses
             */
            in_lc = p_5d_out_shape[3] * p_5d_out_shape[4];
            for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
            {
                const WORD16 *__restrict__ p_inp_itr0 = p_inp_base;
                const WORD16 *__restrict__ p_min_itr0 = p_min_base;
                const WORD16 *__restrict__ p_max_itr0 = p_max_base;
                for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
                {
                    const WORD16 *__restrict__ p_inp_itr1 = p_inp_itr0;
                    const WORD16 *__restrict__ p_min_itr1 = p_min_itr0;
                    const WORD16 *__restrict__ p_max_itr1 = p_max_itr0;
                    for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                    {
                        xa_nn_elm_clamp_16_16(p_out_base,
                                p_inp_itr1,
                                p_min_itr1,
                                p_max_itr1,
                                in_lc);
                        p_out_base += in_lc;
                        p_inp_itr1 += p_inp_strides[2];
                        p_min_itr1 += p_min_strides[2];
                        p_max_itr1 += p_max_strides[2];
                    }
                    p_inp_itr0 += p_inp_strides[1];
                    p_min_itr0 += p_min_strides[1];
                    p_max_itr0 += p_max_strides[1];
                }
                p_inp_base += p_inp_strides[0];
                p_min_base += p_min_strides[0];
                p_max_base += p_max_strides[0];
            }
        } /* 3rd and 4th dimensions need not be broadcasted. */

    } /* p_inp_strides[4] == p_min_strides[4]
     && p_min_strides[4] == p_max_strides[4] */
    else
    {
        for (itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
        {
            const WORD16 *__restrict__ p_inp_itr0 = p_inp_base;
            const WORD16 *__restrict__ p_min_itr0 = p_min_base;
            const WORD16 *__restrict__ p_max_itr0 = p_max_base;
            for (itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
            {
                const WORD16 *__restrict__ p_inp_itr1 = p_inp_itr0;
                const WORD16 *__restrict__ p_min_itr1 = p_min_itr0;
                const WORD16 *__restrict__ p_max_itr1 = p_max_itr0;
                for (itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                {
                    const WORD16 *__restrict__ p_inp_itr2 = p_inp_itr1;
                    const WORD16 *__restrict__ p_min_itr2 = p_min_itr1;
                    const WORD16 *__restrict__ p_max_itr2 = p_max_itr1;
                    for (itr3 = 0; itr3 < p_5d_out_shape[3]; itr3++)
                    {
                        internal_elm_clamp_broadcast_1D_scalar_16_16(
                                p_out_base,
                                p_inp_itr2,
                                p_min_itr2,
                                p_max_itr2,
                                p_5d_out_shape[4],
                                p_5d_inp_shape,
                                p_5d_min_shape,
                                p_5d_max_shape);
                        p_out_base += p_5d_out_shape[4];
                        p_inp_itr2 += p_inp_strides[3];
                        p_min_itr2 += p_min_strides[3];
                        p_max_itr2 += p_max_strides[3];
                    }
                    p_inp_itr1 += p_inp_strides[2];
                    p_min_itr1 += p_min_strides[2];
                    p_max_itr1 += p_max_strides[2];
                }
                p_inp_itr0 += p_inp_strides[1];
                p_min_itr0 += p_min_strides[1];
                p_max_itr0 += p_max_strides[1];
            }
            p_inp_base += p_inp_strides[0];
            p_min_base += p_min_strides[0];
            p_max_base += p_max_strides[0];
        }
    }

    return 0;
}

