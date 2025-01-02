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
#include <string.h>

static void vecmean_inpx3(const xb_vecMxf32 *__restrict__ p_src1,
        const xb_vecMxf32 *__restrict__ p_src2,
        const xb_vecMxf32 *__restrict__ p_src3,
        xb_vecMxf32 *p_dst,
        WORD32 N)
{
    WORD32 i = 0;
    valign align_src1, align_dst;
    valign align_src2, align_src3;
    align_src1 = PDX_LA_MXF32_PP(p_src1);
    align_src2 = PDX_LA_MXF32_PP(p_src2);
    align_src3 = PDX_LA_MXF32_PP(p_src3);
    align_dst = PDX_Z_ALIGN();

    xb_vecMxf32 x1 = 0, x2 = 0, x0 = 0;
    WORD32 rem_elm = (N & (PDX_M - CONST_ONE)) * sizeof(float);
    for (i = 0; i < (N >> 2); i++)
    {
        PDX_LA_MXF32_IP(x0, align_src1, p_src1);

        PDX_LA_MXF32_IP(x1, align_src2, p_src2);
        PDX_LA_MXF32_IP(x2, align_src3, p_src3);

        x0 = x0 + PDX_ADD_MXF32(x1, x2);

        PDX_SA_MXF32_IP(x0, align_dst, p_dst);
    }

    PDX_LAV_MXF32_XP(x0, align_src1, p_src1, rem_elm);
    PDX_LAV_MXF32_XP(x1, align_src2, p_src2, rem_elm);
    PDX_LAV_MXF32_XP(x2, align_src3, p_src3, rem_elm);
    x0 = x0 + PDX_ADD_MXF32(x1, x2);
    PDX_SAV_MXF32_XP(x0, align_dst, p_dst, rem_elm);
    PDX_SAPOS_MXF32_FP(align_dst, p_dst);
}

static void vecmean_inpx2(const xb_vecMxf32 *__restrict__ p_src1,
        const xb_vecMxf32 *__restrict__ p_src2,
        xb_vecMxf32 *p_dst,
        WORD32 N)
{
    valign align_src1, align_dst;
    valign align_src2;
    align_src1 = PDX_LA_MXF32_PP(p_src1);
    align_src2 = PDX_LA_MXF32_PP(p_src2);
    align_dst = PDX_Z_ALIGN();

    WORD32 i = 0;
    xb_vecMxf32 x1 = 0, x0 = 0;
    WORD32 rem_elm = (N & (PDX_M - CONST_ONE)) * sizeof(float);
    for (i = 0; i < (N >> 2); i++)
    {
        PDX_LA_MXF32_IP(x0, align_src1, p_src1);
        PDX_LA_MXF32_IP(x1, align_src2, p_src2);

        x0 = PDX_ADD_MXF32(x0, x1);

        PDX_SA_MXF32_IP(x0, align_dst, p_dst);
    }
    PDX_LAV_MXF32_XP(x0, align_src1, p_src1, rem_elm);
    PDX_LAV_MXF32_XP(x1, align_src2, p_src2, rem_elm);
    x0 = PDX_ADD_MXF32(x0, x1);
    PDX_SAV_MXF32_XP(x0, align_dst, p_dst, rem_elm);
    PDX_SAPOS_MXF32_FP(align_dst, p_dst);
}

static inline void xa_nn_reduce_sum_5D_f32_f32(
        const FLOAT32 *__restrict__ p_inp,
        const WORD32 *const p_5D_inp_shape,
        const WORD32 *__restrict__ p_axis_data,
        WORD32 num_inp_dims,
        WORD32 num_axis_dims,
        void *p_scratch_in)
{
    xtfloat *p_in = (xtfloat*) (p_inp);
    xtfloat *p_scratch = (xtfloat*) (p_scratch_in);

    WORD32 temp_inp_n = p_5D_inp_shape[0];
    WORD32 temp_inp_h = p_5D_inp_shape[1];
    WORD32 temp_inp_w = p_5D_inp_shape[2];
    WORD32 temp_inp_d = p_5D_inp_shape[3];
    WORD32 temp_inp_c = p_5D_inp_shape[4];

    WORD32 itr_axis = 0, itr_n = 0, itr_h = 0, itr_w = 0, itr_d = 0;
    xb_vecMxf32 *p_src2, *p_src3;
    xb_vecMxf32 *p_src1;
    xb_vecMxf32 *p_dst;

    WORD32 axis_dims_count = num_axis_dims;
    if (axis_dims_count)
    {
        switch (p_axis_data[itr_axis])
        {
            case 0:
            {
                WORD32 plane_size = temp_inp_h * temp_inp_w * temp_inp_d
                        * temp_inp_c;
                for (itr_n = 0; itr_n < (temp_inp_n & ~(2 - 1)); itr_n += 2)
                {
                    p_src1 = (xb_vecMxf32*) p_scratch;
                    p_src2 = (xb_vecMxf32*) (p_in + itr_n * plane_size);
                    p_src3 = (xb_vecMxf32*) (p_in + (itr_n + 1) * plane_size);
                    p_dst = (xb_vecMxf32*) p_scratch;
                    vecmean_inpx3(p_src1, p_src2, p_src3, p_dst, plane_size);
                }
                if (temp_inp_n & 1)
                {
                    p_src1 = (xb_vecMxf32*) p_scratch;
                    p_src2 = (xb_vecMxf32*) (p_in + itr_n * plane_size);
                    p_dst = (xb_vecMxf32*) p_scratch;
                    vecmean_inpx2(p_src1, p_src2, p_dst, plane_size);
                }
                temp_inp_n = 1;
            }
                break;
            case 1:
            {
                WORD32 plane_size = temp_inp_h * temp_inp_w * temp_inp_d
                        * temp_inp_c;
                WORD32 wc_plane_size = temp_inp_w * temp_inp_d * temp_inp_c;
                for (itr_n = 0; itr_n < (temp_inp_n); itr_n++)
                {
                    p_src1 =
                            (xb_vecMxf32*) (p_scratch + (itr_n * wc_plane_size));
                    for (itr_h = 0; itr_h < (temp_inp_h & ~(2 - 1)); itr_h += 2)
                    {
                        p_src2 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                + (itr_h * wc_plane_size));
                        p_src3 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                + ((itr_h + 1) * wc_plane_size));
                        p_dst = (xb_vecMxf32*) (p_scratch
                                + (itr_n * wc_plane_size));
                        vecmean_inpx3(p_src1, p_src2, p_src3, p_dst,
                                wc_plane_size);
                        p_src1 = (xb_vecMxf32*) (p_scratch
                                + (itr_n * wc_plane_size));
                    }

                    if (temp_inp_h & 1)
                    {
                        p_src2 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                + (itr_h * wc_plane_size));
                        p_dst = (xb_vecMxf32*) (p_scratch
                                + (itr_n * wc_plane_size));
                        vecmean_inpx2(p_src1, p_src2, p_dst, wc_plane_size);
                    }
                }
                temp_inp_h = 1;
            }
                break;
            case 2:
            {
                WORD32 plane_size = temp_inp_h * temp_inp_w * temp_inp_d
                        * temp_inp_c;
                WORD32 wc_plane_size = temp_inp_w * temp_inp_d * temp_inp_c;
                WORD32 hc_plane_size = temp_inp_h * temp_inp_d * temp_inp_c;
                WORD32 dc_plane_size = temp_inp_d * temp_inp_c;
                for (itr_n = 0; itr_n < (temp_inp_n); itr_n++)
                {
                    for (itr_h = 0; itr_h < (temp_inp_h); itr_h++)
                    {
                        p_src1 = (xb_vecMxf32*) (p_scratch
                                + (((itr_n * hc_plane_size)
                                        + itr_h * dc_plane_size)));
                        for (itr_w = 0; itr_w < (temp_inp_w & ~(2 - 1));
                                itr_w += 2)
                        {
                            p_src2 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                    + (itr_h * wc_plane_size)
                                    + (itr_w * dc_plane_size));
                            p_src3 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                    + (itr_h * wc_plane_size)
                                    + ((itr_w + 1) * dc_plane_size));
                            p_dst = (xb_vecMxf32*) (p_scratch
                                    + (itr_n * hc_plane_size)
                                    + itr_h * dc_plane_size);
                            vecmean_inpx3(p_src1, p_src2, p_src3, p_dst,
                                    dc_plane_size);
                            p_src1 = (xb_vecMxf32*) (p_scratch
                                    + (itr_n * hc_plane_size)
                                    + (itr_h * dc_plane_size));
                        }

                        if (temp_inp_w & 1)
                        {
                            p_src2 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                    + (itr_h * wc_plane_size)
                                    + (itr_w * dc_plane_size));
                            p_dst = (xb_vecMxf32*) (p_scratch
                                    + (itr_n * hc_plane_size)
                                    + itr_h * dc_plane_size);
                            vecmean_inpx2(p_src1, p_src2, p_dst, dc_plane_size);
                        }
                    }
                }
                temp_inp_w = 1;
            }
                break;
            case 3:
            {
                // Calculate the sizes of different planes and dimensions
                WORD32 plane_size = temp_inp_h * temp_inp_w * temp_inp_d
                        * temp_inp_c;
                WORD32 wc_plane_size = temp_inp_w * temp_inp_d * temp_inp_c;
                WORD32 dc_plane_size = temp_inp_d * temp_inp_c;
                WORD32 hw_plane_size = temp_inp_h * temp_inp_w * temp_inp_c;

                for (itr_n = 0; itr_n < temp_inp_n; itr_n++)
                {
                    for (itr_h = 0; itr_h < temp_inp_h; itr_h++)
                    {
                        for (itr_w = 0; itr_w < temp_inp_w; itr_w++)
                        {
                            p_src1 = (xb_vecMxf32*) (p_scratch
                                    + ((itr_n * hw_plane_size)
                                            + (itr_h * temp_inp_w * temp_inp_c)
                                            + (itr_w * temp_inp_c)));
                            for (itr_d = 0; itr_d < (temp_inp_d & ~(2 - 1));
                                    itr_d += 2)
                            {
                                p_src2 = (xb_vecMxf32*) (p_in
                                        + (itr_n * plane_size)
                                        + (itr_h * wc_plane_size)
                                        + (itr_w * dc_plane_size)
                                        + (itr_d * temp_inp_c));
                                p_src3 = (xb_vecMxf32*) (p_in
                                        + (itr_n * plane_size)
                                        + (itr_h * wc_plane_size)
                                        + (itr_w * dc_plane_size)
                                        + ((itr_d + 1) * temp_inp_c));
                                p_dst = (xb_vecMxf32*) (p_scratch
                                        + ((itr_n * hw_plane_size)
                                                + (itr_h * temp_inp_w
                                                        * temp_inp_c)
                                                + (itr_w * temp_inp_c)));
                                vecmean_inpx3(p_src1, p_src2, p_src3, p_dst,
                                        temp_inp_c);
                                p_src1 = (xb_vecMxf32*) (p_scratch
                                        + ((itr_n * hw_plane_size)
                                                + (itr_h * temp_inp_w
                                                        * temp_inp_c)
                                                + (itr_w * temp_inp_c)));
                            }
                            if (temp_inp_d & 1)
                            {
                                p_src2 = (xb_vecMxf32*) (p_in
                                        + (itr_n * plane_size)
                                        + (itr_h * wc_plane_size)
                                        + (itr_w * dc_plane_size)
                                        + (itr_d * temp_inp_c));
                                p_dst = (xb_vecMxf32*) (p_scratch
                                        + ((itr_n * hw_plane_size)
                                                + (itr_h * temp_inp_w
                                                        * temp_inp_c)
                                                + (itr_w * temp_inp_c)));
                                vecmean_inpx2(p_src1, p_src2, p_dst,
                                        temp_inp_c);
                            }
                        }
                    }
                }

                temp_inp_d = 1;
            }
                break;
            case 4:
            {
                WORD32 axis_count = temp_inp_c;
                WORD32 plane_size = temp_inp_n * temp_inp_h * temp_inp_w
                        * temp_inp_d;
                WORD32 length_per_step = axis_count;
                xtfloat out = 0;
                xb_vecMxf32 x0 = 0, x1 = 0;
                WORD32 lead_itr = 0, axis_itr = 0;
                valign ax1;
                /* number of remaining elements to be processed */
                WORD32 rem_elem = ((axis_count & (PDX_M - 1)) * sizeof(float));

                xtfloat *__restrict__ p_out_f32 = (xtfloat*) p_scratch;

                for (lead_itr = 0; lead_itr < plane_size; lead_itr++)
                {
                    const FLOAT32 *__restrict__ p_in2 = p_in;

                    const xb_vecMxf32 *__restrict__ p_in1_mxf32 =
                            (const xb_vecMxf32*) (p_in2);
                    ax1 = PDX_LA_MXF32_PP(p_in1_mxf32);
                    x0 = 0;
                    for (axis_itr = 0; axis_itr < axis_count >> 2; axis_itr++)
                    {
                        PDX_LA_MXF32_IP(x1, ax1, p_in1_mxf32);
                        x0 = PDX_ADD_MXF32(x0, x1);
                    }
                    x1 = 0;
                    PDX_LAV_MXF32_XP(x1, ax1, p_in1_mxf32, rem_elem);
                    x0 = PDX_ADD_MXF32(x0, x1);
                    /* Store output */
                    out = PDX_RADD_MXF32(x0);
                    xtfloat_storeip(out, p_out_f32, 4);

                    /* input pointer update */
                    p_in = p_in + length_per_step;
                }
                temp_inp_c = 1;
            }
                break;
            default:
                break;
        }

        axis_dims_count--;
        itr_axis++;
    }

    while (axis_dims_count)
    {
        p_in = p_scratch;
        switch (p_axis_data[itr_axis])
        {
            case 0:
            {
                WORD32 plane_size = temp_inp_h * temp_inp_w * temp_inp_d
                        * temp_inp_c;
                for (itr_n = 1; itr_n < ((temp_inp_n - 1) & ~(2 - 1)); itr_n +=
                        2)
                {
                    p_src1 = (xb_vecMxf32*) p_scratch;
                    p_src2 = (xb_vecMxf32*) (p_in + itr_n * plane_size);
                    p_src3 = (xb_vecMxf32*) (p_in + (itr_n + 1) * plane_size);
                    p_dst = (xb_vecMxf32*) p_scratch;
                    vecmean_inpx3(p_src1, p_src2, p_src3, p_dst, plane_size);
                }
                if ((temp_inp_n - 1) & 1)
                {
                    p_src1 = (xb_vecMxf32*) p_scratch;
                    p_src2 = (xb_vecMxf32*) (p_in + itr_n * plane_size);
                    p_dst = (xb_vecMxf32*) p_scratch;
                    vecmean_inpx2(p_src1, p_src2, p_dst, plane_size);
                }
                temp_inp_n = 1;
            }
                break;
            case 1:
            {
                WORD32 plane_size = temp_inp_h * temp_inp_w * temp_inp_d
                        * temp_inp_c;
                WORD32 wc_plane_size = temp_inp_w * temp_inp_d * temp_inp_c;
                for (itr_n = 0; itr_n < (temp_inp_n); itr_n++)
                {
                    p_src1 = (xb_vecMxf32*) (p_scratch + (itr_n * plane_size));
                    for (itr_h = 1; itr_h < ((temp_inp_h - 1) & ~(2 - 1));
                            itr_h += 2)
                    {
                        p_src2 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                + (itr_h * wc_plane_size));
                        p_src3 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                + ((itr_h + 1) * wc_plane_size));
                        p_dst = (xb_vecMxf32*) (p_scratch
                                + (itr_n * wc_plane_size));
                        vecmean_inpx3(p_src1, p_src2, p_src3, p_dst,
                                wc_plane_size);
                        p_src1 = (xb_vecMxf32*) (p_scratch
                                + (itr_n * wc_plane_size));
                    }

                    if ((temp_inp_h - 1) & 1)
                    {
                        p_src2 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                + (itr_h * wc_plane_size));
                        p_dst = (xb_vecMxf32*) (p_scratch
                                + (itr_n * wc_plane_size));
                        vecmean_inpx2(p_src1, p_src2, p_dst, wc_plane_size);
                    }
                }
                temp_inp_h = 1;
            }
                break;
            case 2:
            {
                WORD32 plane_size = temp_inp_h * temp_inp_w * temp_inp_d
                        * temp_inp_c;
                WORD32 wc_plane_size = temp_inp_w * temp_inp_d * temp_inp_c;
                WORD32 hc_plane_size = temp_inp_h * temp_inp_d * temp_inp_c;
                WORD32 dc_plane_size = temp_inp_d * temp_inp_c;
                for (itr_n = 0; itr_n < (temp_inp_n); itr_n++)
                {
                    for (itr_h = 0; itr_h < (temp_inp_h); itr_h++)
                    {
                        p_src1 =
                                (xb_vecMxf32*) (p_scratch
                                        + (((itr_n * plane_size)
                                                + itr_h * wc_plane_size)));
                        for (itr_w = 1; itr_w < ((temp_inp_w - 1) & ~(2 - 1));
                                itr_w += 2)
                        {
                            p_src2 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                    + (itr_h * wc_plane_size)
                                    + (itr_w * dc_plane_size));
                            p_src3 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                    + (itr_h * wc_plane_size)
                                    + ((itr_w + 1) * dc_plane_size));
                            p_dst = (xb_vecMxf32*) (p_scratch
                                    + (itr_n * hc_plane_size)
                                    + itr_h * dc_plane_size);
                            vecmean_inpx3(p_src1, p_src2, p_src3, p_dst,
                                    dc_plane_size);
                            p_src1 = (xb_vecMxf32*) (p_scratch
                                    + (itr_n * hc_plane_size)
                                    + (itr_h * dc_plane_size));
                        }

                        if ((temp_inp_w - 1) & 1)
                        {
                            p_src2 = (xb_vecMxf32*) (p_in + (itr_n * plane_size)
                                    + (itr_h * wc_plane_size)
                                    + (itr_w * dc_plane_size));
                            p_dst = (xb_vecMxf32*) (p_scratch
                                    + (itr_n * hc_plane_size)
                                    + itr_h * dc_plane_size);
                            vecmean_inpx2(p_src1, p_src2, p_dst, dc_plane_size);
                        }
                    }
                }
                temp_inp_w = 1;
            }
                break;
            case 3:
            {
                // Calculate the sizes of different planes and dimensions
                WORD32 plane_size = temp_inp_h * temp_inp_w * temp_inp_d
                        * temp_inp_c;
                WORD32 wc_plane_size = temp_inp_w * temp_inp_d * temp_inp_c;
                WORD32 dc_plane_size = temp_inp_d * temp_inp_c;
                WORD32 hw_plane_size = temp_inp_h * temp_inp_w * temp_inp_c;

                for (itr_n = 0; itr_n < temp_inp_n; itr_n++)
                {
                    for (itr_h = 0; itr_h < temp_inp_h; itr_h++)
                    {
                        for (itr_w = 0; itr_w < temp_inp_w; itr_w++)
                        {
                            p_src1 = (xb_vecMxf32*) (p_scratch
                                    + ((itr_n * plane_size)
                                            + (itr_h * wc_plane_size)
                                            + (itr_w * dc_plane_size)));
                            for (itr_d = 1;
                                    itr_d < ((temp_inp_d - 1) & ~(2 - 1));
                                    itr_d += 2)
                            {
                                p_src2 = (xb_vecMxf32*) (p_in
                                        + (itr_n * plane_size)
                                        + (itr_h * wc_plane_size)
                                        + (itr_w * dc_plane_size)
                                        + (itr_d * temp_inp_c));
                                p_src3 = (xb_vecMxf32*) (p_in
                                        + (itr_n * plane_size)
                                        + (itr_h * wc_plane_size)
                                        + (itr_w * dc_plane_size)
                                        + ((itr_d + 1) * temp_inp_c));
                                p_dst = (xb_vecMxf32*) (p_scratch
                                        + ((itr_n * hw_plane_size)
                                                + (itr_h * temp_inp_w
                                                        * temp_inp_c)
                                                + (itr_w * temp_inp_c)));
                                vecmean_inpx3(p_src1, p_src2, p_src3, p_dst,
                                        temp_inp_c);
                                p_src1 = (xb_vecMxf32*) (p_scratch
                                        + ((itr_n * hw_plane_size)
                                                + (itr_h * temp_inp_w
                                                        * temp_inp_c)
                                                + (itr_w * temp_inp_c)));
                            }
                            if ((temp_inp_d - 1) & 1)
                            {
                                p_src2 = (xb_vecMxf32*) (p_in
                                        + (itr_n * plane_size)
                                        + (itr_h * wc_plane_size)
                                        + (itr_w * dc_plane_size)
                                        + (itr_d * temp_inp_c));
                                p_dst = (xb_vecMxf32*) (p_scratch
                                        + ((itr_n * hw_plane_size)
                                                + (itr_h * temp_inp_w
                                                        * temp_inp_c)
                                                + (itr_w * temp_inp_c)));
                                vecmean_inpx2(p_src1, p_src2, p_dst,
                                        temp_inp_c);
                            }
                        }
                    }
                }

                temp_inp_d = 1;
            }
                break;
            case 4:
            {
                WORD32 axis_count = temp_inp_c;
                WORD32 plane_size = temp_inp_n * temp_inp_h * temp_inp_w
                        * temp_inp_d;
                WORD32 length_per_step = axis_count;
                xtfloat out = 0;
                xb_vecMxf32 x0 = 0, x1 = 0;
                WORD32 lead_itr = 0, axis_itr = 0;
                valign ax1;
                /* number of remaining elements to be processed */
                WORD32 rem_elem = ((axis_count & (PDX_M - 1)) * sizeof(float));

                xtfloat *__restrict__ p_out_f32 = (xtfloat*) p_scratch;

                for (lead_itr = 0; lead_itr < plane_size; lead_itr++)
                {
                    const FLOAT32 *__restrict__ p_in2 = p_in;

                    const xb_vecMxf32 *__restrict__ p_in1_mxf32 =
                            (const xb_vecMxf32*) (p_in2);
                    ax1 = PDX_LA_MXF32_PP(p_in1_mxf32);
                    x0 = 0;
                    for (axis_itr = 0; axis_itr < axis_count >> 2; axis_itr++)
                    {
                        PDX_LA_MXF32_IP(x1, ax1, p_in1_mxf32);
                        x0 = PDX_ADD_MXF32(x0, x1);
                    }
                    x1 = 0;
                    PDX_LAV_MXF32_XP(x1, ax1, p_in1_mxf32, rem_elem);
                    x0 = PDX_ADD_MXF32(x0, x1);
                    /* Store output */
                    out = PDX_RADD_MXF32(x0);
                    xtfloat_storeip(out, p_out_f32, 4);

                    /* input pointer update */
                    p_in = p_in + length_per_step;
                }
                temp_inp_c = 1;
            }
                break;
            default:
                break;
        }

        axis_dims_count--;
        itr_axis++;
    }

}

WORD32 xa_nn_mean_f32_f32(FLOAT32 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        WORD32 num_out_dims,
        const FLOAT32 *__restrict__ p_inp,
        const WORD32 *const p_inp_shape,
        WORD32 num_inp_dims,
        const WORD32 *__restrict__ p_axis,
        WORD32 num_axis_dims,
        void *__restrict__ p_scratch_in)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_scratch_in, UNSUPPORTED_PARAM);

    /* Invalid input checks */
    XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND(((num_out_dims <= 0) || (num_out_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND(((num_axis_dims < 0) || (num_axis_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);

    WORD32 axis_itr = 0, inp_itr = 0, out_itr = 0;
    FLOAT32 num_elm_in_axis = 1.0;

    if (p_axis != NULL)
    {
        WORD32 current; 
        WORD32 dim_exist[MAX_DIMS];
        memset(dim_exist, 0, sizeof(dim_exist));
        for (axis_itr = 0; axis_itr < num_axis_dims; axis_itr++)
        {
            current = p_axis[axis_itr];
            XA_NNLIB_ARG_CHK_COND(
                    ((current < 0) || (current > (num_inp_dims - 1))),
                    UNSUPPORTED_PARAM);

            /* Avoid calculation in case of repeated axis dims*/
            if (dim_exist[current] == 0)
            {
                num_elm_in_axis *= (FLOAT32) p_inp_shape[current];
                dim_exist[current] = 1;
            }
        }
    }

    for (inp_itr = 0; inp_itr < num_inp_dims; inp_itr++)
    {
        XA_NNLIB_ARG_CHK_COND((p_inp_shape[inp_itr] <= 0), UNSUPPORTED_PARAM);
    }

    WORD32 out_length = 1;
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

    if ((p_axis == NULL ) || (num_axis_dims == 0))
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
        WORD32 rem_elem = ((num_elm & (PDX_M - CONST_ONE)) * sizeof(float));
        WORD32 Itr = 0;
        for (Itr = 0; Itr < num_elm >> CONST_TWO; Itr++)
        {
            PDX_LA_MXF32_IP(x1, ax, p_x);
            x0 = PDX_ADD_MXF32(x0, x1);
        }
        x1 = 0;
        PDX_LAV_MXF32_XP(x1, ax, p_x, rem_elem);

        x0 = PDX_ADD_MXF32(x0, x1);
        /* Store output */
        out = PDX_RADD_MXF32(x0);

        *p_out = out * one_over_num_elm;

        return 0;
    }

    FLOAT32 *p_in = (FLOAT32*) (p_inp);
    WORD32 *p_scratch = (WORD32*) (p_scratch_in);
    // Changing order of axis data so that reduce max will be first computed
    // across largest inp shape dim in axis. This is required to
    // minimize the scratch usage.
    WORD32 inp_length = 1, p_axis_data[MAX_DIMS] = {0}, inp_shape_max;
    if (num_axis_dims)
    {
        inp_shape_max = p_inp_shape[p_axis[0]];
        axis_itr = 1;
        WORD32 max_axis_itr = 0;
        WORD32 temp_p_axis_0 = p_axis[0];
        for (axis_itr = 0; axis_itr < num_axis_dims; axis_itr++)
        {
            p_axis_data[axis_itr] = p_axis[axis_itr];
        }
        for (axis_itr = 1; axis_itr < num_axis_dims; axis_itr++)
        {
            if (p_inp_shape[p_axis[axis_itr]] > inp_shape_max)
            {
                inp_shape_max = p_inp_shape[p_axis[axis_itr]];
                max_axis_itr = axis_itr;
            }
        }
        p_axis_data[0] = p_axis_data[max_axis_itr];
        p_axis_data[max_axis_itr] = temp_p_axis_0;

        inp_itr = 0;
        for (inp_itr = 0; inp_itr < num_inp_dims; inp_itr++)
        {
            inp_length *= p_inp_shape[inp_itr];
        }

        memset(p_scratch, 0, ((inp_length / inp_shape_max) * sizeof(WORD32)));
    }

    // Promoting lesser dim tensors to 4D tensors. Also modifying axis
    // data accordingly.
    WORD32 p_5D_inp_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE, CONST_ONE,
            CONST_ONE};
    WORD32 itr = num_inp_dims - CONST_ONE;
    WORD32 count = CONST_FOUR;
    while (itr >= 0)
    {
        p_5D_inp_shape[count] = p_inp_shape[itr];
        itr--;
        count--;
    }
    for (itr = 0; itr < num_axis_dims; itr++)
    {
        p_axis_data[itr] = p_axis_data[itr] + (MAX_DIMS - num_inp_dims);
    }

    xb_vecMxf32 *restrict p_z = (xb_vecMxf32*) p_out;
    valign az = PDX_Z_ALIGN();

    if (num_axis_dims)
    {
        if (num_elm_in_axis > CONST_ONE)
        {
            xa_nn_reduce_sum_5D_f32_f32(p_in, p_5D_inp_shape, p_axis_data,
                    num_inp_dims, num_axis_dims, p_scratch);
            itr = 0;
            xb_vecMxf32 *__restrict__ p_x = (xb_vecMxf32*) (p_scratch);
            valign ax = PDX_LA_MXF32_PP(p_x);

            WORD32 rem_elm = (out_length & (PDX_M - CONST_ONE)) * sizeof(float);
            xb_vecMxf32 multiplier = PDX_DIV_MXF32(CONST_ONE, num_elm_in_axis);
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
        else
        {
            memcpy(p_out, p_inp, inp_length * sizeof(FLOAT32));
        }
    }

    return 0;
}
