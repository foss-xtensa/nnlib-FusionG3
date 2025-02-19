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

WORD32 init_prep(const WORD32 *const p_out_shape,
        const WORD32 *const p_inp_shape,
        const WORD32 *__restrict__ p_permute_vec,
        WORD32 num_inp_dims,
        WORD32 *p_5D_inp_shape,
        WORD32 *p_5D_out_shape,
        WORD32 *p_5D_permute_vec)
{
    /*
     * Shift all dimensions with size 1 in the outer part to optimize memory access
     * This ensures that larger dimensions are processed first.
     * Example:
     * Input:
     *   p_out_shape      = {1,2,3,1,4};
     *   p_permute_vec    = {0,1,2,3,4};
     * After processing:
     *   eff_output_shape = {1,1,2,3,4};
     *   eff_permute_vec  = {0,3,1,2,4};
     */

    WORD32 itr, i;
    /* To store the intermediate output shape and intermediate permutation vector */
    WORD32 eff_output_shape[MAX_DIMS];
    WORD32 eff_permute_vec[MAX_DIMS];

    /* Copy the original shapes and permutation vector into the effective variables */
    for (i = 0; i < num_inp_dims; i++)
    {
        eff_output_shape[i] = p_out_shape[i];
        eff_permute_vec[i] = p_permute_vec[i];
    }

    /* Variables to track positions of dimensions with size 1 and non-1 dimensions */
    WORD32 one_i = num_inp_dims - CONST_ONE, non_one_i = num_inp_dims - CONST_ONE;

    /* Shift dimensions of size 1 to the outer part of the shape */
    while (one_i > 0 && non_one_i >= 0)
    {
        /* Find the first dimension with size 1 */
        while (one_i > 0 && eff_output_shape[one_i] != CONST_ONE)
        {
            one_i--;
        }

        non_one_i = one_i;

        /* Find the next dimension with a non-1 size */
        while (non_one_i >= 0 && eff_output_shape[non_one_i] == CONST_ONE)
        {
            non_one_i--;
        }

        /* If both indices are valid, swap the dimensions */
        if (one_i > 0 && non_one_i >= 0)
        {
            WORD32 temp;

            /* Swap the output shape dimensions */
            temp = eff_output_shape[one_i];
            eff_output_shape[one_i] = eff_output_shape[non_one_i];
            eff_output_shape[non_one_i] = temp;

            /* Swap the permutation vector elements */
            temp = eff_permute_vec[one_i];
            eff_permute_vec[one_i] = eff_permute_vec[non_one_i];
            eff_permute_vec[non_one_i] = temp;
        }
    }


    /* Count how many dimensions are the same between input and output */
    itr = num_inp_dims - CONST_ONE; 
    WORD32 merge_dims_count = 0;
    WORD32 merge_end_pos = -1;

    while (itr>0)
    {
        while ((itr >= 1) &&
               (itr == eff_permute_vec[itr]) &&
               ((itr - CONST_ONE) == eff_permute_vec[itr - CONST_ONE]))
        {
            if (!merge_dims_count)
            {
                merge_end_pos = itr;
                merge_dims_count++;
            }
            merge_dims_count++;
            itr--;
        }
        itr--;
    }


    /*
     * Now, we promote smaller tensors to 5D tensors by adding leading dimensions (if needed)
     * we determined the merge_dims_count and merge_end_pos.
     * we update the p_5D_inp_shape, p_5D_out_shape accordingly.
     *
     * At this point:
     *   p_5D_inp_shape[5]   = {1, 1, 1, 1, 1};
     *   p_5D_out_shape[5]   = {1, 1, 1, 1, 1};
     *   p_5D_permute_vec[5] = {0, 1, 2, 3, 4};
     */
    itr = num_inp_dims - CONST_ONE;
    WORD32 count_5d   = MAX_DIMS - CONST_ONE;
    WORD32 merge_start_pos = (merge_dims_count == 0) ? merge_end_pos :
                                (merge_end_pos - merge_dims_count + CONST_ONE);

    while (itr >= 0)
    {
        p_5D_inp_shape[count_5d] *= p_inp_shape[itr];
        p_5D_out_shape[count_5d] *= eff_output_shape[itr];

        if (!((itr >  merge_start_pos) &&
              (itr <= merge_end_pos)))
        {
            count_5d--;
        }
        itr--;
    }


    /* Update the permutation vector for 5D */
    WORD32 dims_added = MAX_DIMS - num_inp_dims;
    itr = num_inp_dims - CONST_ONE;
    count_5d   = MAX_DIMS - CONST_ONE;
    WORD32 offset;

    /* Set the correct permutation indices for the 5D tensor */
    while (itr >= 0)
    {
        offset=0;

        if (eff_permute_vec[itr] < merge_end_pos)
        {
            offset = merge_dims_count - CONST_ONE;
        }

        p_5D_permute_vec[count_5d] = eff_permute_vec[itr] + offset + dims_added;

        if ((itr >= merge_start_pos) &&
            (itr <= merge_end_pos))
        {
            itr = itr - merge_dims_count + CONST_ONE;
        }

        itr--;
        count_5d--;
    }


    /* finding last_n_same_dim */
    WORD32 last_n_same_dim = 0;
    itr = num_inp_dims - CONST_ONE;

    if (eff_permute_vec[itr] == itr)
    {
        if (!merge_dims_count)
        {
            last_n_same_dim = 1;
        }
        else
        {
            last_n_same_dim = merge_dims_count;
        }
    }

    /* Return the number of trailing dimensions that are the same (last_n_same_dim) */
    return last_n_same_dim;
}

WORD32 last_n_same_dim_case(WORD8 *__restrict__ p_out,
        WORD32 *p_5D_out_shape,
        const WORD8 *__restrict__ p_inp,
        WORD32 *p_5D_inp_shape,
        WORD32 *p_5D_permute_vec,
        WORD32 elm_size)
{
    /* Extracting the dimensions of the output tensor */
    WORD32 out_dim0, out_dim1, out_dim2, out_dim3, out_dim4;
    out_dim0 = p_5D_out_shape[0];
    out_dim1 = p_5D_out_shape[CONST_ONE];
    out_dim2 = p_5D_out_shape[CONST_TWO];
    out_dim3 = p_5D_out_shape[CONST_THREE];
    out_dim4 = p_5D_out_shape[CONST_FOUR];

    /* Extracting the dimensions of the input tensor */
    WORD32 inp_dim1, inp_dim2, inp_dim3, inp_dim4;
    inp_dim1 = p_5D_inp_shape[CONST_ONE];
    inp_dim2 = p_5D_inp_shape[CONST_TWO];
    inp_dim3 = p_5D_inp_shape[CONST_THREE];
    inp_dim4 = p_5D_inp_shape[CONST_FOUR];

    /* Calculating the strides for each dimension of the input tensor */
    WORD32 inp_stride[MAX_DIMS];
    inp_stride[0] = inp_dim1 * inp_dim2 * inp_dim3 * inp_dim4;
    inp_stride[CONST_ONE] = inp_dim2 * inp_dim3 * inp_dim4;
    inp_stride[CONST_TWO] = inp_dim3 * inp_dim4;
    inp_stride[CONST_THREE] = inp_dim4;
    inp_stride[CONST_FOUR] = CONST_ONE;

    /* Calculating the number of SIMD operations per loop and scalar operations for the remainder */
    WORD32 num_simd16_ops = (out_dim4 * elm_size) >> LOG2_PDX_4M;
    WORD32 num_scalar16_ops = (out_dim4 * elm_size) & (PDX_4M - CONST_ONE);

    xb_vec4Mx8 *__restrict__ p_o = (xb_vec4Mx8*) (p_out);
    valign a_out = PDX_Z_ALIGN();

    WORD32 itr0, itr1, itr2, itr3, itr4;

    WORD8 *__restrict__ p_inp0 = (WORD8*) p_inp;

    for (itr0 = 0; itr0 < out_dim0; itr0++)
    {
        WORD8 *__restrict__ p_inp1 = p_inp0
                + (itr0 * inp_stride[p_5D_permute_vec[0]]) * elm_size;

#pragma loop_count min=1
        for (itr1 = 0; itr1 < out_dim1; itr1++)
        {
            WORD8 *__restrict__ p_inp2 = p_inp1
                    + (itr1 * inp_stride[p_5D_permute_vec[1]]) * elm_size;

#pragma loop_count min=1
            for (itr2 = 0; itr2 < out_dim2; itr2++)
            {
                WORD8 *__restrict__ p_inp3 = p_inp2
                        + (itr2 * inp_stride[p_5D_permute_vec[2]]) * elm_size;

#pragma loop_count min=1
                for (itr3 = 0; itr3 < out_dim3; itr3++)
                {
                    WORD8 *__restrict__ p_inp4 = p_inp3
                            + (itr3 * inp_stride[p_5D_permute_vec[3]])
                                    * elm_size;

                    xb_vec4Mx8 *__restrict__ p_i = (xb_vec4Mx8*) (p_inp4);

                    valign a_inp = PDX_LA_4MX8_PP(p_i);

                    xb_vec4Mx8 d0;

                    /* Process the SIMD operations for full-sized chunks (16 elements per operation) */
                    for (itr4 = 0; itr4 < num_simd16_ops; itr4++)
                    {
                        PDX_LA_4MX8_IP(d0, a_inp, p_i);
                        PDX_SA_4MX8_IP(d0, a_out, p_o);
                    }

                    /* Process the remaining scalar operations for the remainder of the elements */
                    PDX_LAV_4MX8_XP(d0, a_inp, p_i, num_scalar16_ops);
                    PDX_SAV_4MX8_XP(d0, a_out, p_o, num_scalar16_ops);

                    PDX_SAPOS_4MX8_FP(a_out, p_o);
                }
            }
        }
    }
    return 0;
}

WORD32 last_dim_not_same_case(WORD8 *__restrict__ p_out,
        WORD32 *p_5D_out_shape,
        const WORD8 *__restrict__ p_inp,
        WORD32 *p_5D_inp_shape,
        WORD32 *p_5D_permute_vec,
        WORD32 elm_size)
{
    WORD32 out_dim0, out_dim1, out_dim2, out_dim3, out_dim4;
    WORD32 inp_dim1, inp_dim2, inp_dim3, inp_dim4;
    WORD32 inp_stride[MAX_DIMS];

    /* Extract output tensor dimensions */
    out_dim0 = p_5D_out_shape[0];
    out_dim1 = p_5D_out_shape[CONST_ONE];
    out_dim2 = p_5D_out_shape[CONST_TWO];
    out_dim3 = p_5D_out_shape[CONST_THREE];
    out_dim4 = p_5D_out_shape[CONST_FOUR];

    /* Extract input tensor dimensions */
    inp_dim1 = p_5D_inp_shape[CONST_ONE];
    inp_dim2 = p_5D_inp_shape[CONST_TWO];
    inp_dim3 = p_5D_inp_shape[CONST_THREE];
    inp_dim4 = p_5D_inp_shape[CONST_FOUR];

    /* Stride calculation for each dimension of the input tensor.*/
    inp_stride[0] = inp_dim1 * inp_dim2 * inp_dim3 * inp_dim4;
    inp_stride[CONST_ONE] = inp_dim2 * inp_dim3 * inp_dim4;
    inp_stride[CONST_TWO] = inp_dim3 * inp_dim4;
    inp_stride[CONST_THREE] = inp_dim4;
    inp_stride[CONST_FOUR] = CONST_ONE;

    /* If elm_size is 4, input data is of type 32-bit integer */
    if (elm_size == CONST_FOUR)
    {
        /* Selection patterns for vector operations */
        xb_vecMx32 sel_pat_1 = { 0, 4, 1, 2 };
        xb_vecMx32 sel_pat_2 = { 1, 2, 0, 4 };

        xb_vecMx32 d0, d1, d2, d3;
        xb_vecMx32 data10, data32, data;
        xb_int32 da, db;

        WORD32 itr0, itr1, itr2, itr3, itr4;
        WORD32 *__restrict__ p_inp0 = (WORD32*) p_inp;
        WORD32 *__restrict__ p_out0 = (WORD32*) p_out;
        valign a_out = PDX_Z_ALIGN();

        /* Calculate the byte stride for input data */
        WORD32 stride_val = inp_stride[p_5D_permute_vec[4]];
        WORD32 stride_val_bytes = stride_val * elm_size;

        for (itr0 = 0; itr0 < out_dim0; itr0++)
        {
            WORD32 *__restrict__ p_inp1 = p_inp0
                    + (itr0 * inp_stride[p_5D_permute_vec[0]]);
            for (itr1 = 0; itr1 < out_dim1; itr1++)
            {
                WORD32 *__restrict__ p_inp2 = p_inp1
                        + (itr1 * inp_stride[p_5D_permute_vec[1]]);
                for (itr2 = 0; itr2 < out_dim2; itr2++)
                {
                    WORD32 *__restrict__ p_inp3 = p_inp2
                            + (itr2 * inp_stride[p_5D_permute_vec[2]]);
                    for (itr3 = 0; itr3 < out_dim3; itr3++)
                    {
                        WORD32 *__restrict__ p_inp4 = p_inp3
                                + (itr3 * inp_stride[p_5D_permute_vec[3]]);


                        for (itr4 = 0; itr4 < (out_dim4 >> LOG2_PDX_M); itr4++)
                        {
                            d1 = PDX_LSR_32_X(p_inp4, stride_val_bytes);

                            db = PDX_LS_32_X(p_inp4, 2 * stride_val_bytes);
                            d2 = PDX_MOV_MX32_FROM_32(db);

                            d3 = PDX_LSR_32_X(p_inp4, 3 * stride_val_bytes);

                            PDX_LS_32_XP(da, p_inp4, 4 * stride_val_bytes);
                            d0 = PDX_MOV_MX32_FROM_32(da);

                            /* Apply selection patterns and combine the vectors */
                            data10 = PDX_SEL_MX32(d1, d0, sel_pat_1);
                            data32 = PDX_SEL_MX32(d3, d2, sel_pat_2);
                            data = PDX_OR_MX32(data10, data32);

                            PDX_SA_MX32_IP(data, a_out, (xb_vecMx32*) p_out0);
                        }
                        PDX_SAPOS_MX32_FP(a_out, (xb_vecMx32*) p_out0);
#pragma loop_count max=3
                        /* Handle any remaining elements if the output dimension is not a multiple of the vector size */
                        for (itr4 = 0; itr4 < (out_dim4 & (PDX_M - CONST_ONE)); itr4++)
                        {
                            xb_vecMx32 d0;
                            PDX_LSR_32_XP(d0, p_inp4, stride_val_bytes);
                            PDX_SS_32_XP(d0, p_out0, elm_size);
                        }
                    }
                }
            }
        }
    }
    /* If elm_size == 2, then input data type is 16-bit */
    else if (elm_size == CONST_TWO)
    {
        /* Selection patterns for vector operations */
        xb_vec2Mx16 sel_pat_1 = { 0, 8, 1, 2, 3, 4, 5, 6 };
        xb_vec2Mx16 sel_pat_2 = { 1, 2, 0, 8, 3, 4, 5, 6 };
        xb_vec2Mx16 sel_pat_3 = { 1, 2, 3, 4, 0, 8, 5, 6 };
        xb_vec2Mx16 sel_pat_4 = { 1, 2, 3, 4, 5, 6, 0, 8 };

        xb_vec2Mx16 d0, d1, d2, d3, d4, d5, d6, d7;
        xb_vec2Mx16 data10, data32, data54, data76;
        xb_vec2Mx16 data3210, data7654;
        xb_vec2Mx16 data;
        xb_int16 da, db, dc, dd;

        WORD32 itr0, itr1, itr2, itr3, itr4;
        WORD16 *__restrict__ p_inp0 = (WORD16*) p_inp;
        WORD16 *__restrict__ p_out0 = (WORD16*) p_out;
        valign a_out = PDX_Z_ALIGN();

        /* Calculate the byte stride for input data */
        WORD32 stride_val = inp_stride[p_5D_permute_vec[4]];
        WORD32 stride_val_bytes = stride_val * elm_size;

        for (itr0 = 0; itr0 < out_dim0; itr0++)
        {
            WORD16 *__restrict__ p_inp1 = p_inp0
                    + (itr0 * inp_stride[p_5D_permute_vec[0]]);
            for (itr1 = 0; itr1 < out_dim1; itr1++)
            {
                WORD16 *__restrict__ p_inp2 = p_inp1
                        + (itr1 * inp_stride[p_5D_permute_vec[1]]);
                for (itr2 = 0; itr2 < out_dim2; itr2++)
                {
                    WORD16 *__restrict__ p_inp3 = p_inp2
                            + (itr2 * inp_stride[p_5D_permute_vec[2]]);
                    for (itr3 = 0; itr3 < out_dim3; itr3++)
                    {
                        WORD16 *__restrict__ p_inp4 = p_inp3
                                + (itr3 * inp_stride[p_5D_permute_vec[3]]);

#pragma unroll 2
                        for (itr4 = 0; itr4 < (out_dim4 >> LOG2_PDX_2M); itr4++)
                        {
                            d1 = PDX_LSR_16_X(p_inp4, stride_val_bytes);

                            db = PDX_LS_16_X(p_inp4, 2 * stride_val_bytes);
                            d2 = PDX_MOV_2MX16_FROM_16(db);

                            d3 = PDX_LSR_16_X(p_inp4, 3 * stride_val_bytes);

                            dc = PDX_LS_16_X(p_inp4, 4 * stride_val_bytes);
                            d4 = PDX_MOV_2MX16_FROM_16(dc);

                            d5 = PDX_LSR_16_X(p_inp4, 5 * stride_val_bytes);

                            dd = PDX_LS_16_X(p_inp4, 6 * stride_val_bytes);
                            d6 = PDX_MOV_2MX16_FROM_16(dd);

                            d7 = PDX_LSR_16_X(p_inp4, 7 * stride_val_bytes);

                            PDX_LS_16_XP(da, p_inp4, 8 * stride_val_bytes);
                            d0 = PDX_MOV_2MX16_FROM_16(da);

                            /* Apply selection patterns and combine the vectors */
                            data10 = PDX_SEL_2MX16(d1, d0, sel_pat_1);
                            data32 = PDX_SEL_2MX16(d3, d2, sel_pat_2);
                            data54 = PDX_SEL_2MX16(d5, d4, sel_pat_3);
                            data76 = PDX_SEL_2MX16(d7, d6, sel_pat_4);

                            data3210 = PDX_OR_2MX16(data10, data32);
                            data7654 = PDX_OR_2MX16(data54, data76);
                            data = PDX_OR_2MX16(data3210, data7654);

                            PDX_SA_2MX16_IP(data, a_out, (xb_vec2Mx16*) p_out0);
                        }
                        PDX_SAPOS_2MX16_FP(a_out, (xb_vec2Mx16*) p_out0);
#pragma loop_count max=7
                        /* Handle any remaining elements if the output dimension is not a multiple of the vector size */
                        for (itr4 = 0; itr4 < (out_dim4 & (PDX_2M - CONST_ONE)); itr4++)
                        {
                            xb_vec2Mx16 d0;
                            PDX_LSR_16_XP(d0, p_inp4, stride_val_bytes);
                            PDX_SS_16_XP(d0, p_out0, elm_size);
                        }
                    }
                }
            }
        }
    }
    /*If elm_size == 1, then input data type is 8-bit char */
    else if (elm_size == CONST_ONE)
    {
        /* Selection patterns for vector operations */
        xb_vec4Mx8 sel_pat_1 =
                { 0, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        xb_vec4Mx8 sel_pat_2 =
                { 1, 1, 0, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        xb_vec4Mx8 sel_pat_3 =
                { 1, 1, 1, 1, 0, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        xb_vec4Mx8 sel_pat_4 =
                { 1, 1, 1, 1, 1, 1, 0, 16, 1, 1, 1, 1, 1, 1, 1, 1 };
        xb_vec4Mx8 sel_pat_5 =
                { 1, 1, 1, 1, 1, 1, 1, 1, 0, 16, 1, 1, 1, 1, 1, 1 };
        xb_vec4Mx8 sel_pat_6 =
                { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 16, 1, 1, 1, 1 };
        xb_vec4Mx8 sel_pat_7 =
                { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 16, 1, 1 };
        xb_vec4Mx8 sel_pat_8 =
                { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 16 };

        xb_vec4Mx8 d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13,
                d14, d15;
        xb_vec4Mx8 data01, data23, data45, data67, data89, data1011, data1213, data1415;
        xb_vec4Mx8 data0_3, data4_7, data8_11, data12_15;
        xb_vec4Mx8 data0_7, data8_15;
        xb_vec4Mx8 data0_15;
        xb_int8 da, db, dc, dd, de, df, dg, dh;

        WORD32 itr0, itr1, itr2, itr3, itr4;
        WORD8 *__restrict__ p_inp0 = (WORD8*) p_inp;
        WORD8 *__restrict__ p_out0 = (WORD8*) p_out;
        valign a_out = PDX_Z_ALIGN();

        /* Calculate the byte stride for input data */
        WORD32 stride_val = inp_stride[p_5D_permute_vec[4]];
        WORD32 stride_val_bytes = stride_val * elm_size;

        /* 2 pointers p_inp4_0, p_inp4_1 are created
         p_inp4_0 store values in d0,d2,d4,...,d14
         p_inp4_1 store values in d1,d3,d5,...,d15
         for that doubled_stride_bytes should be used. */

        WORD32 doubled_stride_bytes = stride_val_bytes * CONST_TWO;

        for (itr0 = 0; itr0 < out_dim0; itr0++)
        {
            WORD8 *__restrict__ p_inp1 = p_inp0
                    + (itr0 * inp_stride[p_5D_permute_vec[0]]);
            for (itr1 = 0; itr1 < out_dim1; itr1++)
            {
                WORD8 *__restrict__ p_inp2 = p_inp1
                        + (itr1 * inp_stride[p_5D_permute_vec[1]]);
                for (itr2 = 0; itr2 < out_dim2; itr2++)
                {
                    WORD8 *__restrict__ p_inp3 = p_inp2
                            + (itr2 * inp_stride[p_5D_permute_vec[2]]);
                    for (itr3 = 0; itr3 < out_dim3; itr3++)
                    {
                        WORD8 *__restrict__ p_inp4 = p_inp3
                                + (itr3 * inp_stride[p_5D_permute_vec[3]]);
                        WORD8 *__restrict__ p_inp4_0 = p_inp4;
                        WORD8 *__restrict__ p_inp4_1 = p_inp4
                                + inp_stride[p_5D_permute_vec[4]];


#pragma unroll 2
                        for (itr4 = 0; itr4 < (out_dim4 >> LOG2_PDX_4M); itr4++)
                        {
                            PDX_LS_8_XP(da, p_inp4_0, doubled_stride_bytes);
                            d0 = PDX_MOV_4MX8_FROM_8(da);

                            PDX_LSR_8_XP(d1, p_inp4_1, doubled_stride_bytes);

                            PDX_LS_8_XP(db, p_inp4_0, doubled_stride_bytes);
                            d2 = PDX_MOV_4MX8_FROM_8(db);

                            PDX_LSR_8_XP(d3, p_inp4_1, doubled_stride_bytes);

                            PDX_LS_8_XP(dc, p_inp4_0, doubled_stride_bytes);
                            d4 = PDX_MOV_4MX8_FROM_8(dc);

                            PDX_LSR_8_XP(d5, p_inp4_1, doubled_stride_bytes);

                            PDX_LS_8_XP(dd, p_inp4_0, doubled_stride_bytes);
                            d6 = PDX_MOV_4MX8_FROM_8(dd);

                            PDX_LSR_8_XP(d7, p_inp4_1, doubled_stride_bytes);

                            PDX_LS_8_XP(de, p_inp4_0, doubled_stride_bytes);
                            d8 = PDX_MOV_4MX8_FROM_8(de);

                            PDX_LSR_8_XP(d9, p_inp4_1, doubled_stride_bytes);

                            PDX_LS_8_XP(df, p_inp4_0, doubled_stride_bytes);
                            d10 = PDX_MOV_4MX8_FROM_8(df);

                            PDX_LSR_8_XP(d11, p_inp4_1, doubled_stride_bytes);

                            PDX_LS_8_XP(dg, p_inp4_0, doubled_stride_bytes);
                            d12 = PDX_MOV_4MX8_FROM_8(dg);

                            PDX_LSR_8_XP(d13, p_inp4_1, doubled_stride_bytes);

                            PDX_LS_8_XP(dh, p_inp4_0, doubled_stride_bytes);
                            d14 = PDX_MOV_4MX8_FROM_8(dh);

                            PDX_LSR_8_XP(d15, p_inp4_1, doubled_stride_bytes);

                            /* Apply selection patterns and combine the vectors */
                            data01 = PDX_SEL_4MX8(d1, d0, sel_pat_1);
                            data23 = PDX_SEL_4MX8(d3, d2, sel_pat_2);
                            data45 = PDX_SEL_4MX8(d5, d4, sel_pat_3);
                            data67 = PDX_SEL_4MX8(d7, d6, sel_pat_4);
                            data89 = PDX_SEL_4MX8(d9, d8, sel_pat_5);
                            data1011 = PDX_SEL_4MX8(d11, d10, sel_pat_6);
                            data1213 = PDX_SEL_4MX8(d13, d12, sel_pat_7);
                            data1415 = PDX_SEL_4MX8(d15, d14, sel_pat_8);

                            data0_3 = PDX_OR_4MX8(data01, data23);
                            data4_7 = PDX_OR_4MX8(data45, data67);
                            data8_11 = PDX_OR_4MX8(data89, data1011);
                            data12_15 = PDX_OR_4MX8(data1213, data1415);

                            data0_7 = PDX_OR_4MX8(data0_3, data4_7);
                            data8_15 = PDX_OR_4MX8(data8_11, data12_15);
                            data0_15 = PDX_OR_4MX8(data0_7, data8_15);

                            PDX_SA_4MX8_IP(data0_15, a_out,
                                    (xb_vec4Mx8*) p_out0);
                        }
                        PDX_SAPOS_4MX8_FP(a_out, (xb_vec4Mx8*) p_out0);
#pragma loop_count max=15
                        /* Handle any remaining elements if the output dimension is not a multiple of the vector size */
                        for (itr4 = 0; itr4 < (out_dim4 & PDX_4M - CONST_ONE); itr4++)
                        {
                            xb_vec4Mx8 d0;
                            PDX_LSR_8_XP(d0, p_inp4_0, stride_val_bytes);
                            PDX_SS_8_XP(d0, p_out0, elm_size);
                        }
                    }
                }
            }
        }
    }

    return 0;
}

WORD32 xa_nn_permute(WORD8 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        const WORD8 *__restrict__ p_inp,
        const WORD32 *const p_inp_shape,
        const WORD32 *__restrict__ p_permute_vec,
        WORD32 num_inp_dims,
        WORD32 elm_size)
{
    /* Always num_out_dims == num_inp_dims */
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_permute_vec, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_shape, UNSUPPORTED_PARAM);

    /* Invalid input checks */
    XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);

    WORD32 i, itr;


    /* Shape values should be greater than 0 .
     * permute vector indices should be positive. */
    for (i = 0; i < num_inp_dims; i++)
    {
        XA_NNLIB_ARG_CHK_COND((p_out_shape[i] <= 0), UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_COND((p_inp_shape[i] <= 0), UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_COND((p_permute_vec[i] < 0), UNSUPPORTED_PARAM);
    }

    /*
     * Output shape provided must be correct based on input
     * shape and permute values.
     */
    for (itr = 0; itr < num_inp_dims; itr++)
    {
        WORD32 output_dim = p_out_shape[itr];
        WORD32 expected_dim = p_inp_shape[p_permute_vec[itr]];
        XA_NNLIB_ARG_CHK_COND((output_dim != expected_dim), UNSUPPORTED_PARAM);
    }

    XA_NNLIB_ARG_CHK_COND((elm_size <= 0) || (elm_size == CONST_THREE)
                 || (elm_size > CONST_FOUR), UNSUPPORTED_PARAM);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8) * elm_size, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8) * elm_size, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_permute_vec, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), UNSUPPORTED_PARAM);


    /* Supports upto 5D input tensors.*/
    WORD32 p_5D_inp_shape[MAX_DIMS] = { CONST_ONE, CONST_ONE, CONST_ONE, CONST_ONE, CONST_ONE };
    WORD32 p_5D_out_shape[MAX_DIMS] = { CONST_ONE, CONST_ONE, CONST_ONE, CONST_ONE, CONST_ONE };
    WORD32 p_5D_permute_vec[MAX_DIMS] = { 0, CONST_ONE, CONST_TWO, CONST_THREE, CONST_FOUR };
    WORD32 last_n_same_dim;
    last_n_same_dim = init_prep(p_out_shape, p_inp_shape, p_permute_vec,
            num_inp_dims, p_5D_inp_shape, p_5D_out_shape, p_5D_permute_vec);

    if (last_n_same_dim)
    {
        last_n_same_dim_case(p_out, p_5D_out_shape, p_inp, p_5D_inp_shape,
                p_5D_permute_vec, elm_size);
    }
    else
    {
        last_dim_not_same_case(p_out, p_5D_out_shape, p_inp, p_5D_inp_shape,
                p_5D_permute_vec, elm_size);
    }

    return 0;
}
