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

// Function to check if two consecutive axes are contiguous
WORD32 are_two_axes_contiguous(WORD32 a, WORD32 b) {
    return (b == a + 1);
}


// Function to check if an axis is in the reduced list
WORD32 is_axis_preserved(WORD32 axis, WORD32 *axes, WORD32 num_axes) {
    for (WORD32 i = 0; i < num_axes; i++) {
        if (axes[i] == axis)
		{
			return 1;
		}
    }
    return 0;
}
// Check if an axis is in the given axes list
WORD32 is_axis_in_list(WORD32 axis, const WORD32 *axes, WORD32 num_axes) {
    for (WORD32 i = 0; i < num_axes; i++) {
        if (axes[i] == axis)
		{
        	return 1;
		}
    }
    return 0;
}
// Merge contiguous subsets of axes while keeping preserved axes intact
void merge_axes(const WORD32 *const shape, WORD32 num_dims, const WORD32 *axes, WORD32 num_axes,
                WORD32 *new_input_shape, WORD32 *new_num_dims, WORD32 *new_axes, WORD32 *new_num_axes) {
    *new_num_dims = 0;
    *new_num_axes = 0;

    WORD32 i = 0;
    WORD32 merge_occurred = 0;
    WORD32 axis_map[MAX_DIMS]; // Map original axes to new indices

    while (i < num_dims) {
        if (is_axis_in_list(i, axes, num_axes)) {
            // If the axis is in the given axes list, check if it can merge
            WORD32 merged_size = shape[i];
            WORD32 start_idx = i;

            while (i + 1 < num_dims && is_axis_in_list(i + 1, axes, num_axes)
                   && are_two_axes_contiguous(i, i + 1)) {
                merged_size *= shape[i + 1];
                i++;
                merge_occurred = 1;
            }

            // Store merged dimension
            new_input_shape[*new_num_dims] = merged_size;
            axis_map[start_idx] = *new_num_dims; // First axis maps to merged index
            new_axes[*new_num_axes] = *new_num_dims; // Store the merged axis position
            (*new_num_dims)++;
            (*new_num_axes)++;
            i++;
        } else {
            // Copy unmerged dimensions
            new_input_shape[*new_num_dims] = shape[i];
            axis_map[i] = *new_num_dims;
            (*new_num_dims)++;
            i++;
        }
    }

    // Update the new axes with the correct mapped positions
    *new_num_axes = 0;
    for (WORD32 j = 0; j < num_axes; j++) {
        if (axis_map[axes[j]] >= 0) {
            new_axes[*new_num_axes] = axis_map[axes[j]];
            (*new_num_axes)++;
        }
    }

    // If no merging was possible, restore the original shape and axes
    if (!merge_occurred) {
        for (WORD32 i = 0; i < num_dims; i++) {
            new_input_shape[i] = shape[i];
        }
        for (WORD32 i = 0; i < num_axes; i++) {
            new_axes[i] = axes[i];
        }
        *new_num_dims = num_dims;
        *new_num_axes = num_axes;
    }
}
void merge_dims(WORD32 *shape, WORD32 num_dims, WORD32 *axes, WORD32 num_axes,
                WORD32 *new_input_shape, WORD32 *new_num_dims, WORD32 *new_axes, WORD32 *new_num_axes) {
    *new_num_dims = 0;
    *new_num_axes = 0;
    WORD32 i = 0;

    while (i < num_dims) {
        // If the current dimension is a preserved axis, do not merge it
        if (is_axis_preserved(i, axes, num_axes)) {
            new_input_shape[*new_num_dims] = shape[i];
            new_axes[*new_num_axes] = *new_num_dims; // Store new index
            (*new_num_dims)++;
            (*new_num_axes)++;
            i++; // Move to the next dimension
        } else {
            // Start merging contiguous dimensions
            WORD32 merged_size = shape[i];
            while (i + 1 < num_dims && !is_axis_preserved(i + 1, axes, num_axes)) {
                merged_size *= shape[i + 1];
                i++; // Move to the next dimension
            }
            new_input_shape[*new_num_dims] = merged_size;
            (*new_num_dims)++;
            i++; // Move past the last merged dimension
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

	if ((p_axis == NULL ) || (num_axis_dims == 0) || (num_axis_dims == num_inp_dims))
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
	/* For contiguous axis merge */
	WORD32 new_shape1[MAX_DIMS];  // Buffer for new shape
	WORD32 new_axes1[MAX_DIMS];   // Buffer for new axes
	WORD32 new_input_shape[MAX_DIMS];  // Buffer for new shape
	WORD32 new_axes[MAX_DIMS];   // Buffer for new axes
	WORD32 new_num_dims1 = 0, new_num_axes1 = 0;

	merge_axes(p_inp_shape, num_inp_dims, p_axis, num_axis_dims, new_shape1, &new_num_dims1, new_axes1, &new_num_axes1);

	/* For contigous dimensions merge other than axis */
	WORD32 new_num_dims = 0, new_num_axes = 0;
	merge_dims(new_shape1, new_num_dims1, new_axes1, new_num_axes1, new_input_shape, &new_num_dims, new_axes, &new_num_axes);

	WORD32 last_dim = 0;
	for(WORD32 i = 0; i < new_num_axes; i++)
	{
		if(new_axes[i] == new_num_dims - CONST_ONE)
		{
			last_dim = CONST_ONE;
		}
	}

	switch(new_num_dims)
	{
		case 2:
		{
			if(last_dim)
			{
				xb_vecMxf32 x0;
				const xb_vecMxf32 *__restrict__ p_src1 = (xb_vecMxf32 *)p_inp;
				valign align_src1 = PDX_LA_MXF32_PP(p_src1);
				WORD32 axis_count = new_input_shape[CONST_ONE];
				WORD32 out_loop_count = new_input_shape[0];

				WORD32 rem_elm = (axis_count & (PDX_M - CONST_ONE)) * sizeof(float);

				for (WORD32 d0 = 0; d0 < out_loop_count; d0++)
				{
					xb_vecMxf32 sum = PDX_ZERO_MXF32();
					for (WORD32 d1 = 0; d1 < axis_count >> CONST_TWO; d1++)
					{
						PDX_LA_MXF32_IP(x0, align_src1, p_src1);
						sum = PDX_ADD_MXF32(sum, x0);
					}
					PDX_LAV_MXF32_XP(x0, align_src1, p_src1, rem_elm);
					sum = PDX_ADD_MXF32(sum, x0);
					// Reduce and store result
					float out = PDX_RADD_MXF32(sum);
					p_out[d0] = out;
				}
			}
			else
			{
				xb_vecMxf32 x0, x1, x2;
				const xb_vecMxf32 *__restrict__ p_src1 = (xb_vecMxf32 *)p_inp;
				const xb_vecMxf32 *__restrict__ p_src2 = (xb_vecMxf32 *)p_inp;
				valign align_src1 = PDX_LA_MXF32_PP(p_src1);
				valign align_src2 = PDX_LA_MXF32_PP(p_src2);
				xb_vecMxf32 *__restrict__ p_dst = (xb_vecMxf32 *)p_out;
				valign align_dst = PDX_Z_ALIGN();
				WORD32 axis_count = new_input_shape[0];

				WORD32 out_loop_count = new_input_shape[1];
				WORD32 inner_stride_bytes = out_loop_count << CONST_TWO;
				WORD32 rem_elem = ((out_loop_count & (PDX_M - CONST_ONE)) * sizeof(float));
				WORD32 rem_axis = ((axis_count - CONST_ONE) & (CONST_ONE));
				if(axis_count == CONST_ONE)
				{
					for (WORD32 j = 0; j < out_loop_count - CONST_THREE; j += CONST_FOUR)
					{
						PDX_LA_MXF32_IP(x1, align_src1, p_src1);
						PDX_SA_MXF32_IP(x1, align_dst, p_dst);
					}
					PDX_LAV_MXF32_XP(x1, align_src1, p_src1, rem_elem);
					PDX_SAV_MXF32_XP(x1, align_dst, p_dst, rem_elem);
					PDX_SAPOS_MXF32_FP(align_dst, p_dst);
				}
				else
				{
					WORD32 d1 = 0;
					for (d1 = 0; d1 < out_loop_count - CONST_THREE ; d1 +=CONST_FOUR)
					{
						xb_vecMxf32 sum = PDX_ZERO_MXF32();
						FLOAT32 *p_inp1 = (FLOAT32 *)p_inp + d1;

						p_src1 = (const xb_vecMxf32 *)(p_inp1);
						p_src2 = (const xb_vecMxf32 *)(p_inp1 + out_loop_count);
						align_src1 = PDX_LA_MXF32_PP(p_src1);
						PDX_LA_MXF32_XP(x0, align_src1, p_src1, inner_stride_bytes * 2);
						sum = sum + x0;
						for (WORD32 d0 = 0; d0 < (axis_count - CONST_ONE) >> CONST_ONE; d0++)
						{
							/* Align load priming of input */
							align_src2 = PDX_LA_MXF32_PP(p_src2);
							align_src1 = PDX_LA_MXF32_PP(p_src1);

							/* Load input elements with stride "inner_stride" */
							PDX_LA_MXF32_XP(x1, align_src2, p_src2, inner_stride_bytes * 2);
							PDX_LA_MXF32_XP(x2, align_src1, p_src1, inner_stride_bytes * 2);

							/* Calculate sum across each lane of vector */
							sum = sum + PDX_ADD_MXF32(x1, x2);
						}
						if (rem_axis)
						{
							/* Align load priming of input */
							align_src2 = PDX_LA_MXF32_PP(p_src2);

							/* Load input elements with stride "inner_stride" */
							PDX_LA_MXF32_XP(x1, align_src2, p_src2, inner_stride_bytes * 2);

							/* Calculate maximum across each lane of vector */
							sum = PDX_ADD_MXF32(sum, x1);
						}
						/* Store output */
						PDX_SA_MXF32_IP(sum, align_dst, p_dst);
					}
					if(rem_elem)
					{
						 /* Process remaining elements */
						FLOAT32 *p_inp2 = (FLOAT32 *)(p_inp + d1);

						const FLOAT32 *p_inp3 = (FLOAT32 *)p_inp2;
						const xb_vecMxf32 *__restrict__ p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);

						xb_vecMxf32 rem_sum = 0;
						align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);
						PDX_LAV_MXF32_XP(rem_sum, align_src1, p_in_mxf32, rem_elem);

						for (WORD32 k = 0; k < axis_count - CONST_ONE; k++)
						{
							p_inp3 += out_loop_count;
							p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);

							/* Align load priming of input */
							align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);

							/* Load input elements with stride "inner_stride" */
							PDX_LAV_MXF32_XP(x1, align_src1, p_in_mxf32, rem_elem);

							rem_sum = PDX_ADD_MXF32(rem_sum, x1);
						}

						/* Store output */
						PDX_SAV_MXF32_XP(rem_sum, align_dst, p_dst, rem_elem);
						PDX_SAPOS_MXF32_FP(align_dst, p_dst);
					}
				}

			}
		}
			break;
		case 3:
		{
			if(last_dim)
			{
				xb_vecMxf32 x0;
				const xb_vecMxf32 *__restrict__ p_src1 = (xb_vecMxf32 *)p_inp;
				valign align_src1 = PDX_LA_MXF32_PP(p_src1);

				WORD32 Dim0 = new_input_shape[0];
				WORD32 Dim1 = new_input_shape[1];
				WORD32 Dim2 = new_input_shape[2];

				WORD32 out_loop_count = Dim1;

				WORD32 rem_elem = ((Dim2 & (PDX_M - CONST_ONE)) * sizeof(float));
				for(WORD32 d1 = 0; d1 < out_loop_count; d1++)
				{
					WORD32 base_offset = d1 * Dim2;
					xb_vecMxf32 sum = PDX_ZERO_MXF32();
					for(WORD32 d0 = 0; d0 < Dim0; d0++)
					{
						FLOAT32 *p_inp1 = (FLOAT32 *)(p_inp + (d0 * Dim1 * Dim2) + base_offset);
						p_src1 = (xb_vecMxf32 *)p_inp1;
						align_src1 = PDX_LA_MXF32_PP(p_src1);
						for(WORD32 d2 = 0; d2 < Dim2 >> 2; d2++)
						{
							PDX_LA_MXF32_IP(x0, align_src1, p_src1);
							sum = PDX_ADD_MXF32(sum, x0);
						}
						PDX_LAV_MXF32_XP(x0, align_src1, p_src1, rem_elem);
						sum = PDX_ADD_MXF32(sum, x0);
					}
					// Reduce and store result
					float out = PDX_RADD_MXF32(sum);
					p_out[d1] = out;
				}
			}
			else
			{
				xb_vecMxf32 x0, x1, x2;//, out;
				const xb_vecMxf32 *__restrict__ p_src1 = (xb_vecMxf32 *)p_inp;
				const xb_vecMxf32 *__restrict__ p_src2 = (xb_vecMxf32 *)p_inp;
				valign align_src1 = PDX_LA_MXF32_PP(p_src1);
				valign align_src2;
				const xb_vecMxf32 *__restrict__ p_in_mxf32 = (xb_vecMxf32 *)p_inp;
				xb_vecMxf32 *__restrict__ p_dst = (xb_vecMxf32 *)p_out;
				valign align_dst = PDX_Z_ALIGN();

				WORD32 Dim0 = new_input_shape[0];
				WORD32 Dim1 = new_input_shape[1];
				WORD32 Dim2 = new_input_shape[2];

				WORD32 axis_count = Dim1;
				WORD32 out_loop_count = Dim0 * Dim2;

				WORD32 inner_stride_bytes = Dim2 << CONST_TWO;
				WORD32 rem_elem = ((Dim2 & (PDX_M - CONST_ONE)) * sizeof(float));
				WORD32 rem_axis = ((axis_count - CONST_ONE) & (CONST_ONE));
				if(axis_count == 1)
				{
					for (WORD32 j = 0; j < out_loop_count - CONST_THREE; j +=CONST_FOUR)
					{
						PDX_LA_MXF32_IP(x1, align_src1, p_src1);
						PDX_SA_MXF32_IP(x1, align_dst, p_dst);
					}
					PDX_LAV_MXF32_XP(x1, align_src1, p_src1, rem_elem);
					PDX_SAV_MXF32_XP(x1, align_dst, p_dst, rem_elem);
					PDX_SAPOS_MXF32_FP(align_dst, p_dst);
				}
				else
				{
					WORD32 d2 = 0;
					for(WORD32 d0d2 = 0; d0d2 < Dim0; d0d2++)
					{
						WORD32 base_offset = d0d2 * Dim1 * Dim2;
						xb_vecMxf32 sum = PDX_ZERO_MXF32();
						for (d2 = 0; d2 < Dim2 - CONST_THREE ; d2 +=CONST_FOUR)
						{
							float *p_inp2 = (float*)(p_inp + d2 + base_offset);
							sum = PDX_ZERO_MXF32();
							p_src1 = (const xb_vecMxf32 *)(p_inp2);
							p_src2 = (const xb_vecMxf32 *)(p_inp2 + Dim2);
							align_src1 = PDX_LA_MXF32_PP(p_src1);
							PDX_LA_MXF32_XP(x0, align_src1, p_src1, inner_stride_bytes * 2);
							sum = sum + x0;
							for (WORD32 d1 = 0; d1 < (axis_count - CONST_ONE) >> CONST_ONE; d1++)
							{
								/* Align load priming of input */
								valign align_src2 = PDX_LA_MXF32_PP(p_src2);
								align_src1 = PDX_LA_MXF32_PP(p_src1);

								/* Load input elements with stride "inner_stride" */
								PDX_LA_MXF32_XP(x1, align_src2, p_src2, inner_stride_bytes * 2);
								PDX_LA_MXF32_XP(x2, align_src1, p_src1, inner_stride_bytes * 2);

								/* Calculate sum across each lane of vector */
								sum = sum + PDX_ADD_MXF32(x1, x2);
							}
							if (rem_axis)
							{
								/* Align load priming of input */
								align_src2 = PDX_LA_MXF32_PP(p_src2);

								/* Load input elements with stride "inner_stride" */
								PDX_LA_MXF32_XP(x1, align_src2, p_src2, inner_stride_bytes * 2);

								/* Calculate maximum across each lane of vector */
								sum = PDX_ADD_MXF32(sum, x1);
							}
							/* Store output */
							PDX_SA_MXF32_IP(sum, align_dst, p_dst);
						}
						if(rem_elem)
						{
							 /* Process remaining elements */
							float *p_inp2 = (float *)(p_inp + d2 + base_offset);

							const FLOAT32 *p_inp3 = p_inp2;
						    p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);

							xb_vecMxf32 rem_sum = 0;
							align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);
							PDX_LAV_MXF32_XP(rem_sum, align_src1, p_in_mxf32, rem_elem);

							for (WORD32 k = 0; k < axis_count - CONST_ONE; k++)
							{
								p_inp3 += Dim2;
								p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);

								/* Align load priming of input */
								align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);

								/* Load input elements with stride "inner_stride" */
								PDX_LAV_MXF32_XP(x1, align_src1, p_in_mxf32, rem_elem);

								rem_sum = PDX_ADD_MXF32(rem_sum, x1);
							}
							/* Store output */
							PDX_SAV_MXF32_XP(rem_sum, align_dst, p_dst, rem_elem);
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
				xb_vecMxf32 x0;
				const xb_vecMxf32 *__restrict__ p_src1 = (xb_vecMxf32 *)p_inp;
				valign align_src1 = PDX_LA_MXF32_PP(p_src1);

				WORD32 Dim0 = new_input_shape[0];
				WORD32 Dim1 = new_input_shape[1];
				WORD32 Dim2 = new_input_shape[2];
				WORD32 Dim3 = new_input_shape[3];

				WORD32 out_loop_count = Dim0 * Dim2;

				WORD32 d_0 = 0, d_2 = 0, d2_offset = 0, d0_offset = 0;
				WORD32 rem_elem = ((Dim3 & (PDX_M - CONST_ONE)) * sizeof(float));
				for(WORD32 d0d2 = 0; d0d2 < out_loop_count; d0d2++)
				{
					xb_vecMxf32 sum = PDX_ZERO_MXF32();
					for(WORD32 d1 = 0; d1 < Dim1; d1++)
					{
						float *p_inp1 = (float*)p_inp  + (d1 * Dim2 * Dim3) + d0_offset + d2_offset;
						p_src1 = (xb_vecMxf32 *)p_inp1;
						align_src1 = PDX_LA_MXF32_PP(p_src1);
						for(WORD32 d3 = 0; d3 < Dim3 >> CONST_TWO; d3++)
						{
							PDX_LA_MXF32_IP(x0, align_src1, p_src1);
							sum = PDX_ADD_MXF32(sum, x0);
						}
						PDX_LAV_MXF32_XP(x0, align_src1, p_src1, rem_elem);
						sum = PDX_ADD_MXF32(sum, x0);
					}
					// Reduce and store result
					float out = PDX_RADD_MXF32(sum);
					p_out[d0d2] = out;

					d2_offset += Dim3;
					if (++d_2 == Dim2) {
						d_2 = 0;
						d2_offset = 0;
						d_0++;
						d0_offset += Dim1 * Dim2 * Dim3;  // Move to next d0 block
					}
				}
			}
			else
			{
				xb_vecMxf32 x0, x1, x2;
				const xb_vecMxf32 *__restrict__ p_src1 = (xb_vecMxf32 *)p_inp;
				const xb_vecMxf32 *__restrict__ p_src2 = (xb_vecMxf32 *)p_inp;
				const xb_vecMxf32 *__restrict__ p_in_mxf32 = (xb_vecMxf32 *)p_inp;
				valign align_src1 = PDX_LA_MXF32_PP(p_src1);
				valign align_src2;
				xb_vecMxf32 *__restrict__ p_dst = (xb_vecMxf32 *)p_out;
				valign align_dst = PDX_Z_ALIGN();

				WORD32 Dim0 = new_input_shape[0];
				WORD32 Dim1 = new_input_shape[1];
				WORD32 Dim2 = new_input_shape[2];
				WORD32 Dim3 = new_input_shape[3];

				WORD32 out_loop_count = Dim1 ;

				WORD32 inner_stride_bytes = Dim3 << CONST_TWO;
				WORD32 rem_elem = ((Dim3 & (PDX_M - CONST_ONE)) * sizeof(float));
				WORD32 rem_axis = ((Dim2 - CONST_ONE) & (CONST_ONE));
				WORD32 d3 = 0;
				for(WORD32 d1d3 = 0; d1d3 < out_loop_count; d1d3++)
				{
					WORD32 base_offset = (d1d3 * Dim2 * Dim3);
					xb_vecMxf32 sum = PDX_ZERO_MXF32();
					for (d3 = 0; d3 < Dim3 - CONST_THREE ; d3 +=CONST_FOUR)
					{
						float *p_inp2 = (float *)(p_inp + d3 + base_offset);
						sum = PDX_ZERO_MXF32();
						for(WORD32 d0 = 0; d0 < Dim0; d0++)
						{
							float *temp_ptr = p_inp2 + (d0 * Dim1 * Dim2 * Dim3);
							p_src1 = (const xb_vecMxf32 *)(temp_ptr);
							p_src2 = (const xb_vecMxf32 *)(temp_ptr + Dim3);
							align_src1 = PDX_LA_MXF32_PP(p_src1);
							PDX_LA_MXF32_XP(x0, align_src1, p_src1, inner_stride_bytes * 2);
							sum = sum + x0;
							for (WORD32 d1 = 0; d1 < (Dim2 - CONST_ONE) >> CONST_ONE; d1++)
							{
								/* Align load priming of input */
							    align_src2 = PDX_LA_MXF32_PP(p_src2);
								align_src1 = PDX_LA_MXF32_PP(p_src1);

								/* Load input elements with stride "inner_stride" */
								PDX_LA_MXF32_XP(x1, align_src2, p_src2, inner_stride_bytes * 2);
								PDX_LA_MXF32_XP(x2, align_src1, p_src1, inner_stride_bytes * 2);

								/* Calculate sum across each lane of vector */
								sum = sum + PDX_ADD_MXF32(x1, x2);
							}
							if (rem_axis)
							{
								/* Align load priming of input */
								align_src2 = PDX_LA_MXF32_PP(p_src2);

								/* Load input elements with stride "inner_stride" */
								PDX_LA_MXF32_XP(x1, align_src2, p_src2, inner_stride_bytes * 2);

								/* Calculate maximum across each lane of vector */
								sum = PDX_ADD_MXF32(sum, x1);
							}
						}
						/* Store output */
						PDX_SA_MXF32_IP(sum, align_dst, p_dst);
					}
					if(rem_elem)
					{
						 /* Process remaining elements */
						float *p_inp2 = (float *)(p_inp + d3 + base_offset);

						xb_vecMxf32 rem_sum = 0, x0 = 0;
						for(WORD32 d0 = 0; d0 < Dim0; d0++)
						{
							FLOAT32 *p_inp3 = (FLOAT32*)(p_inp2 +  (d0 * Dim1 * Dim2 * Dim3));
							p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);
							align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);
							PDX_LAV_MXF32_XP(x0, align_src1, p_in_mxf32, rem_elem);
							rem_sum = rem_sum + x0;
							for (WORD32 k = 0; k < Dim2 - CONST_ONE; k++)
							{
								p_inp3 += Dim3;
								p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);

								/* Align load priming of input */
								align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);

								/* Load input elements with stride "inner_stride" */
								PDX_LAV_MXF32_XP(x1, align_src1, p_in_mxf32, rem_elem);

								rem_sum = PDX_ADD_MXF32(rem_sum, x1);
							}
						}
						/* Store output */
						PDX_SAV_MXF32_XP(rem_sum, align_dst, p_dst, rem_elem);

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
				xb_vecMxf32 x0;

				const xb_vecMxf32 *__restrict__ p_src1 = (xb_vecMxf32 *)p_inp;
								valign align_src1 = PDX_LA_MXF32_PP(p_src1);

				WORD32 Dim0 = new_input_shape[0];
				WORD32 Dim1 = new_input_shape[1];
				WORD32 Dim2 = new_input_shape[2];
				WORD32 Dim3 = new_input_shape[3];
				WORD32 Dim4 = new_input_shape[4];

				WORD32 out_loop_count = Dim1 * Dim3;

				// Initialize d1, d3 manually
				WORD32 d1 = 0, d3 = 0;
				WORD32 d1_offset = 0;  // d1 * D2 * D3 * D4
				WORD32 d3_offset = 0;  // d3 * D4

				WORD32 rem_elm = (Dim4 & (PDX_M - CONST_ONE)) * sizeof(float);
				for (WORD32 d1d3 = 0; d1d3 < out_loop_count; d1d3++)
				{
					xb_vecMxf32 sum = PDX_ZERO_MXF32();

					for (WORD32 d0 = 0; d0 < Dim0; d0++)
					{
						WORD32 d0_offset = d0 * Dim1 * Dim2 * Dim3 * Dim4;

						for (WORD32 d2 = 0; d2 < Dim2; d2++)
						{
							WORD32 d2_offset = d2 * Dim3 * Dim4;

							float *p_inp1 = (float*)(p_inp + d0_offset + d1_offset + d2_offset + d3_offset);
							p_src1 = (xb_vecMxf32 *)p_inp1;
							align_src1 = PDX_LA_MXF32_PP(p_src1);

							for (WORD32 d4 = 0; d4 < Dim4 >> CONST_TWO; d4++)
							{
								PDX_LA_MXF32_IP(x0, align_src1, p_src1);
								sum = PDX_ADD_MXF32(sum, x0);
							}
							PDX_LAV_MXF32_XP(x0, align_src1, p_src1, rem_elm);
							sum = PDX_ADD_MXF32(sum, x0);
						}
					}

					// Reduce and store result
					float out = PDX_RADD_MXF32(sum);
					p_out[d1d3] = out;

					// **Manually increment d1 and d3**
					d3_offset += Dim4;  // Move to the next d3 index
					if (++d3 == Dim3) {  // Reset d3 and increment d1 when d3 reaches D3
						d3 = 0;
						d3_offset = 0;
						d1++;
						d1_offset += Dim2 * Dim3 * Dim4;  // Move to next d1 block
					}
				}
			}
			else
			{
				xb_vecMxf32 x0, x1, x2;
				const xb_vecMxf32 *__restrict__ p_src1 = (xb_vecMxf32 *)p_inp;
				const xb_vecMxf32 *__restrict__ p_src2 = (xb_vecMxf32 *)p_inp;
				const xb_vecMxf32 *__restrict__ p_in_mxf32 = (xb_vecMxf32 *)p_inp;
				valign align_src1 = PDX_LA_MXF32_PP(p_src1);
				valign align_src2;
				xb_vecMxf32 *__restrict__ p_dst = (xb_vecMxf32 *)p_out;
				valign align_dst = PDX_Z_ALIGN();

				WORD32 Dim0 = new_input_shape[0];
				WORD32 Dim1 = new_input_shape[1];
				WORD32 Dim2 = new_input_shape[2];
				WORD32 Dim3 = new_input_shape[3];
				WORD32 Dim4 = new_input_shape[4];

				WORD32 inner_stride_bytes = Dim4 << CONST_TWO;
				WORD32 rem_elem = ((Dim4 & (PDX_M - CONST_ONE)) * sizeof(float));
				WORD32 rem_axis = ((Dim3 - CONST_ONE) & (CONST_ONE));
				WORD32 d4 = 0;
				for(WORD32 d0d2d4 = 0; d0d2d4 < (Dim0 * Dim2); d0d2d4++)
				{
					WORD32 base_offset = (d0d2d4 * Dim3 * Dim4);

					xb_vecMxf32 sum = PDX_ZERO_MXF32();
					for (d4 = 0; d4 < Dim4 - CONST_THREE ; d4 +=CONST_FOUR)
					{
						sum = PDX_ZERO_MXF32();
						float *p_inp2 = (float*)(p_inp + d4 + base_offset);
						for(WORD32 d1 = 0; d1 < Dim1; d1++)
						{
							float *temp_ptr = p_inp2 + (d1 * Dim2 * Dim3 * Dim4);
							p_src1 = (const xb_vecMxf32 *)(temp_ptr);
							p_src2 = (const xb_vecMxf32 *)(temp_ptr + Dim4);
							align_src1 = PDX_LA_MXF32_PP(p_src1);
							PDX_LA_MXF32_XP(x0, align_src1, p_src1, inner_stride_bytes * 2);
							sum = sum + x0;
							for (WORD32 d3 = 0; d3 < (Dim3 - CONST_ONE) >> CONST_ONE; d3++)
							{
								/* Align load priming of input */
								valign align_src2 = PDX_LA_MXF32_PP(p_src2);
								align_src1 = PDX_LA_MXF32_PP(p_src1);

								/* Load input elements with stride "inner_stride" */
								PDX_LA_MXF32_XP(x1, align_src2, p_src2, inner_stride_bytes * 2);
								PDX_LA_MXF32_XP(x2, align_src1, p_src1, inner_stride_bytes * 2);

								/* Calculate sum across each lane of vector */
								sum = sum + PDX_ADD_MXF32(x1, x2);
							}
							if (rem_axis)
							{
								/* Align load priming of input */
								align_src2 = PDX_LA_MXF32_PP(p_src2);

								/* Load input elements with stride "inner_stride" */
								PDX_LA_MXF32_XP(x1, align_src2, p_src2, inner_stride_bytes * 2);

								/* Calculate maximum across each lane of vector */
								sum = PDX_ADD_MXF32(sum, x1);
							}
						}
						/* Store output */
						PDX_SA_MXF32_IP(sum, align_dst, p_dst);
					}
					if(rem_elem)
					{
						 /* Process remaining elements */
						const FLOAT32 *p_inp2 = (float *)(p_inp + d4 + base_offset);

						xb_vecMxf32 rem_sum = 0, x0 = 0;;
						for(WORD32 d1 = 0; d1 < Dim1; d1++)
						{
							FLOAT32 *p_inp3 = (FLOAT32*)(p_inp2 +  (d1 * Dim2 * Dim3 * Dim4));
							p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);
							align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);
							PDX_LAV_MXF32_XP(x0, align_src1, p_in_mxf32, rem_elem);
							rem_sum = rem_sum + x0;
							for (WORD32 k = 0; k < Dim3 - CONST_ONE; k++)
							{
								p_inp3 += Dim4;
								p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);

								/* Align load priming of input */
								align_src1 = PDX_LA_MXF32_PP(p_in_mxf32);

								/* Load input elements with stride "inner_stride" */
								PDX_LAV_MXF32_XP(x1, align_src1, p_in_mxf32, rem_elem);

								rem_sum = PDX_ADD_MXF32(rem_sum, x1);
							}
						}
						/* Store output */
						PDX_SAV_MXF32_XP(rem_sum, align_dst, p_dst, rem_elem);
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
	return 0;
}
