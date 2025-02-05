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

/*
    tables for tanhf() approximation
*/
#ifndef TANHF_TBL_H__
#define TANHF_TBL_H__

#include "xa_type_def.h"
#include "xa_nnlib_common_internal.h"

/* polynomial approximation of tanh(x) in range [-log(3)/2...-log(3)/2]
    only odd coefficients are non zero
    s=pow2(2,-16);
    x=[s:s:log(3)/2+0.008]; x=[-x(end:-1:1) x];
    y=tanh(x); z=tanh(x)./x;
    p=polyfit(x,z,8);
    p=p(1:2:end); p(end)=[];
*/

extern const WORD32 tbl[3];

extern const FLOAT32 polytanhf_tbl[4];

extern const FLOAT32 halfln3; /* log(3)/2 - tanh(log(3)/2)==0.5 */

#endif
