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

#include "tanhf_tbl.h"

const WORD32 tbl[3] = {(WORD32) 1549082005L, /* 1/ln(2), Q30 */
    (WORD32) 127 - 31, (WORD32) 255};

const FLOAT32 polytanhf_tbl[4]=
{
    1.6437442973e-002,
    -5.2661579102e-002,
    1.3320191205e-001,
    -3.3332869411e-001
};

const FLOAT32 halfln3={0.5493061542510986328125} ; /* log(3)/2 - tanh(log(3)/2)==0.5 */

