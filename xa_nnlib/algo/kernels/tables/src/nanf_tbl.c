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

#include "nanf_tbl.h"

const union UFLOAT32UWORD32 ALIGN(32) xa_nnlib_sNaNf       = { 0x7f800001 }; /* Signalling NaN          */
const union UFLOAT32UWORD32 ALIGN(32) xa_nnlib_qNaNf       = { 0x7fc00000 }; /* Quiet NaN               */
const union UFLOAT32UWORD32 ALIGN(32) xa_nnlib_minus_sNaNf = { 0xff800001 }; /* Negative Signalling NaN */
const union UFLOAT32UWORD32 ALIGN(32) xa_nnlib_minus_qNaNf = { 0xffc00000 }; /* Negative Quiet NaN      */

