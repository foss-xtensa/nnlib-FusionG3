#
# Copyright (c) 2024 Cadence Design Systems, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to use this Software with Cadence processor cores only and
# not with any other processors and platforms, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

ISS = xt-run $(XTCORE)
CONFIGDIR := $(shell $(ISS) --show-config=config)
include $(CONFIGDIR)/misc/hostenv.mk

GREPARGS =
ifeq ($(HOSTTYPE),win)
GREPARGS = /c:
endif

ifeq ("", "$(detected_core)")

fusion_g3="0"
fusion_g3_tmp:=$(shell $(GREP) $(GREPARGS)"IsaUseFusionG = 1"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")

#check if the detected core is Fusion G3
    ifneq ("", "$(fusion_g3_tmp)")
        detected_core=fusion_g3
    endif

ifeq ("$(detected_core)", "fusion_g3")
    fusion_g3=1
    CFLAGS+= -DCORE_FUG3=1
else
    $(error "$(fusion_g3_tmp)" Core Not Found)
endif
endif

xclib_tmp:=$(shell $(GREP) $(GREPARGS)"SW_CLibrary = xclib"  "$(XTENSA_SYSTEM)$(S)$(XTENSA_CORE)-params")
ifneq ("", "$(xclib_tmp)")
    xclib=1
else
    xclib=0
endif

