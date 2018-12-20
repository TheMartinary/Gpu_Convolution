###############################
#
# OS and architecture detection
#

# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))



###############################
#
# Cuda setup
#
# CUDA code generation flags
GENCODE_SM10    := -Wno-deprecated-gpu-targets -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -Wno-deprecated-gpu-targets -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50

#NOTE: Set this based on your device!!$(GENCODE_SM35)
GENCODE_FLAGS   := 

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif


#Try to get cuda compiler
NVCC=$(shell which nvcc)
ifeq ($(NVCC),)
NVCC=$(CUDA_PATH_BIN)/nvcc
NVCC_VALID=$(shell if [ ! -f "$(NVCC)" ]; then echo "INVALID"; else echo "VALID"; fi)
ifeq ($(NVCC_VALID),INVALID)
$(error ERROR: Could not locate nvcc. Possibly modify the CUDA_PATH variable in settings.mk)
endif
endif

#Sanity check to see if cuda path at least exists
CUDA_PATH_VALID=$(shell if [ ! -d "$(CUDA_PATH)" ]; then echo "INVALID"; else echo "VALID"; fi)
ifeq ($(CUDA_PATH_VALID),INVALID)
$(warning WARNING: Specified cuda path ($(CUDA_PATH)) is invalid!)
endif

# OS-specific build flags
LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
NVCCFLAGS += $(GENCODE_FLAGS) -D_FORCE_INLINES
ifneq ($(DARWIN),)
      LDFLAGS     += -Xlinker -rpath $(CUDA_LIB_PATH)
      NVCCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      NVCCFLAGS   += -m32
  else
      NVCCFLAGS   += -m64
  endif
endif

# Common includes and paths for CUDA
INCLUDES      := -I$(CUDA_INC_PATH) -I.
