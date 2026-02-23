# Lista collettivi
COLLECTIVES := allreduce alltoall # allgather broadcast reduce scatter
COMMON_INC  := -Isrc/common/include

# Comma helper (needed inside Make function calls like $(if) )
comma := ,

# ============== Toolchain (from environment, override via make VAR=...) ==============
# NCCL
NVCC       ?= nvcc
# NCCL_ROOT  - root of NCCL installation (expects include/ and lib/ subdirs)

# DPC++ / oneCCL
# DPCPP_CLANGXX  - path to clang++ with SYCL support (falls back to icpx)
# DPCPP_LIB      - path to DPC++ runtime libraries
# ONECCL_INSTALL - root of oneCCL installation (expects include/ and lib/ subdirs)
# SYCL_TARGET    - fsycl-targets value, auto-set by 'target' variable
#
# Usage:
#   make oneccl target=nvidia   (default, NVIDIA GPU via CUDA)
#   make oneccl target=amd      (AMD GPU via HIP/ROCm)
#   make oneccl target=intel    (Intel GPU via Level Zero)
ONECCL_CXX  ?= $(or $(DPCPP_CLANGXX),icpx)

target ?= nvidia
ifeq ($(target),amd)
    SYCL_TARGET    := amdgcn-amd-amdhsa
    SYCL_AMD_ARCH  ?= gfx908
    ONECCL_EXTRA   := -Xsycl-target-backend --offload-arch=$(SYCL_AMD_ARCH)
else ifeq ($(target),intel)
    SYCL_TARGET    := spir64
    ONECCL_EXTRA   :=
else
    # nvidia (default)
    SYCL_TARGET    ?= nvptx64-nvidia-cuda
    ONECCL_EXTRA   :=
endif

# ============== NCCL ==============
NCCL_SRC_DIR   := src/nccl
NCCL_BUILD_DIR := build/nccl
NCCL_CFLAGS    := $(if $(NCCL_ROOT),-I$(NCCL_ROOT)/include)
NCCL_LDFLAGS   := $(if $(NCCL_ROOT),-L$(NCCL_ROOT)/lib -Xlinker -rpath -Xlinker $(NCCL_ROOT)/lib)
NCCL_LIBS      := $(NCCL_CFLAGS) $(NCCL_LDFLAGS) -lmpi -lnvidia-ml -lnccl $(COMMON_INC)

nccl_collective ?= all
ifeq ($(nccl_collective),all)
NCCL_TARGETS := $(addprefix $(NCCL_BUILD_DIR)/,$(COLLECTIVES))
else
NCCL_TARGETS := $(addprefix $(NCCL_BUILD_DIR)/,$(nccl_collective))
endif

# ============== OneCCL ==============
ONECCL_SRC_DIR   := src/oneccl
ONECCL_BUILD_DIR := build/oneccl
ONECCL_CFLAGS    := $(if $(ONECCL_INSTALL),-I$(ONECCL_INSTALL)/include)
ONECCL_LDFLAGS   := $(if $(ONECCL_INSTALL),-L$(ONECCL_INSTALL)/lib -Wl$(comma)-rpath$(comma)$(ONECCL_INSTALL)/lib)
ONECCL_LDFLAGS   += $(if $(DPCPP_LIB),-L$(DPCPP_LIB) -Wl$(comma)-rpath$(comma)$(DPCPP_LIB))
ONECCL_LIBS      := $(ONECCL_CFLAGS) $(ONECCL_LDFLAGS) -lmpi -lccl $(COMMON_INC)

oneccl_collective ?= all
ifeq ($(oneccl_collective),all)
ONECCL_TARGETS := $(addprefix $(ONECCL_BUILD_DIR)/,$(COLLECTIVES))
else
ONECCL_TARGETS := $(addprefix $(ONECCL_BUILD_DIR)/,$(oneccl_collective))
endif

# ============== RCCL ==============
RCCL_SRC_DIR   := src/rccl
RCCL_BUILD_DIR := build/rccl
HIPCC          ?= hipcc
ROCM_PATH      ?= /opt/rocm
RCCL_CFLAGS    := -I$(ROCM_PATH)/include
RCCL_LDFLAGS   := -L$(ROCM_PATH)/lib -Wl$(comma)-rpath$(comma)$(ROCM_PATH)/lib
RCCL_LIBS      := $(RCCL_CFLAGS) $(RCCL_LDFLAGS) -Wno-unused-result -lmpi -lrccl $(COMMON_INC)

rccl_collective ?= all
ifeq ($(rccl_collective),all)
RCCL_TARGETS := $(addprefix $(RCCL_BUILD_DIR)/,$(COLLECTIVES))
else
RCCL_TARGETS := $(addprefix $(RCCL_BUILD_DIR)/,$(rccl_collective))
endif

# ============== Targets principali ==============
.PHONY: nccl oneccl rccl clean dirs

nccl: dirs $(NCCL_TARGETS) $(NCCL_BUILD_DIR)/init_time
oneccl: dirs $(ONECCL_TARGETS) $(ONECCL_BUILD_DIR)/init_time
rccl: dirs $(RCCL_TARGETS) $(RCCL_BUILD_DIR)/init_time

# ============== Creazione cartelle ==============
dirs:
	@mkdir -p $(NCCL_BUILD_DIR)
	@mkdir -p $(ONECCL_BUILD_DIR)
	@mkdir -p $(RCCL_BUILD_DIR)

# ============== Regole dinamiche NCCL ==============
$(foreach c,$(COLLECTIVES),\
  $(eval $(NCCL_BUILD_DIR)/$(c): $(NCCL_SRC_DIR)/$(c)/$(c).cu ; \
    $(NVCC) $(NCCL_LIBS) $$< -o $$@))

# ============== Regole dinamiche OneCCL ==============
$(foreach c,$(COLLECTIVES),\
  $(eval $(ONECCL_BUILD_DIR)/$(c): $(ONECCL_SRC_DIR)/$(c)/$(c).cpp ; \
    $(ONECCL_CXX) -std=c++17 -fsycl -fsycl-targets=$(SYCL_TARGET) $(ONECCL_EXTRA) $(ONECCL_LIBS) $$< -o $$@))

# ============== Regole dinamiche RCCL ==============
$(foreach c,$(COLLECTIVES),\
  $(eval $(RCCL_BUILD_DIR)/$(c): $(RCCL_SRC_DIR)/$(c)/$(c).cpp ; \
    $(HIPCC) $(RCCL_LIBS) $$< -o $$@))

# ============== init_time targets (one per library, standalone binary) ==============
$(NCCL_BUILD_DIR)/init_time: $(NCCL_SRC_DIR)/init_time/init_time.cu
	$(NVCC) $(NCCL_LIBS) $< -o $@

$(ONECCL_BUILD_DIR)/init_time: $(ONECCL_SRC_DIR)/init_time/init_time.cpp
	$(ONECCL_CXX) -std=c++17 -fsycl -fsycl-targets=$(SYCL_TARGET) $(ONECCL_EXTRA) $(ONECCL_LIBS) $< -o $@

$(RCCL_BUILD_DIR)/init_time: $(RCCL_SRC_DIR)/init_time/init_time.cpp
	$(HIPCC) $(RCCL_LIBS) $< -o $@

# ============== Clean ==============
clean:
	rm -rf build