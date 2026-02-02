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
# SYCL_TARGET    - fsycl-targets value (e.g. nvidia_gpu_sm80, nvptx64-nvidia-cuda)
ONECCL_CXX  ?= $(or $(DPCPP_CLANGXX),icpx)
SYCL_TARGET ?= nvptx64-nvidia-cuda

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

# ============== Targets principali ==============
.PHONY: nccl oneccl clean dirs

nccl: dirs $(NCCL_TARGETS)
oneccl: dirs $(ONECCL_TARGETS)

# ============== Creazione cartelle ==============
dirs:
	@mkdir -p $(NCCL_BUILD_DIR)
	@mkdir -p $(ONECCL_BUILD_DIR)

# ============== Regole dinamiche NCCL ==============
$(foreach c,$(COLLECTIVES),\
  $(eval $(NCCL_BUILD_DIR)/$(c): $(NCCL_SRC_DIR)/$(c)/$(c).cu ; \
    $(NVCC) $(NCCL_LIBS) $$< -o $$@))

# ============== Regole dinamiche OneCCL ==============
$(foreach c,$(COLLECTIVES),\
  $(eval $(ONECCL_BUILD_DIR)/$(c): $(ONECCL_SRC_DIR)/$(c)/$(c).cpp ; \
    $(ONECCL_CXX) -std=c++17 -fsycl -fsycl-targets=$(SYCL_TARGET) $(ONECCL_LIBS) $$< -o $$@))

# ============== Clean ==============
clean:
	rm -rf build