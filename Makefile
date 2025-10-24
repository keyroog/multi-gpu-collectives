# Lista collettivi (uguali o diverse per ogni lib se vuoi)
COLLECTIVES := allgather allreduce alltoall broadcast gather reduce reduce_scatter scatter

# ============== NCCL ==============
NCCL_SRC_DIR := src/nccl
NCCL_BUILD_DIR := build/nccl
NCCL_LIBS := -lmpi -lnvidia-ml -lnccl

nccl_collective ?= all
ifeq ($(nccl_collective),all)
NCCL_TARGETS := $(addprefix $(NCCL_BUILD_DIR)/,$(COLLECTIVES))
else
NCCL_TARGETS := $(addprefix $(NCCL_BUILD_DIR)/,$(nccl_collective))
endif

# ============== OneCCL ==============
ONECCL_SRC_DIR := src/oneccl/coll
ONECCL_BUILD_DIR := build/oneccl
ONECCL_LIBS := -lmpi -lccl

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
    nvcc $(NCCL_LIBS) $$< -o $$@))

# ============== Regole dinamiche OneCCL ==============
$(foreach c,$(COLLECTIVES),\
  $(eval $(ONECCL_BUILD_DIR)/$(c): $(ONECCL_SRC_DIR)/$(c)/$(c).cpp ; \
    icpx -fsycl -O3 $(ONECCL_LIBS) $$< -o $$@))

# ============== Clean ==============
clean:
	rm -rf build