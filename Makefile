# --- configurazione di default ---
LIBRARY    ?= nccl
COLLECTIVE ?= all

# directory di build
BUILD_DIR := build/$(LIBRARY)

# elenco dei sorgenti (.cu) sotto la cartella della library
SRCS       := $(wildcard $(LIBRARY)/*.cu)
ALL_COLS   := $(basename $(notdir $(SRCS)))

# se voglio una sola collective, la uso, altrimenti tutte
ifeq ($(COLLECTIVE),all)
  TARGETS := $(ALL_COLS)
else
  TARGETS := $(COLLECTIVE)
endif

# flags specifiche per library
NCFLAGS    := -lmpi -lnvidia-ml -lnccl
ONEFLAGS   := -lmpi -lnvidia-ml -loneccl
LIB_FLAGS  := $(if $(filter nccl,$(LIBRARY)),$(NCFLAGS),$(ONEFLAGS))

.PHONY: all clean
all: $(BUILD_DIR) $(TARGETS:%=$(BUILD_DIR)/%)

# creazione cartella build se non esiste
$(BUILD_DIR):
    mkdir -p $@

# regola pattern per compilare ciascuna collective
$(BUILD_DIR)/%: $(LIBRARY)/%.cu
    nvcc $(LIB_FLAGS) $< -o $@

clean:
    rm -rf build