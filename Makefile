# Makefile per compilazione di collettive GPU con supporto variabile di libreria e nome della collettiva

# Specificare la libreria (nccl o oneccl)
LIB ?= nccl
# Specificare la collettiva da compilare oppure "all"
COLLECTIVE ?= all

# Directory sorgenti e destinazione
SRCDIR := src/$(LIB)
BUILD_DIR := build/$(LIB)
EXT := cu

# Selezione dei file sorgente in base a COLLECTIVE
ifeq ($(COLLECTIVE),all)
	SRCS := $(wildcard $(SRCDIR)/*/*.${EXT})
else
	SRCS := $(SRCDIR)/$(COLLECTIVE)/$(COLLECTIVE).${EXT}
endif

# Ricavo dei nomi delle collettive e dei binari
COLLECTIVES := $(patsubst $(SRCDIR)/%/%.${EXT},%,$(SRCS))
BINS := $(addprefix $(BUILD_DIR)/,$(COLLECTIVES))

.PHONY: all clean dirs

# Target di default
all: dirs $(BINS)

# Creazione cartella di destinazione
dirs:
	@mkdir -p $(BUILD_DIR)

# Regola di compilazione generica
$(BUILD_DIR)/%: $(SRCDIR)/%/%.${EXT}
	nvcc -lmpi -lnvidia-ml -l$(LIB) $< -o $@

# Pulizia dei file generati
clean:
	rm -rf build
