# ==============================================================================
# OpenTensorCore Cycle-Accurate Simulator — Makefile
# ==============================================================================
#
# Usage:
#   make              — Build the simulator (release)
#   make debug        — Build with debug info and sanitizers
#   make test         — Build and run all tests (all precisions)
#   make test PREC=FP8_E4M3        — Run all tests, restrict to one precision
#   make test TEST=3                — Run only test 3 (all precisions)
#   make test PREC=FP16 TEST=1     — Run test 1 with FP16 only
#   make stress       — Run stress test (test 3) across all precisions
#   make viz          — Run pipeline visualization (test 4)
#   make clean        — Remove build artifacts
#   make help         — Show this help
#
# Precision values: FP4_E2M1 | FP8_E4M3 | FP8_E5M2 | FP16 | ALL (default)
# Test IDs:
#   1 = Single matmul per precision
#   2 = Back-to-back pipelined matmuls
#   3 = Stress test (20 random matrices/precision)
#   4 = Pipeline stage visualization
#   5 = Output format conversion
#   6 = Edge cases (identity, zero)
#   ALL = Run all tests (default)
# ==============================================================================

# Compiler and flags
CXX       := g++
CXXFLAGS  := -std=c++17 -Wall -Wextra -Wpedantic
OPTFLAGS  := -O2
DBGFLAGS  := -g -O0 -fsanitize=address,undefined -DDEBUG

# Target
TARGET    := tensorcore_sim
SRCS      := main.cpp main/main.cpp test/test.cpp otc_driver/otc_driver.cpp pipeline/pipeline.cpp dot_product/dot_product.cpp pre_conv/pre_conv.cpp tensor_core_cfg.cpp
HDRS      := fp_types.h fp_arith.h tensor_core_sim.h tensor_core_cfg.h main/main.h test/test.h otc_driver/otc_driver.h pipeline/pipeline.h dot_product/dot_product.h pre_conv/pre_conv.h

# Configurable parameters (override on command line)
PREC      ?= ALL
TEST      ?= ALL
RM_MODE   ?= RNE
SEED      ?= 0

# Build CLI arguments from parameters
RUN_ARGS  :=
ifneq ($(PREC),ALL)
    RUN_ARGS += --prec $(PREC)
endif
ifneq ($(TEST),ALL)
    RUN_ARGS += --test $(TEST)
endif
ifneq ($(RM_MODE),RNE)
    RUN_ARGS += --rm $(RM_MODE)
endif
ifneq ($(SEED),0)
    RUN_ARGS += --seed $(SEED)
endif

# ==============================================================================
# Targets
# ==============================================================================

.PHONY: all debug test stress viz clean help

# Default: release build
all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ $(SRCS)
	@echo ""
	@echo "  Build complete: ./$(TARGET)"
	@echo "  Run './$(TARGET) --help' for usage information."
	@echo ""

# Debug build with sanitizers
debug: OPTFLAGS = $(DBGFLAGS)
debug: $(TARGET)

# Run tests
test: $(TARGET)
	@echo ""
	./$(TARGET) $(RUN_ARGS)

# Shortcut: stress test only
stress: $(TARGET)
	./$(TARGET) --test 3 $(if $(filter-out ALL,$(PREC)),--prec $(PREC),)

# Shortcut: pipeline visualization
viz: $(TARGET)
	./$(TARGET) --test 4

# Clean
clean:
	rm -f $(TARGET)

# Help
help:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║  OpenTensorCore Simulator — Build System                ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Build targets:"
	@echo "    make              Build release binary"
	@echo "    make debug        Build with debug info + sanitizers"
	@echo "    make clean        Remove build artifacts"
	@echo ""
	@echo "  Test targets:"
	@echo "    make test                    Run all tests, all precisions"
	@echo "    make test PREC=FP8_E4M3      Restrict to one precision"
	@echo "    make test TEST=3             Run only test 3"
	@echo "    make test PREC=FP16 TEST=1   Combine precision + test filter"
	@echo "    make stress                  Run stress test (test 3)"
	@echo "    make stress PREC=FP16        Stress test, FP16 only"
	@echo "    make viz                     Pipeline visualization (test 4)"
	@echo ""
	@echo "  Parameters:"
	@echo "    PREC    Precision filter: FP4_E2M1 | FP8_E4M3 | FP8_E5M2 | FP16 | ALL"
	@echo "    TEST    Test ID: 1-6 | ALL"
	@echo "    RM_MODE Rounding mode: RNE | RTZ | RDN | RUP | RMM"
	@echo "    SEED    RNG seed (0 = use time)"
	@echo ""
