# ============================================================================
# OpenTensorCore SimX — Makefile
# Mirrors: Vortex sim/simX/Makefile + ci/blackbox.sh test targets
# ============================================================================

CXX       = g++
CXXFLAGS  = -std=c++17 -O2 -Wall -Wno-format-security
CXXFLAGS += -I./include -I./sim/simx -I./runtime
LDFLAGS   = -lm

ifdef DEBUG
CXXFLAGS += -g -O0 -DOTC_DEBUG
endif

TARGET    = otc_simx
SRCS      = tests/main.cpp
HDRS      = include/otc_types.h include/otc_fp.h sim/simx/pipeline.h runtime/otc_driver.h

.PHONY: all clean test test-all help

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET) otc_run.log

# ==================== Test Targets (mirrors ci/blackbox.sh) ====================

test: $(TARGET)
	@echo "═══ Test: ones 8×8×8 FP8(E5M2) ═══"
	@./$(TARGET) --test=ones --debug=1

test-simple: $(TARGET)
	@echo "═══ Test: simple 2×2×2 (known result [19,22;43,50]) ═══"
	@./$(TARGET) --test=simple --debug=2

test-identity: $(TARGET)
	@echo "═══ Test: identity 4×4×4 ═══"
	@./$(TARGET) --test=identity --M=4 --K=4 --N=4 --debug=1

test-random: $(TARGET)
	@echo "═══ Test: random 8×8×8 ═══"
	@./$(TARGET) --test=random --debug=1

test-fp4: $(TARGET)
	@echo "═══ Test: FP4 input ═══"
	@./$(TARGET) --test=ones --type_ab=fp4 --debug=1

test-fp16: $(TARGET)
	@echo "═══ Test: FP16 input ═══"
	@./$(TARGET) --test=ones --type_ab=fp16 --debug=1

test-large: $(TARGET)
	@echo "═══ Test: 16×16×16 ═══"
	@./$(TARGET) --test=ones --M=16 --K=16 --N=16 --debug=1

test-trace: $(TARGET)
	@echo "═══ Test: 2×2×2 with full trace ═══"
	@./$(TARGET) --test=simple --debug=3 --trace
	@echo "Trace written to otc_run.log"

test-all: test-simple test-identity test test-random test-fp4 test-fp16 test-large
	@echo ""
	@echo "════════════════════════════════"
	@echo "  All tests completed"
	@echo "════════════════════════════════"

help:
	@echo "OpenTensorCore SimX Build System"
	@echo "================================"
	@echo "  make            Build simulator"
	@echo "  make DEBUG=1    Build with debug symbols"
	@echo "  make test       Run default 8x8x8 test"
	@echo "  make test-all   Run full regression suite"
	@echo "  make test-trace Generate full pipeline trace"
	@echo "  make clean      Remove build artifacts"
	@echo ""
	@echo "Runtime: ./otc_simx --help"
