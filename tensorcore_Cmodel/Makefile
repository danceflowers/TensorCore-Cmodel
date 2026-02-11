CXX       = g++
CXXFLAGS  = -std=c++17 -O2 -Wall -Wno-format-security
CXXFLAGS += -I.
LDFLAGS   = -lm

ifdef DEBUG
CXXFLAGS += -g -O0 -DOTC_DEBUG
endif

TARGET    = otc_simx
SRCS      = main.cpp pipeline.cpp otc_driver.cpp otc_fp.cpp otc_types.cpp otc_decode.cpp
OBJS      = $(SRCS:.cpp=.o)
HDRS      = pipeline.h otc_driver.h otc_fp.h otc_types.h otc_decode.h

# Comprehensive test binary
TEST_TARGET = otc_test
TEST_SRCS   = otc_test.cpp pipeline.cpp otc_driver.cpp otc_fp.cpp otc_types.cpp otc_decode.cpp
TEST_OBJS   = $(TEST_SRCS:.cpp=.o)

.PHONY: all clean test test-all test-comprehensive help

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(TEST_TARGET): $(TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_OBJS) $(LDFLAGS)

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(TEST_TARGET) $(OBJS) $(TEST_OBJS) otc_run.log

test: $(TARGET)
	@./$(TARGET) --test=ones --debug=1

test-simple: $(TARGET)
	@./$(TARGET) --test=simple --debug=2

test-identity: $(TARGET)
	@./$(TARGET) --test=identity --M=4 --K=4 --N=4 --debug=1

test-random: $(TARGET)
	@./$(TARGET) --test=random --debug=1

test-fp4: $(TARGET)
	@./$(TARGET) --test=ones --type_ab=fp4 --debug=1

test-fp16: $(TARGET)
	@./$(TARGET) --test=ones --type_ab=fp16 --debug=1

test-large: $(TARGET)
	@./$(TARGET) --test=ones --M=16 --K=16 --N=16 --debug=1

test-trace: $(TARGET)
	@./$(TARGET) --test=simple --debug=3 --trace

test-all: test-simple test-identity test test-random test-fp4 test-fp16 test-large

# Run comprehensive test suite (all types × all sizes × all patterns)
test-comprehensive: $(TEST_TARGET)
	@./$(TEST_TARGET)

help:
	@echo "make / make test / make test-all / make test-comprehensive / make clean"
