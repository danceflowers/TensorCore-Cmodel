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


.PHONY: all clean test help

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) otc_run.log

test: $(TARGET)
	@./$(TARGET)

help:
	@echo "make / make test / make clean"
