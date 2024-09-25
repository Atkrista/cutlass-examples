# Compiler and flags
CXX = g++
NVCC = nvcc
CUDA_FLAGS =  -DCUTLASS_VERSIONS_GENERATED -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[compute_90a,sm_90a] -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 --expt-relaxed-constexpr -DCUTE_USE_PACKED_TUPLE=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -std=c++17

# Project dirs
BASEDIR = $(abspath ../../)
INCDIR = $(BASEDIR)/inc
BINDIR = $(BASEDIR)/bin

# Include directories
CUDA_INCLUDES = -I$(INCDIR) -I$(INCDIR)/cutlass/include -I$(INCDIR)/cutlass/tools/util/include

# Library directoriess
LIBDIRS = -L"$(CUDA_ROOT)/targets/sbsa-linux/lib/stubs"  -L"$(CUDA_ROOT)/targets/sbsa-linux/lib"

# Libraries to link
LIBS = -ldl -lcudart -lcuda -lcudadevrt

# Source files
SRCS = $(wildcard *.cu)
# Object files
OBJS = $(SRCS:.cu=.o)

# Executable name
TARGET = $(BINDIR)/$(notdir $(CURDIR)) 

# Ensure the bin directory exists
$(shell mkdir -p $(BINDIR))

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LIBS) $(LIBDIRS)

# Compile CUDA source files to object files
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c -o $@ $<

# Clean up
clean: 
	rm -f *.o $(TARGET)

# DEBUG
debug:
	exit

# Phony targets
.PHONY: all clean debug
