# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -O3 -std=c++17
NVCCFLAGS = -O3 -std=c++17 --generate-code=arch=compute_90,code=sm_90a

# Include directories
INCLUDES = -I../src/in -I../src/inc/cutlass/include -I../src/inc/cutlass/tools/util/include

# Library directoriess
LIBDIRS = -L$(CUDA_ROOT)/lib64

# Libraries to link
LIBS = -lcudart

# Source files
SRCS = $(wildcard *.cu)
# Object files
OBJS = $(SRCS:.cu=.o)

# Output directory
BINDIR = ../bin

# Executable name
TARGET = $(BINDIR)/test

# Ensure the bin directory exists
$(shell mkdir -p $(BINDIR))

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIBDIRS) -o $@ $^ $(LIBS)

# Compile CUDA source files to object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

# Clean up
clean:
	rm -f *.o $(TARGET)

# Phony targets
.PHONY: all clean

