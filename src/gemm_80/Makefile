
# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17 --generate-code=arch=compute_90,code=[sm_90a,compute_90]

# Project dirs
BASEDIR = $(abspath ../../)
INCDIR = $(BASEDIR)/inc
BINDIR = $(BASEDIR)/bin

# Include directories
INCLUDES = -I$(INCDIR) -I$(INCDIR)/cutlass/include -I$(INCDIR)/cutlass/tools/util/include

# Library directoriess
LIBDIRS = -L$(CUDA_ROOT)/lib64

# Libraries to link
LIBS = -lcudart

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
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIBDIRS) -o $@ $^ $(LIBS)

# Compile CUDA source files to object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

# Clean up
clean: 
	rm -f *.o $(TARGET)

# DEBUG
debug:
	@echo "$(PWD)"
	@echo "$(basename $(PWD))"
	exit

# Phony targets
.PHONY: all clean debug

