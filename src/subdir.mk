CUDA_SOURCES = src/fem/equation_kernel.cu
SOURCES = src/mathematics.cpp src/version.cpp

-include src/eit/subdir.mk
-include src/fem/subdir.mk
-include src/mwi/subdir.mk
-include src/numeric/subdir.mk

OBJS = $(SOURCES:.cpp=.o)
OBJS += $(CUDA_SOURCES:.cu=.o)

%.o: %.cu
	$(NVCC) $(CFLAGS) $(EXTRA_CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(GCC) -std=gnu++11 $(CFLAGS) $(EXTRA_CFLAGS) -c -o $@ $<
