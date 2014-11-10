#environment setup
GCC=$(NDK_ROOT)/toolchains/arm-linux-androideabi-4.6/gen_standalone/linux-x86_64/bin/arm-linux-androideabi-g++
NVCC=$(CUDA_TOOLKIT_ROOT)/bin/nvcc -ccbin $(GCC) -target-cpu-arch=ARM -m32 -arch=sm_32 -O3 -Xptxas '-dlcm=ca' -target-os-variant=Android

INCLUDES+= /usr/local/include ./include $(CUDA_TOOLKIT_ROOT)/targets/armv7-linux-androideabi/include

CFLAGS += $(addprefix -I, $(INCLUDES)) -DGIT_VERSION=\"bla\"

-include src/subdir.mk

libmpflow.a: $(OBJS) $(CUDA_OBJS)
	$(NVCC) -lib -o "$@" $(OBJS) $(CUDA_OBJS)

clean:
	rm -rf *.a $(OBJS) $(CUDA_OBJS)
