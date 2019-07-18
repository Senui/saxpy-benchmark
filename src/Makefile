###################################################################
# Step 1: Configure your host compiler
###################################################################
# Linux/MacOS/Unix tools
CC = g++
CC_O = -o
CFLAGS = -std=c++11 -Ofast -march=native -mtune=native -funroll-loops
RM = rm -f


###################################################################
# Step 2: Select which implementations to enable
################################################################### 
TARGETS = saxpy_cpu \
	  saxpy_ocl1 \
	  saxpy_ocl1_hg \
#	  saxpy_ocl2 \


###################################################################
# Step 3: Configure additional settings for the SDKs
###################################################################

## OpenCL settings
OCL_CFLAGS = -I.. -DCL_HPP_ENABLE_EXCEPTIONS=1 -Wno-deprecated-declarations -I/tutorial/OpenCL-Headers
OS=$(shell uname)
ifeq ($(OS), Darwin)
	OCL_LDFLAGS = -framework OpenCL
else
	OCL_LDFLAGS = -lOpenCL
endif
OCL_LDFLAGS += -lstdc++

###################################################################
# Done. 
# Usually you don't need to change anything beyond this point
###################################################################

all: $(TARGETS)

saxpy_cpu: saxpy_cpu.cpp saxpy.h
	$(CC) $(CFLAGS) $(CC_O) $(@) $<

saxpy_ocl1: saxpy_ocl1.cpp saxpy.h
	$(CC) $(CFLAGS) $(OCL_CFLAGS) $(CC_O) $(@) $< $(OCL_LDFLAGS)

saxpy_ocl1_hg: saxpy_ocl1_hg.cpp saxpy.h
	$(CC) $(CFLAGS) $(OCL_CFLAGS) $(CC_O) $(@) $< $(OCL_LDFLAGS)
	
# saxpy_ocl2: saxpy_ocl2.cpp saxpy.h
# 	$(CC) $(CFLAGS) $(OCL_CFLAGS) $(CC_O) $(@) saxpy_ocl2.cpp $(OCL_LDFLAGS)

clean:
	$(RM) $(TARGETS) *.lib *.a *.exe *.obj *.o *.exp *.pyc

