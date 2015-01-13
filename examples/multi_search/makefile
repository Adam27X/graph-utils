#TODO: Objects directory, automated detection and separable compilation of .cu and .cpp files

MULTI_SEARCH := multi_search

program_CXX_SRCS := $(wildcard *.cpp)
program_CXX_SRCS += $(wildcard ../../*.cpp)
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_CU_SRCS := $(wildcard *.cu)
program_CU_SRCS += $(wildcard ../../graph-utils/multi_search/*.cu)
program_CU_SRCS += $(wildcard ../../*.cu)
program_CU_SRCS += $(wildcard ../../graph-utils/*.cu)
program_CU_HEADERS := $(wildcard ../../graph-utils/multi_search/*.cuh)
program_CU_HEADERS += $(wildcard ../../*.cuh)
program_CU_HEADERS += $(wildcard ../../*.cuh)
program_CU_OBJS := ${program_CU_SRCS:.cu=.cu.o}

program_INCLUDE_DIRS := . ../../ ../../graph-utils/multi_search/ /usr/local/cuda-6.0/include/ ../../graph-utils
program_CU_INCLUDE_DIRS := /home/users/amclaugh/CUB/cub-1.3.2/
CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -g -O3 -std=c++0x -Wall -pedantic

GEN_SM35 := -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GEN_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
NVFLAGS := -O3 -rdc=true
NVFLAGS += $(GEN_SM35)
NVFLAGS += $(foreach includedir,$(program_CU_INCLUDE_DIRS),-I$(includedir))

OBJECTS := common.cu.o CTA_warp.cu.o edge_parallel.cu.o linear_with_atomics.cu.o scan_based.cu.o shuffle_based.cu.o warp_based.cu.o race_and_resolve.cu.o util_device.cu.o $(program_CXX_OBJS)

.PHONY: all clean distclean

all: $(MULTI_SEARCH) 

debug: CXXFLAGS = -g -O0 -std=c++0x -Wall -pedantic -DDEBUG $(EXTRA_FLAGS)
debug: NVFLAGS = -O0 $(GEN_SM35) -g -G
debug: NVFLAGS += $(foreach includedir,$(program_CU_INCLUDE_DIRS),-I$(includedir))
debug: $(MULTI_SEARCH)

#$(MULTI_SEARCH): $(program_CXX_OBJS) $(program_CU_SRCS) $(program_HEADERS) $(program_CU_HEADERS)
	#nvcc $(NVFLAGS) $(CPPFLAGS) $(LDFLAGS) $(program_CU_SRCS) $(program_CXX_OBJS) -o $(MULTI_SEARCH)

common.cu.o: ../../graph-utils/multi_search/common.cu
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

CTA_warp.cu.o: ../../graph-utils/multi_search/CTA_warp.cu
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

edge_parallel.cu.o: ../../graph-utils/multi_search/edge_parallel.cu
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

linear_with_atomics.cu.o: ../../graph-utils/multi_search/linear_with_atomics.cu
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

scan_based.cu.o: ../../graph-utils/multi_search/scan_based.cu
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

shuffle_based.cu.o: ../../graph-utils/multi_search/shuffle_based.cu
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

warp_based.cu.o: ../../graph-utils/multi_search/warp_based.cu
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

race_and_resolve.cu.o: ../../graph-utils/race_and_resolve.cu
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

util_device.cu.o: ../../util_device.cu
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

$(MULTI_SEARCH): $(OBJECTS)
	echo $(program_CXX_OBJS)
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ $+ 

clean:
	@- $(RM) $(MULTI_SEARCH) $(OBJECTS) *~ 

distclean: clean
