BC := bc

program_CXX_SRCS := $(wildcard *.cpp)
program_CXX_SRCS += $(wildcard ../../*.cpp)
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_CU_SRCS := $(wildcard *.cu)
program_CU_SRCS += $(wildcard ../../graph-utils/multi_search/*.cu)
program_CU_SRCS += $(wildcard ../../graph-utils/multi_search/betweenness_centrality/*.cu)
program_CU_SRCS += $(wildcard ../../*.cu)
program_CU_SRCS += $(wildcard ../../graph-utils/*.cu)
program_CU_HEADERS := $(wildcard ../../graph-utils/multi_search/*.cuh)
program_CU_HEADERS += $(wildcard ../../graph-utils/multi_search/betweenness_centrality/*.cuh)
program_CU_HEADERS += $(wildcard ../../*.cuh)
program_CU_HEADERS += $(wildcard ../../graph-utils/*.cuh)
program_CU_OBJS := ${program_CU_SRCS:.cu=.cuo}

program_INCLUDE_DIRS := . ../../ ../../graph-utils/multi_search/ ../../graph-utils/multi_search/betweenness_centrality/ /usr/local/cuda-7.0/include/ ../../graph-utils
program_CU_INCLUDE_DIRS := /home/users/amclaugh/CUB/cub-1.3.2/
CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -g -O3 -std=c++11 -Wall -pedantic

GEN_SM35 := -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GEN_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
NVFLAGS := -O3 -rdc=true -std=c++11
NVFLAGS += $(GEN_SM35)
NVFLAGS += $(foreach includedir,$(program_CU_INCLUDE_DIRS),-I$(includedir))

CUO_O_OBJECTS := ${program_CU_OBJS:.cuo=.cuo.o}

OBJECTS = $(program_CU_OBJS) $(program_CXX_OBJS)

.PHONY: all clean distclean

all: $(BC) 

verbose: $(BC)
verbose: NVFLAGS += --ptxas-options="-v"

debug: CXXFLAGS = -g -O0 -std=c++11 -Wall -pedantic -DDEBUG $(EXTRA_FLAGS)
debug: NVFLAGS = -O0 $(GEN_SM35) -g -G -std=c++11
debug: NVFLAGS += $(foreach includedir,$(program_CU_INCLUDE_DIRS),-I$(includedir))
debug: $(BC)

# Rule for compilation of CUDA source (C++ source can be handled automatically)
%.cuo: %.cu 
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ -dc $<

$(BC): $(OBJECTS) 
	@ for cu_obj in $(program_CU_OBJS); \
	do				\
		mv $$cu_obj $$cu_obj.o; \
	done				#append a .o suffix for nvcc
	nvcc $(NVFLAGS) $(CPPFLAGS) -o $@ $(program_CXX_OBJS) $(CUO_O_OBJECTS)
	@ for cu_obj in $(CUO_O_OBJECTS); 	\
	do					\
		mv $$cu_obj $${cu_obj%.*};	\
	done				#remove the .o for make

clean:
	@- $(RM) $(BC) $(OBJECTS) *~ 

distclean: clean
