MULTI_SEARCH := multi_search

program_CXX_SRCS := $(wildcard *.cpp)
program_CXX_SRCS += $(wildcard ../../*.cpp)
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_CU_SRCS := $(wildcard *.cu)
program_CU_SRCS += $(wildcard ../../graph-utils/multi_search/*.cu)
program_CU_OBJS := ${program_CU_SRCS:.cu=.cu.o}

program_INCLUDE_DIRS := . ../../ ../../graph-utils/multi_search/ /usr/local/cuda-6.0/include/
CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -g -O3 -std=c++0x -Wall -pedantic

GEN_SM35 := -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GEN_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
NVFLAGS := -O3 
NVFLAGS += $(GEN_SM35)

.PHONY: all clean distclean

all: $(MULTI_SEARCH)

debug: CXXFLAGS = -g -O0 -std=c++0x -Wall -pedantic -DDEBUG $(EXTRA_FLAGS)
debug: $(MULTI_SEARCH)

$(MULTI_SEARCH): $(program_CXX_OBJS) $(program_CU_SRCS) $(program_HEADERS) $(program_CU_HEADERS)
	nvcc $(NVFLAGS) $(CPPFLAGS) $(LDFLAGS) $(program_CU_SRCS) $(program_CXX_OBJS) -o $(MULTI_SEARCH)

clean:
	@- $(RM) $(MULTI_SEARCH) $(program_CXX_OBJS) *~ 

distclean: clean
