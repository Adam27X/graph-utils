REACH := reach

program_CXX_SRCS := $(wildcard *.cpp)
program_CXX_SRCS += $(wildcard ../../*.cpp)
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_INCLUDE_DIRS := . ../../
CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -g -O3 -std=c++0x -Wall -pedantic
LDFLAGS := -L/usr/lib #Boost

.PHONY: all clean distclean

all: $(REACH)

debug: CXXFLAGS = -g -O0 -std=c++0x -Wall -pedantic -DDEBUG $(EXTRA_FLAGS)
debug: $(REACH)

$(REACH): $(program_CXX_OBJS) 
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(program_CXX_OBJS) -o $(REACH)

clean:
	@- $(RM) $(REACH) $(program_CXX_OBJS) *~ 

distclean: clean
