$(info CXX=$(CXX))

GOOGLE_BENCHMARK_DIR ?= /home/kcvaughn/Desktop/packages/benchmark

# -- Debug --

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -g
	CPPFLAGS += -DDEBUG
else
	CPPFLAGS += -DNDEBUG
endif
$(info DEBUG=$(DEBUG))

# -- Optimization level -- 

OPTIMIZATION_LEVEL ?= 3
CXXFLAGS += -O$(OPTIMIZATION_LEVEL)
$(info OPTIMIZATION_LEVEL=$(OPTIMIZATION_LEVEL))

# -- Architecture --

#ARCH ?= native
#CXXFLAGS += -march=$(ARCH)
#$(info ARCH=$(ARCH))

# -- Gmsh --

UM2_ENABLE_GMSH ?= 1
ifeq ($(UM2_ENABLE_GMSH), 1)
	CPPFLAGS += -DUM2_ENABLE_GMSH=1
	LDLIBS += -lgmsh
endif
$(info UM2_ENABLE_GMSH=$(UM2_ENABLE_GMSH))

# -- CUDA, OpenMP, and Thrust --

UM2_ENABLE_OMP ?= 0
ifeq ($(UM2_ENABLE_OMP), 1)
	CPPFLAGS += -fopenmp
endif
$(info UM2_ENABLE_OMP=$(UM2_ENABLE_OMP))

UM2_ENABLE_CUDA ?= 0
$(info UM2_ENABLE_CUDA=$(UM2_ENABLE_CUDA))

ifeq ($(UM2_ENABLE_CUDA), 1) 
	# Currently only supported via Clang++
	ifeq ($(notdir $(CXX)), clang++)
		CPPFLAGS += -DUM2_ENABLE_CUDA
		# Assumes hpc_sdk directory structure, meaning we need to add the math_libs
		# directory to the include path
		CUDA_ARCH ?= sm_86
		CUDA_PATH ?= /usr/local/cuda
		CUDA_INC_DIR ?= $(CUDA_PATH)/include
		CUDA_LIB_DIR ?= $(CUDA_PATH)/lib64
		CUDA_MATH_LIB_DIR ?= $(CUDA_PATH)/../math_libs/lib64
		CUDA_MATH_INC_DIR ?= $(CUDA_PATH)/../math_libs/include
		CUDA_LIBS := -lcudart_static -ldl -lrt -pthread
		CUDA_INC_FLAGS := $(addprefix -I,$(CUDA_INC_DIR) $(CUDA_MATH_INC_DIR))
		CUDA_LIB_FLAGS := $(addprefix -L,$(CUDA_LIB_DIR) $(CUDA_MATH_LIB_DIR))
		INC_FLAGS += $(CUDA_INC_FLAGS)
		# Unified CUDA and C++ source files, so we need to add "-x cuda" to the C++
		# compiler flags
		CXXFLAGS += -x cuda --cuda-gpu-arch=$(CUDA_ARCH) --cuda-path=$(CUDA_PATH)
		LDFLAGS += $(CUDA_LIB_FLAGS)
		LDLIBS += $(CUDA_LIBS)
		# Use the thrust included in the SDK
		THRUST_DIR ?= $(CUDA_INC_DIR)
	else
$(error CUDA is currently only supported for clang++)
	endif
# If anyone knows how to tab this or make it prettier, please do so :(
$(info CUDA_ARCH=$(CUDA_ARCH))
$(info CUDA_PATH=$(CUDA_PATH))
$(info CUDA_INC_DIR=$(CUDA_INC_DIR))
$(info CUDA_LIB_DIR=$(CUDA_LIB_DIR))
$(info CUDA_MATH_LIB_DIR=$(CUDA_MATH_LIB_DIR))
$(info CUDA_MATH_INC_DIR=$(CUDA_MATH_INC_DIR))
else
	# Use the thrust in /dependencies/
	THRUST_DIR ?= $(PWD)/../dependencies/thrust
	ifeq ($(UM2_ENABLE_OMP), 1)
		CPPFLAGS += -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
	else
	    # Serial
		CPPFLAGS += -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
	endif
endif # ifeq($(UM2_ENABLE_CUDA), 1)
INC_FLAGS += -isystem $(THRUST_DIR)
$(info THRUST_DIR=$(THRUST_DIR))

# Repo/build structure
# ------------------------------------------------------------------------------
DEPENDS_DIR := $(PWD)/../dependencies
BUILD_INC_DIR := $(PWD)/../build/include
BUILD_LIB_DIR := $(PWD)/../build/lib

SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
EXES := $(patsubst %.cpp, %, $(SRCS))

INC_FLAGS += -I$(BUILD_INC_DIR) -I$(GOOGLE_BENCHMARK_DIR)/include
LDFLAGS += -L$(BUILD_LIB_DIR) -L$(GOOGLE_BENCHMARK_DIR)/build/src

LDFLAGS += -Wl,--as-needed
LDLIBS += -lhdf5 -lhdf5_cpp -lbenchmark -lpthread

CPPFLAGS += -MMD -MP

# C++ standard and warnings
CXXFLAGS += -std=gnu++2b -Wall -Wextra -pedantic -Wformat=2
CXXFLAGS += -Wcast-align -Wcast-qual -Wdisabled-optimization
CXXFLAGS += -Wmissing-declarations -Wmissing-include-dirs
CXXFLAGS += -Wold-style-cast -Wredundant-decls -Wshadow -Wsign-conversion
CXXFLAGS += -Wswitch-default -Wundef -Wconversion
CXXFLAGS += -Wfloat-equal -Wswitch-enum -Walloca -Wnull-dereference
CXXFLAGS += -Wsuggest-override -Wdouble-promotion -Wzero-as-null-pointer-constant
CXXFLAGS += -Wno-pre-c++2b-compat -Wunused-parameter 
# CXXFLAGS += -Wpadded -Werror

# PugiXML
# ------------------------------------------------------------------------------
# Overwrite the pugixml config file with our own
INC_FLAGS += -isystem $(DEPENDS_DIR)/pugixml/src

# Targets 
# ------------------------------------------------------------------------------

all: $(EXES) 

$(EXES): %: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INC_FLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm $(EXES) *.d
