SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
EXES := $(patsubst %.cpp, %, $(SRCS))

CPPFLAGS += -DUM2_ENABLE_GMSH=1 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
CXXFLAGS += -std=gnu++2b -Wall -Wextra -pedantic
LDLIBS += -lum2 -lhdf5 -lhdf5_cpp
INCFLAGS += -isystem $(PWD)/../../dependencies/thrust
INCFLAGS += -isystem $(PWD)/../../dependencies/pugixml/src

all: $(EXES) 

$(EXES): %: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm $(EXES) *.brep *.info *.h5 *.inp *.xdmf
