CXXFLAGS += -O3 -Wall -std=c++17 -fopenmp -mavx2 -mfma -fPIC

CPPFLAGS += $(shell python3 -m pybind11 --includes)
CPPFLAGS += $(shell python3-config --includes)
CPPFLAGS += -DNDEBUG

# Sources and objects
SRCS = lowess.cc lowesslib.cc expectile.cc
OBJS = $(SRCS:.cc=.o)

lowesslib.so: $(OBJS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -shared -o $@ $(OBJS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

test: unit_test
	./unit_test

unit_test: test_nelder_mead.cc
	$(CXX) $(CXXFLAGS) -o $@ $< -lgtest -lgtest_main

format:
	astyle -A4 -S -z2 -n -j *.cc inc/*.h

clean:
	rm -rf build dist lowesslib.egg-info lowesslib.so $(OBJS) unit_test
