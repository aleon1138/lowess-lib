CXXFLAGS += -O3 -Wall -std=c++17 -fopenmp -march=native

CPPFLAGS += $(shell python3 -m pybind11 --includes)
CPPFLAGS += -DNDEBUG

lowesslib.so: lowess.cc lowesslib.cc
	g++  $(CXXFLAGS) $(CPPFLAGS) -shared -fPIC -o $@ $^

format:
	astyle -A4 -S -z2 -n -j *.cc

clean:
	rm -f lowesslib.so
