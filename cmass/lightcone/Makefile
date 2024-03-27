CC = gcc
CXX = g++

FLAGS = -Wall -Wextra -O3 -ggdb
CFLAGS = --std=gnu99 $(FLAGS)
CXXFLAGS = --std=c++17 -fopenmp $(FLAGS)

LFLAGS = -lm -lgsl -lgslcblas -lcmangle -lhealpixlite -fopenmp

pymangle_src = ./pymangle/pymangle
pymangle = libcmangle.a
healpixlite_src = ./healpix_lite
healpixlite = libhealpixlite.a
cuboidremap_src = cuboidremap/c++
main = lightcone

$(main): lightcone.cpp $(pymangle) $(healpixlite)
	$(CXX) $(CXXFLAGS) -I$(pymangle_src) -I$(healpixlite_src) -I$(cuboidremap_src) -DHAVE_INLINE -o $(main) lightcone.cpp $(LFAGS)


$(pymangle): $(patsubst $(pymangle_src)/%.c,$(pymangle_src)/%.o,$(wildcard $(pymangle_src)/*.c))
	ar -r $@ $^

$(healpixlite): $(patsubst $(healpixlite_src)/%.cc,$(healpixlite_src)/%.o,$(wildcard $(healpixlite_src)/*.cc))
	ar -r $@ $^

$(pymangle_src)/%.o: $(pymangle_src)/%.c
	$(CC) -c -I$(pymangle_src) $(CFLAGS) -o $@ $<

$(healpixlite_src)/%.o: $(healpixlite_src)/%.cc
	$(CXX) -c -I$(healpixlite_src) $(CXXFLAGS) -o $@ $<

all: $(main)

.PHONY: clean
clean:
	rm -f $(pymangle_src)/*.o
	rm -f $(healpixlite_src)/*.o
	rm -f $(main)
