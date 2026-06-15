CXX = g++
CXXFLAGS = -Wall -O3 -fopenmp -std=c++17
FFTW_HOME = /sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/fftw-3.3.10-auz2xhp
GSL_HOME = /sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/gsl-2.8-zty4u3k
INC = -Iinclude -I$(FFTW_HOME)/include -I$(GSL_HOME)/include
LIB = -L$(FFTW_HOME)/lib -L$(GSL_HOME)/lib -lfftw3_threads -lfftw3 -lgslcblas -lgsl
BIN =
TARGET = ngenic_white_noise
default: $(TARGET)
all: $(TARGET)
ngenic_white_noise: main.cpp
	$(CXX) $(CXXFLAGS) $(INC) -o $(TARGET) $^ $(LIB) 
.PHONY: install
install:
	cp $(TARGET) $(BIN)
.PHONY: uninstall
uninstall:
	rm -f $(BIN)/$(TARGET)
.PHONY: clean
clean:
	rm -f $(TARGET)
