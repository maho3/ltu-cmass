CXX = g++
CXXFLAGS = -Wall -O3 -fopenmp -std=c++17
LIB = -lfftw3_omp -lfftw3 -lgslcblas -lgsl
INC = -Iinclude
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
