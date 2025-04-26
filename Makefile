INC_DIR := inc
SRC_DIR := src
BIN_DIR := bin

NVCC := /usr/local/cuda/bin/nvcc
CXX := g++

NVCCFLAGS := -ccbin $(CXX)
CXXFLAGS := -std=c++17 -Xcompiler -Wall
INCLUDEFLAGS := -I $(INC_DIR)
LDFLAGS := -lfreeimage


DEP := $(INC_DIR)/SimilarityFinder.h $(INC_DIR)/GpuSimilarityFinder.h $(INC_DIR)/SimilarityFinderOptions.h $(INC_DIR)/ImageIO.h
SRC := $(SRC_DIR)/SimilarityFinder.cpp $(SRC_DIR)/GpuSimilarityFinder.cu $(SRC_DIR)/SimilarityFinderOptions.cpp $(SRC_DIR)/ImageIO.cpp $(SRC_DIR)/compress.cpp
TARGET := $(BIN_DIR)/compress

.PHONY: help build clean install

help:
	@echo "Available make commands:"
	@echo "  make build  - Build the project."
	@echo "  make clean  - Clean up the build files."
	@echo "  make install- Install the project (not applicable here)."
	@echo "  make help   - Display this help message."

build: $(TARGET)

$(TARGET): $(SRC) $(DEP)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDEFLAGS) $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)/*

install:
	@echo "No installation is required."
