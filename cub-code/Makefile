CUB=cub-1.8.0
CFLAGS= -g
all: cub-sort

cub-sort: sorting_test.cu helper.cu.h
	nvcc -I$(CUB)/cub $(CFLAGS) -o test-cub sorting_test.cu
	./test-cub 4096

cub-compile: sorting_test.cu helper.cu.h
	nvcc -I$(CUB)/cub $(CFLAGS) -o test-cub sorting_test.cu

clean:
	rm -f test-cub

