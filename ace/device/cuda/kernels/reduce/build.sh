nvcc -ccbin g++ -I../utils -m64 --std=c++11 -gencode arch=compute_75,code=compute_75 -o reduce_kernel.o -c reduce_kernel.cu
nvcc -ccbin g++ -I../utils -m64 --std=c++11 -gencode arch=compute_75,code=compute_75 -o test_reduce.o -c test_reduce.cc
nvcc -ccbin g++ -I../utils -m64 -gencode arch=compute_75,code=compute_75 -o test_reduce test_reduce.o reduce_kernel.o
