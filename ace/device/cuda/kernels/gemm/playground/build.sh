nvcc -ccbin g++ -I../../utils -m64 --std=c++11 -gencode arch=compute_75,code=compute_75 -o matrix.o -c matrix.cc
nvcc -ccbin g++ -I../../utils -m64 --std=c++11 -gencode arch=compute_75,code=compute_75 -o launch.o -c launch.cc
nvcc -ccbin g++ -I../../utils -m64 --std=c++11 -gencode arch=compute_75,code=compute_75 -o sgemm_bench.o -c sgemm_bench.cc
nvcc -ccbin g++ -I../../utils -m64 --std=c++11 -gencode arch=compute_75,code=compute_75 -o kernels/sgemm_v1.o -c kernels/sgemm_v1.cu
nvcc -ccbin g++ -I../../utils  -L/usr/local/cuda/lib64 -lcudart -lcublas -m64 -gencode arch=compute_75,code=compute_75 -o sgemm_bench sgemm_bench.o kernels/sgemm_v1.o matrix.o launch.o
