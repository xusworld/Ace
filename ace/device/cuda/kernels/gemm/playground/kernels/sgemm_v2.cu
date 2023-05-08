#include<stdio.h>
#include<stdlib.h>

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa(i,j) sa[((i)<<5) + (j)]
#define sb(i,j) sb[((i)<<5) + (j)]

#define MS 32
#define NS 32
#define KS 32

// cache blocking version, without register-level data re-use
__global__  __launch_bounds__(1024)
void mysgemm_v2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M;
    int ldb = K;
    int ldc = M;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 调整行
    A = &A((bx<<5),0);
    // 调整列
    B = &B(0,(by<<5));
    // 调整行列位置
    C = &C((bx<<5),(by<<5));

    // 使用共享内存
    __shared__ float sa[MS*KS];
    __shared__ float sb[KS*NS];

    float tmp=0.;
    for (int k_count = 0; k_count < K; k_count += KS){
        // 将A B的元素读入到共享内存中
        sa(tx,ty)=A(tx,ty);
        sb(ty,tx)=B(tx,ty);
        // A向右移动32行
        A+=(lda<<5);
        // B向下移动32列
        B+=32;
        // 等待同一个 block 内，所有的线程读取完数据
        __syncthreads();
        for (int inner_k_count=0; inner_k_count < KS; inner_k_count++) {
            tmp += sa(tx,inner_k_count) * sb(ty,inner_k_count);
        }
        __syncthreads();
    }
    C(tx,ty) = alpha * tmp + beta*C(tx,ty);
}

/*


void test_mysgemm_v2(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    dim3 blockDim(32,32);
    mysgemm_v2<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

*/