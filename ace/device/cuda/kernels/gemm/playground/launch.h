#pragma once

void launch_sgemm_v1(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C);