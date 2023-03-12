/**
 * Implementation of a Simple Matrix Multiplication kernel using CUDA
 *
 * @author: Yvo Elling
 * @date: 10-03-23
 */

#include <stdio.h>

#include <iostream>
#include <cstdint>
#include <chrono>
#include <array>

#define NROF_TEST_RUNS 500
#define MATRIX_WIDTH 8192
#define MATRIX_HEIGHT 8192
#define MATRIX_SIZE MATRIX_WIDTH * MATRIX_HEIGHT
#define BLOCK_DIM 1
#define PLACEHOLDER_NONE 1

typedef uint8_t RowIdx;
typedef uint8_t ColumnIdx;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

template <class T> 
__global__ void matrixMultiplicationCUDA(T* a, T* b, T* c) {
    RowIdx row = blockIdx.y * blockDim.y + threadIdx.y;
    ColumnIdx col = blockIdx.x * blockDim.x + threadIdx.x;

    auto columnRowSum = 0.0f;
    for (int i = 0; i < MATRIX_HEIGHT; ++i) {
        columnRowSum += b[row * MATRIX_WIDTH + i] * a[i * MATRIX_HEIGHT + col];
    }
    c[row * MATRIX_WIDTH + col] = columnRowSum;
}

template <class T>
void matrixMultiplicationCPU(T* a, T* b, T* c) {
    auto t1 = high_resolution_clock::now();
    for (int i = 0; i < MATRIX_HEIGHT; ++i) {
        float rowColumnSum = 0.0f;
        for (int j = 0; j < MATRIX_WIDTH; ++j) {
            rowColumnSum += b[i * MATRIX_HEIGHT + j] * a[j * MATRIX_WIDTH + i];
        }
        c[i * MATRIX_WIDTH + MATRIX_HEIGHT] = rowColumnSum;
    }
    auto t2 = high_resolution_clock::now();
    int total_execution_time = duration_cast<milliseconds>(t2 - t1).count();
    std::cout << "Total execution time on CPU is: " << total_execution_time << " ms" << std::endl;
}

int main (int argc, char** argv) {
    std::cout << "Starting CUDA Application" << std::endl;
    std::cout << "Launching CUDA Program for Matrix Multiplication" << std::endl;

    int32_t * h_a = (int32_t *)calloc(MATRIX_SIZE, sizeof(int32_t));
    int32_t * h_b = (int32_t *)calloc(MATRIX_SIZE, sizeof(int32_t));
    int32_t * h_c = (int32_t *)calloc(MATRIX_SIZE, sizeof(int32_t));

    int32_t *d_a, *d_b, *d_c; 

    cudaMalloc((void**)&d_a, MATRIX_SIZE*sizeof(int32_t));
    cudaMalloc((void**)&d_b, MATRIX_SIZE*sizeof(int32_t));
    cudaMalloc((void**)&d_c, MATRIX_SIZE*sizeof(int32_t));

    cudaMemcpy(d_a, h_a, MATRIX_SIZE*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, MATRIX_SIZE*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, MATRIX_SIZE*sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blocksPerGrid(BLOCK_DIM, PLACEHOLDER_NONE, PLACEHOLDER_NONE);
    dim3 threadsPerBlock(MATRIX_WIDTH, MATRIX_HEIGHT, PLACEHOLDER_NONE);

    std::array<float, NROF_TEST_RUNS> execution_times;

    for (int i = 0; i < NROF_TEST_RUNS; ++i) {
        cudaEventRecord(start);
        matrixMultiplicationCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
        cudaEventRecord(stop);

        cudaDeviceSynchronize();
        cudaMemcpy(d_c, h_c, MATRIX_SIZE*sizeof(int32_t), cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        execution_times[i] = milliseconds;
    }

    float execution_time_sum = 0;
    for (int i = 0; i < NROF_TEST_RUNS; ++i) {
        execution_time_sum += execution_times[i];
    }
    float avg_execution_time = execution_time_sum / execution_times.size();
    std::cout << "Total average kernel execution time is: " << avg_execution_time << "ms" << std::endl;


    matrixMultiplicationCPU(h_a, h_b, h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
}