/**
 * Implementation of a Dense (Tensorflow) or Fully Connected (PyTorch) network layer
 *
 * @author: Yvo Elling
 * @date: 10-03-23
 */

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <chrono>

#include <stdio.h>
#include "hw_data.h"

typedef uint8_t CoreIdx;

#define VECTOR_LENGTH 1'000'000
#define NROF_TEST_RUNS 100

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

__global__ void computeDenseLayerCUDA(float* weights, float* input, float* bias, float* output) {
    CoreIdx idx = threadIdx.x;
    
    float nodeOutputSum = 0.0f;
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        nodeOutputSum += input[idx] * weights[i] + bias[i];
    }
    output[idx] = nodeOutputSum;
}

void computeDenseLayerCPU(float* weights, float* input, float* bias, float* output) {
    float nodeOutputSum = 0.0f;
    auto t1 = high_resolution_clock::now();
    for (int idx = 0; idx < VECTOR_LENGTH; ++idx) {
        for (int i = 0; i < VECTOR_LENGTH; ++i) {
            nodeOutputSum += input[idx] * weights[i] + bias[i];
        }
        output[idx] = nodeOutputSum;
    }
    auto t2 = high_resolution_clock::now();
    int total_execution_time = duration_cast<milliseconds>(t2 - t1).count();
    std::cout << "Total execution time on CPU is: " << total_execution_time << " ms" << std::endl;
}

int main (int argc, char** argv) {
    std::cout << "Starting CUDA Application" << std::endl;
    std::cout << "Launching CUDA Program for Dense Layer" << std::endl;

    auto h_weights = (float *)calloc(VECTOR_LENGTH, sizeof(float));
    auto h_input = (float *)calloc(VECTOR_LENGTH, sizeof(float));
    auto h_bias = (float *)calloc(VECTOR_LENGTH, sizeof(float));
    auto h_output = (float *)calloc(VECTOR_LENGTH, sizeof(float));

    float *d_weights, *d_input, *d_bias, *d_output;

    cudaMalloc((void**)&d_weights, VECTOR_LENGTH * sizeof(float));
    cudaMalloc((void**)&d_input, VECTOR_LENGTH * sizeof(float));
    cudaMalloc((void**)&d_bias, VECTOR_LENGTH * sizeof(float));
    cudaMalloc((void**)&d_output, VECTOR_LENGTH * sizeof(float));

    cudaMemcpy(d_weights, h_weights, VECTOR_LENGTH*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, VECTOR_LENGTH*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, VECTOR_LENGTH*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, VECTOR_LENGTH*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::array<float, NROF_TEST_RUNS> execution_times;

    for (int i = 0; i < NROF_TEST_RUNS; ++i) {
        cudaEventRecord(start);
        computeDenseLayerCUDA<<<QUADRO_P2000_SM*3, QUADRO_P200_THREADS_PER_SM*3>>>(d_weights, d_input, d_bias, d_output);
        cudaEventRecord(stop);
    
        cudaDeviceSynchronize();
        cudaMemcpy(h_output, d_output, VECTOR_LENGTH*sizeof(float), cudaMemcpyDeviceToHost);
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        execution_times[i] = milliseconds;
    }

    float execution_time_sum = 0;
    for (int i = 0; i < NROF_TEST_RUNS; ++i) {
        execution_time_sum += execution_times[i];
    }
    float avg_execution_time = execution_time_sum / execution_times.size();
    std::cout << "Total average kernel execution time is: " << avg_execution_time << "ms" << std::endl;

    computeDenseLayerCPU(h_weights, h_input, h_bias, h_output);

    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);

    free(h_weights);
    free(h_input);
    free(h_bias);
    free(h_output);
}