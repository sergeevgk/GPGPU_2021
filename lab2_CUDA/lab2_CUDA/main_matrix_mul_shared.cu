#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <windows.h>
#include <stdio.h>
#include <time.h> 
#include <random>
#include <iostream>

static const size_t CUDA_BLOCK_SIZE = 32;
static const size_t MATRIX_SIZE = 1000; // square matrix
static const double MATRIX_VALUE_MIN = -10;
static const double MATRIX_VALUE_MAX = 10;

void mulCpuFunction(double* matrix_A, double* matrix_B, double* result, size_t matrix_size) {
    for (size_t i = 0; i < matrix_size; i++) {
        for (size_t j = 0; j < matrix_size; j++) {
            size_t current_index = i * matrix_size + j;
            result[current_index] = 0.0;

            for (size_t k = 0; k < matrix_size; k++) {
                result[current_index] += matrix_A[i * matrix_size + k] * matrix_B[k * matrix_size + j];
            }
        }
    }
}

double processMulCpu(double* matrix_A, double* matrix_B, double* result, size_t matrix_size) {
    clock_t start_time = clock();

    mulCpuFunction(matrix_A, matrix_B, result, matrix_size);

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    return elapsed_time;
}

__global__ void mulGpuKernel(double* a, double* b, double* result, size_t matrix_size) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= matrix_size || j >= matrix_size)
        return;

    size_t ind = i * matrix_size + j;
    result[ind] = 0;

    for (size_t k = 0; k < matrix_size; ++k) {
        result[ind] += a[i * matrix_size + k] * b[k * matrix_size + j];
    }
}

double processMulGpu(double* matrix_A, double* matrix_B, double* result, size_t matrix_size) {
    float elapsed_time;
    double* mat_A;
    double* mat_B;
    double* mat_res;
    size_t bytes_count = matrix_size * matrix_size * sizeof(double);
    
    cudaEvent_t start, end;
    dim3 cuda_threads(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    int cuda_blocks_count_x = (matrix_size + cuda_threads.x - 1) / cuda_threads.x;
    int cuda_blocks_count_y = (matrix_size + cuda_threads.y - 1) / cuda_threads.y;
    dim3 cuda_blocks(cuda_blocks_count_x, cuda_blocks_count_y);

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    cudaMalloc((void**)&mat_A, bytes_count);
    cudaMalloc((void**)&mat_B, bytes_count);
    cudaMalloc((void**)&mat_res, bytes_count);

    cudaMemcpy(mat_A, matrix_A, bytes_count, cudaMemcpyHostToDevice);
    cudaMemcpy(mat_B, matrix_B, bytes_count, cudaMemcpyHostToDevice);

    mulGpuKernel << <cuda_blocks, cuda_threads >> > (mat_A, mat_B, mat_res, matrix_size);

    cudaMemcpy(result, mat_res, bytes_count, cudaMemcpyDeviceToHost);

    cudaFree(mat_A);
    cudaFree(mat_B);
    cudaFree(mat_res);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return elapsed_time / 1000.0f;
}

double findResultsMaxDiff(double* matrix_A, double* matrix_B, size_t matrix_dim) {
    double result = 0.0;
    for (size_t i = 0; i < matrix_dim; i++) {
        result = std::max(result, std::fabs(matrix_A[i] - matrix_B[i]));
    }
    return result;
}

void clearResources(double* matrix_A, double* matrix_B, double* matrix_res_cpu, double* matrix_res_gpu) {
    delete[] matrix_A;
    delete[] matrix_B;
    delete[] matrix_res_cpu;
    delete[] matrix_res_gpu;
}


double* generate_random_matrix(size_t matrix_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(MATRIX_VALUE_MIN, MATRIX_VALUE_MAX);
    double* result = new double[matrix_dim];
    for (size_t i = 0; i < matrix_dim; i++) {
        result[i] = distrib(gen);
    }
    return result;
}

int main(int argc, char* argv[]) {
    size_t matrix_size = MATRIX_SIZE;
    size_t matrix_dim = matrix_size * matrix_size;

    double* matrix_A = generate_random_matrix(matrix_dim);
    double* matrix_B = generate_random_matrix(matrix_dim);
    double* matrix_res_cpu = new double[matrix_dim];
    double* matrix_res_gpu = new double[matrix_dim];

    float cpu_time = processMulCpu(matrix_A, matrix_B, matrix_res_cpu, matrix_size);
    float gpu_time = processMulGpu(matrix_A, matrix_B, matrix_res_gpu, matrix_size);

    double max_diff = findResultsMaxDiff(matrix_res_cpu, matrix_res_gpu, matrix_dim);

    printf("Matrix size: %d\n", matrix_size);
    printf("CPU execution time: %lf\n", cpu_time);
    printf("GPU execution time: %lf\n", gpu_time);
    printf("Max error: %lf\n", max_diff);
    printf("\n");

    clearResources(matrix_A, matrix_B, matrix_res_cpu, matrix_res_gpu);
    return 0;
}