

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

__global__ void mulSharedGpuKernel(double* a, double* b, double* result, size_t matrix_size)
{
    double result_value = 0;

    size_t row_index = blockIdx.y * CUDA_BLOCK_SIZE + threadIdx.y;
    size_t column_index = blockIdx.x * CUDA_BLOCK_SIZE + threadIdx.x;
    size_t index_a_j;
    size_t index_b_k;
    __shared__ double matrix_a_shared[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
    __shared__ double matrix_b_shared[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

    for (int current_idx = 0; current_idx < (CUDA_BLOCK_SIZE + matrix_size - 1) / CUDA_BLOCK_SIZE; current_idx++) {

        index_a_j  = current_idx * CUDA_BLOCK_SIZE + threadIdx.x;
        index_b_k = current_idx * CUDA_BLOCK_SIZE + threadIdx.y;

        if (index_a_j < matrix_size && row_index < matrix_size)
            matrix_a_shared[threadIdx.y][threadIdx.x] = a[row_index * matrix_size + index_a_j];
        else
            matrix_a_shared[threadIdx.y][threadIdx.x] = 0;

        if (index_b_k < matrix_size && column_index < matrix_size)
            matrix_b_shared[threadIdx.y][threadIdx.x] = b[index_b_k * matrix_size + column_index];
        else
            matrix_b_shared[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int n = 0; n < CUDA_BLOCK_SIZE; n++)
            result_value += matrix_a_shared[threadIdx.y][n] * matrix_b_shared[n][threadIdx.x];

        __syncthreads();
    }

    if (row_index < matrix_size && column_index < matrix_size)
        result[((blockIdx.y * blockDim.y + threadIdx.y) * matrix_size) +
        (blockIdx.x * blockDim.x) + threadIdx.x] = result_value;
}

double processMulGpu(double* matrix_A, double* matrix_B, double* result, size_t matrix_size, bool isShared) {
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

    if (isShared) {
        mulSharedGpuKernel << <cuda_blocks, cuda_threads >> > (mat_A, mat_B, mat_res, matrix_size);
    }
    else {
        mulGpuKernel << <cuda_blocks, cuda_threads >> > (mat_A, mat_B, mat_res, matrix_size);
    }

    cudaMemcpy(result, mat_res, bytes_count, cudaMemcpyDeviceToHost);

    cudaFree(mat_A);
    cudaFree(mat_B);
    cudaFree(mat_res);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end); // ms

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return elapsed_time / 1000.0f; // seconds
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

double* generateRandomMatrix(size_t matrix_dim) {
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

    double* matrix_A = generateRandomMatrix(matrix_dim);
    double* matrix_B = generateRandomMatrix(matrix_dim);
    double* matrix_res_cpu = new double[matrix_dim];
    double* matrix_res_gpu = new double[matrix_dim];
    bool isShared = false;
    
    float cpu_time = processMulCpu(matrix_A, matrix_B, matrix_res_cpu, matrix_size);

    float gpu_time = processMulGpu(matrix_A, matrix_B, matrix_res_gpu, matrix_size, isShared);
    double max_diff = findResultsMaxDiff(matrix_res_cpu, matrix_res_gpu, matrix_dim);
    
    isShared = true;
    float gpu_time_shared = processMulGpu(matrix_A, matrix_B, matrix_res_gpu, matrix_size, isShared);

    double max_diff_shared = findResultsMaxDiff(matrix_res_cpu, matrix_res_gpu, matrix_dim);

    double result_max_diff = std::max(max_diff, max_diff_shared);

    printf("Matrix size: %d\n", matrix_size);
    printf("CPU execution time: %lf s\n", cpu_time);
    printf("GPU execution time (normal): %lf s\n", gpu_time);
    printf("GPU execution time (shared memory): %lf s\n", gpu_time_shared);
    printf("Max error: %lf\n", result_max_diff);
    printf("\n");

    clearResources(matrix_A, matrix_B, matrix_res_cpu, matrix_res_gpu);
    return 0;
}