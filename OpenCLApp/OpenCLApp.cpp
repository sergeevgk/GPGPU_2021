#include <stdio.h>
#include <stdlib.h>    
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include <cstdlib>
#include <iostream>

#pragma warning(disable: 4996)
#define eps 1e-6

#define SIZE_M 100
#define SIZE_N 100
#define SIZE_P 100

//define function 
float* init_matrix(float* A, cl_int row, cl_int col, int value);
void print_matrix(float* A, cl_int row, cl_int col);
int  check_result(float* res, float* c_res, float* a, float* b, cl_int m, cl_int n, cl_int p);

float* init_matrix(float* A, cl_int row, cl_int col, float value) {
    int i = 0;
    for (i = 0; i < row * col; i++) {
        if (value == 0)
            A[i] = 0;
        else
            A[i] = value;
    }
    return A;
}
void print_matrix(float* A, cl_int row, cl_int col) {
    int i = 0;
    for (i = 0; i < row * col; i++) {
        if (i % col == 0)
            printf("\n");
        printf("%f ", A[i]);
    }
    printf("\n================\n");
}
int  check_result(float* res, float* c_res, float* a, float* b, cl_int m, cl_int n, cl_int p) {
    int i, k, error = 0;
    for (i = 0; i < m * p; i++) {
        int div_i = i / p;
        int mod_i = i % p;
        cl_float tmp = 0;
        for (k = 0; k < n; k++) {
            tmp += a[div_i * n + k] * b[p * k + mod_i];
        }
        c_res[i] = tmp;
        if (abs(c_res[i] - res[i]) > eps)
            error = -1;
    }
    return error;
}

cl_program load_program(cl_context context, cl_device_id device, const char* filename)
{
    FILE* fp;
    fopen_s(&fp, filename, "rt");
    size_t length;
    char* data;
    cl_program program = 0;
    cl_int status = 0;
    if (!fp) return 0;

    // get file length
    fseek(fp, 0, SEEK_END);
    length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // read program source
    data = (char*)malloc(length + 1);
    fread(data, sizeof(char), length, fp);
    data[length] = '\0';
    // create and build program 
    program = clCreateProgramWithSource(context, 1, (const char**)&data, 0, 0);
    if (program == 0)
        return 0;
    status = clBuildProgram(program, 1, &device, 0, 0, 0);
    if (status != CL_SUCCESS) {
        printf("Error:  Building Program from file %s.\n", filename);
        return 0;
    }
    return program;
}

void Release(cl_kernel kernelFunction,
    cl_program program,
    cl_mem cl_a,
    cl_mem cl_b,
    cl_mem cl_res,
    cl_command_queue queue,
    cl_context context)
{
    clReleaseKernel(kernelFunction);
    clReleaseProgram(program);
    clReleaseMemObject(cl_a);
    clReleaseMemObject(cl_b);
    clReleaseMemObject(cl_res);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

}

int main(int argc, char** argv)
{
    cl_int err = 0;
    cl_uint num = 0;
    cl_platform_id* platforms = NULL;
    cl_context_properties prop[3] = { 0 };
    cl_context context = 0;
    cl_device_id* devices = NULL;
    cl_command_queue queue = 0;
    cl_program program = 0;
    cl_mem cl_a = 0, cl_b = 0, cl_res = 0;
    cl_kernel kernelFunction = 0;
    cl_event event;
    int num_total_devices = 0;
    char devname[16][256] = { {0} };
    size_t cb, work_size;

    const cl_int m = SIZE_M, n = SIZE_N, p = SIZE_P;
    cl_float* a, *b, *res, *cpu_res;
    a = (cl_float*)malloc(m * n * sizeof(cl_float));
    b = (cl_float*)malloc(n * p * sizeof(cl_float));
    res = (cl_float*)malloc(m * p * sizeof(cl_float));
    int resultCorrectness = 0;
    cpu_res = (cl_float*)malloc(m * p * sizeof(cl_float));
    //Initialize matrices
    init_matrix(a, m, n, 1.0f);
    init_matrix(b, n, p, 2.0f);
    init_matrix(res, m, p, 0.0f);
    init_matrix(cpu_res, m, p, 0.0f);

    if (clGetPlatformIDs(0, 0, &num) != CL_SUCCESS) {
        printf("Unable to get platforms.\n");
        return 0;
    }

    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num);
    if (clGetPlatformIDs(num, platforms, NULL) != CL_SUCCESS) {
        printf("Unable to get platform ID.\n");
        return 0;
    }

    if (clGetPlatformIDs(0, 0, &num) != CL_SUCCESS) {
        printf("Unable to get platforms.\n");
        return 0;
    }

    prop[0] = CL_CONTEXT_PLATFORM;
    prop[1] = (cl_context_properties)platforms[0];
    prop[2] = 0;
    context = clCreateContextFromType(prop, CL_DEVICE_TYPE_ALL, NULL, NULL, NULL);
    if (context == 0) {
        printf("Can't create OpenCL context.\n");
        Release(kernelFunction, program, cl_a, cl_b, cl_res, queue, context);
        return -1;
    }
    
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    devices = (cl_device_id*)malloc(cb);
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, 0);
    if (cb == 0) {
        printf("Can't get devices.\n");
        Release(kernelFunction, program, cl_a, cl_b, cl_res, queue, context);
        return -1;
    }
    num_total_devices = cb / sizeof(cl_device_id);
    const cl_queue_properties queueProperies = CL_QUEUE_PROFILING_ENABLE;
    //Specify the queue
    queue = clCreateCommandQueue(context, devices[0], queueProperies, 0);
    if (queue == 0) {
        printf("Can't create command queue.\n");
        Release(kernelFunction, program, cl_a, cl_b, cl_res, queue, context);
        return -1;
    }
    program = load_program(context, devices[0], "program.cl");
    if (program == 0) {
        printf("Can't build program.\n");
        Release(kernelFunction, program, cl_a, cl_b, cl_res, queue, context);
        return -1;
    }
    cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * m * n, a, NULL);
    cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * n * p, b, NULL);
    cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * m * p, res, NULL);
    if (cl_a == 0 || cl_b == 0 || cl_res == 0) {
        printf("Can't create OpenCL buffers.\n");
        Release(kernelFunction, program, cl_a, cl_b, cl_res, queue, context);
        return 1;
    }

    kernelFunction = clCreateKernel(program, "multiply", &err);

    if (kernelFunction == 0) {
        printf("Can't load kernel.\n");
        Release(kernelFunction, program, cl_a, cl_b, cl_res, queue, context);
        return -1;
    }
    clSetKernelArg(kernelFunction, 0, sizeof(cl_mem), &cl_a);
    clSetKernelArg(kernelFunction, 1, sizeof(cl_mem), &cl_b);
    clSetKernelArg(kernelFunction, 2, sizeof(cl_mem), &cl_res);
    clSetKernelArg(kernelFunction, 3, sizeof(cl_int), &m);
    clSetKernelArg(kernelFunction, 4, sizeof(cl_int), &n);
    clSetKernelArg(kernelFunction, 5, sizeof(cl_int), &p);
    // number of work items == number of elements in res matrix
    work_size = m * p;
    // runs kernel worksize times => one work item counts one element in res matrix
    if (clEnqueueNDRangeKernel(queue, kernelFunction, 1, 0, &work_size, 0, 0, 0, &event) != CL_SUCCESS)
        printf("Can't enqueue kernel.\n");

    if(clEnqueueReadBuffer(queue, cl_res, CL_TRUE, 0, sizeof(cl_float) * work_size, res, 0, 0, 0) != CL_SUCCESS)
        printf("Can't enqueue read buffer.\n");

    clWaitForEvents(1, &event);

    clFinish(queue);
    // check results
    err = 0;
    resultCorrectness = check_result(res, cpu_res, a, b, m, n, p);
    if (resultCorrectness == -1) 
        printf("Matrix multiplication has errors.\n");
    else 
        printf("Matrix multiplication is correct.\n");
    printf("Press any key...");
    std::cin.get();
    // print results
    //print_matrix(a, m, n);
    //print_matrix(b, n, p);
    //print_matrix(res, m, p);
    //print_matrix(cpu_res, m, p);

    Release(kernelFunction, program, cl_a, cl_b, cl_res, queue, context);

    return 0;
}

