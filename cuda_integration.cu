#include "cuda_integration.h"

__device__ double linearFunction(double a, double b, double x) {
    return a * x + b;
}

__device__ double linearFunctionHighDimension(const double *coeffs, const double *x, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += coeffs[i] * x[i];
    }
    return result;
}

__global__ void integrateLinearGPU(double a, double b, double left, double right, uint64_t seed, uint32_t samples, double *results){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    double sum = 0.0;
    for (uint32_t i = 0; i < samples; i++) {
        double x = curand_uniform(&state) * (right - left) + left;
        double y = linearFunction(a, b, x);
        sum += y; 
    }
    results[idx] = sum / samples * (right - left);
}

__global__ void integrateLinearGPUHighDim(const double *coeffs, int dim, double left, double right, uint64_t seed, uint32_t samples, double *results){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    double sum = 0.0;
    for (uint32_t i = 0; i < samples; i++) {
        double *x = new double[dim];
        for (int d = 0; d < dim; d++) {
            x[d] = curand_uniform(&state) * (right - left) + left;
        }
        double y = linearFunctionHighDimension(coeffs, x, dim);
        sum += y; 
        delete[] x;
    }
    results[idx] = sum / samples * pow((right - left), dim);
}


IntegrationResult integrateLinearCUDA(double a, double b, double left, double right, uint32_t totalSamples) {
    const int threadsPerBlock = 256;
    const int blocks = (totalSamples + threadsPerBlock - 1) / threadsPerBlock;
    const int samplesPerThread = (totalSamples + blocks * threadsPerBlock - 1) / (blocks * threadsPerBlock);

    double *d_results;
    cudaMalloc(&d_results, blocks * threadsPerBlock * sizeof(double));

    integrateLinearGPU<<<blocks, threadsPerBlock>>>(a, b, left, right, time(NULL), samplesPerThread, d_results);

    double *h_results = new double[blocks * threadsPerBlock];
    cudaMemcpy(h_results, d_results, blocks * threadsPerBlock * sizeof(double), cudaMemcpyDeviceToHost);

    double integralValue = 0.0;
    for (int i = 0; i < blocks * threadsPerBlock; i++) {
        integralValue += h_results[i];
    }
    integralValue /= (blocks * threadsPerBlock);

    cudaFree(d_results);
    delete[] h_results;

    IntegrationResult result;
    result.integralValue = integralValue;
    result.errorEstimate = 0.0; // Placeholder for error estimate
    return result;
}