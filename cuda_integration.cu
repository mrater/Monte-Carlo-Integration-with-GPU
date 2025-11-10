#include "cuda_integration.h"

__device__ double linearFunction(double a, double b, double x) {
    return a * x + b;
}

__device__ double linearFunctionHighDimension(const double *coeffs, const double *x, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += coeffs[i] * x[i];
    }
    result += coeffs[n]; //bias
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

__global__ void integrateLinearGPUHighDim(const double *coeffs, int dim, double *left, double *right, uint64_t seed, uint32_t samples, double *results){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    double sum = 0.0;
    for (uint32_t i = 0; i < samples; i++) {
        double *x = new double[dim];
        for (int d = 0; d < dim; d++) {
            x[d] = curand_uniform(&state) * (right[d] - left[d]) + left[d];
        }
        double y = linearFunctionHighDimension(coeffs, x, dim);
        sum += y; 
        delete[] x;
    }
    // F = a/N* sum(f(Xi)), a = (x1 - x0)(y1 - y0)...
    double a = 1.0;
    for (int d = 0; d < dim; d++) {
        a *= (right[d] - left[d]);
    }
    results[idx] = a * sum / (double)samples;
}


IntegrationResult integrateLinearCUDA(double a, double b, double left, double right, uint32_t totalSamples) {
    const int threadsPerBlock = 256;
    const int blocks = (totalSamples + threadsPerBlock - 1) / threadsPerBlock; // each block has some threads
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
    return result;
}

IntegrationResult integrateLinearCUDA(const double *coeffs, int dim, double *left, double *right, uint32_t totalSamples) {
    const int threadsPerBlock = 256;
    const int blocks = (totalSamples + threadsPerBlock - 1) / threadsPerBlock;
    const int samplesPerThread = (totalSamples + blocks * threadsPerBlock - 1) / (blocks * threadsPerBlock);

    double *d_coeffs, *d_left, *d_right;
    cudaMalloc(&d_coeffs, (dim + 1) * sizeof(double));
    cudaMalloc(&d_left, dim * sizeof(double));
    cudaMalloc(&d_right, dim * sizeof(double));

    cudaMemcpy(d_coeffs, coeffs, (dim + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_left, left, dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right, dim * sizeof(double), cudaMemcpyHostToDevice);

    double *d_results;
    cudaMalloc(&d_results, blocks * threadsPerBlock * sizeof(double));

    integrateLinearGPUHighDim<<<blocks, threadsPerBlock>>>(d_coeffs, dim, d_left, d_right, time(NULL), samplesPerThread, d_results);

    double *h_results = new double[blocks * threadsPerBlock];
    cudaMemcpy(h_results, d_results, blocks * threadsPerBlock * sizeof(double), cudaMemcpyDeviceToHost);

    double integralValue = 0.0;
    for (int i = 0; i < blocks * threadsPerBlock; i++) {
        integralValue += h_results[i];
    }
    integralValue /= (blocks * threadsPerBlock);

    cudaFree(d_coeffs);
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_results);
    delete[] h_results;

    IntegrationResult result;
    result.integralValue = integralValue;
    return result;
}