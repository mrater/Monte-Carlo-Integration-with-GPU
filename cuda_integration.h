#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

#pragma once

// f(x) = a * x + b
__device__ double linearFunction(double a, double b, double x);

// f(x0, x1, ..., x_n) = c0*x0 + c1*x1 + ... + cn*xn (last coefficient is bias)
__device__ double linearFunctionHighDimension(const double *coeffs, const double *x, int n);

__global__ void integrateLinearGPU(double a, double b, double left, double right, uint64_t seed, uint32_t samples, double *results);
__global__ void integrateLinearGPUHighDim(const double *coeffs, int dim, double left, double right, uint64_t seed, uint32_t samples, double *results);

struct IntegrationResult {
    double integralValue;
    double errorEstimate;
};

IntegrationResult integrateLinearCUDA(double a, double b, double left, double right, uint32_t totalSamples);
IntegrationResult integrateLinearCUDA(const double *coeffs, int dim, double *left, double *right, uint32_t totalSamples);