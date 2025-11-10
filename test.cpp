#include "cuda_integration.h"
#include <iostream>

int main(){
    double left = 0, right = 1;
    uint32_t totalSamples = 10000;

    // trapezoid: (0, 0), (0, 1), (1, 0), (1, 3). area (integral) = 1*1 + 1*2/2 = 2
    // linear function: y = 2x + 1 over [0, 1]
    IntegrationResult result = integrateLinearCUDA(2.0, 1.0, left, right, totalSamples);

    // high-dimensional linear function test
    double coeffs[2] = {2.0, 1.0}; // y = 2x + 1
    double left_tab[1] = {0.0};
    double right_tab[1] = {1.0};
    IntegrationResult resultHighDim = integrateLinearCUDA(coeffs, 1, left_tab, right_tab, totalSamples);


    std::cout << result.integralValue << "\n";
    std::cout << resultHighDim.integralValue << "\n";
}