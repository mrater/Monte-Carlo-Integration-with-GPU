#include "cuda_integration.h"
#include <iostream>

int main(){
    // trapezoid: (0, 0), (0, 1), (1, 0), (1, 3). area (integral) = 1*1 + 1*2/2 = 2
    // linear function: y = 2x + 1 over [0, 1]
    IntegrationResult result = integrateLinearCUDA(2.0, 1.0, 0.0, 1.0, 10000);

    std::cout << result.integralValue << "\n";
}