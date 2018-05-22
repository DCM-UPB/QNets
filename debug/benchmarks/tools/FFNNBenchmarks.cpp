#include <iostream>

#include "FeedForwardNeuralNetwork.hpp"
#include "Timer.cpp"

double benchmark_FFPropagate(FeedForwardNeuralNetwork * const ffnn, const double * const * const xdata, const int neval) {
    Timer * const timer = new Timer();

    timer->reset();
    for (int i=0; i<neval; ++i) {
        ffnn->setInput(xdata[i]);
        ffnn->FFPropagate();
    }
    return timer->elapsed();
}

double benchmark_actf_derivs(ActivationFunctionInterface * actf, const double * const xdata, const int neval, const bool flag_d1 = true, const bool flag_d2 = true, const bool flag_d3 = true, const bool flag_fad = true) {
    Timer * const timer = new Timer();
    double v, v1d, v2d, v3d;

    if (flag_fad) {
        timer->reset();
        for (int i=0; i<neval; ++i) {
            actf->fad(xdata[i], v, v1d, v2d, v3d, flag_d1, flag_d2, flag_d3);
        }
        return timer->elapsed();
    }
    else {
        timer->reset();
        for (int i=0; i<neval; ++i) {
            v = actf->f(xdata[i]);
            v1d = flag_d1 ? actf->f1d(xdata[i]) : 0.0;
            v2d = flag_d2 ? actf->f2d(xdata[i]) : 0.0;
            v3d = flag_d3 ? actf->f3d(xdata[i]) : 0.0;
        }
        return timer->elapsed();
    }
}
