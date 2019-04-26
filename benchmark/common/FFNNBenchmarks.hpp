#include <cmath>
#include <iostream>
#include <tuple>

#include "Timer.hpp"
#include "qnets/poly/FeedForwardNeuralNetwork.hpp"

inline double benchmark_FFPropagate(FeedForwardNeuralNetwork * const ffnn, const double * const xdata, const int neval)
{
    Timer timer(1.);
    const int ninput = ffnn->getNInput();

    timer.reset();
    for (int i = 0; i < neval; ++i) {
        ffnn->setInput(xdata + i*ninput);
        ffnn->FFPropagate();
    }

    return timer.elapsed();
}

template <class TemplNet>
inline double benchmark_TemplProp(TemplNet &tnet, const double xdata[], const int neval)
{
    Timer timer(1.);
    const int ninput = tnet.getNInput();

    double test = 0;
    timer.reset();
    for (int i = 0; i < neval; ++i) {
        tnet.setInput(xdata + i*ninput, xdata + (i+1)*ninput);
        tnet.FFPropagate();
        test += tnet.getVD1(0,i%tnet.nvd1);
    }
    std::cout << test << std::endl;

    return timer.elapsed();
}


inline double benchmark_actf_derivs(ActivationFunctionInterface * const actf, const double * const xdata, const int neval, const bool flag_d1 = true, const bool flag_d2 = true, const bool flag_d3 = true, const bool flag_fad = true)
{
    Timer timer(1.);
    double v = 0., v1d = 0., v2d = 0., v3d = 0.;

    if (flag_fad) {
        timer.reset();
        for (int i = 0; i < neval; ++i) {
            actf->fad(xdata[i], v, v1d, v2d, v3d, flag_d1, flag_d2, flag_d3);
        }
        return timer.elapsed();
    }

    timer.reset();
    for (int i = 0; i < neval; ++i) {
        v = actf->f(xdata[i]);
        v1d = flag_d1 ? actf->f1d(xdata[i]) : 0.0;
        v2d = flag_d2 ? actf->f2d(xdata[i]) : 0.0;
        v3d = flag_d3 ? actf->f3d(xdata[i]) : 0.0;
    }
    return timer.elapsed();
}

template <class BenchT, class ... Args>
inline std::pair<double, double> sample_benchmark(BenchT bench, const int nruns, Args&& ... args)
{
    double times[nruns];
    double mean = 0., err = 0.;

    for (int i = 0; i < nruns; ++i) {
        times[i] = bench(args...);
        mean += times[i];
    }
    mean /= nruns;
    for (int i = 0; i < nruns; ++i) {
        err += pow(times[i] - mean, 2);
    }
    err /= (nruns - 1)*nruns; // variance of the mean
    err = sqrt(err); // standard error of the mean

    const std::pair<double, double> result(mean, err);
    return result;
}
