#include <iostream>
#include <random>

#include "ActivationFunctionManager.hpp"
#include "PrintUtilities.hpp"

#include "../tools/FFNNBenchmarks.cpp"

int main (void) {
    using namespace std;

    const int neval = 20000000;

    const int nactfs = 7;
    const string actf_ids[nactfs] = {"lgs", "gss", "id_", "tans", "sin", "relu", "selu"};

    double * const indata = new double[neval]; // 1d input data for actf bench

    // generate some random input
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(18984687);
    rd = uniform_real_distribution<double>(-sqrt(3.), sqrt(3.)); // uniform with variance 1
    for (int i=0; i<neval; ++i) indata[i] = rd(rgen);

    // ACTF deriv benchmark
    for (int iactf=0; iactf<nactfs; ++iactf) {
        cout << "ACTF derivative benchmark with " << neval << " evaluations for " << actf_ids[iactf] << " activation function." << endl;
        cout << "====================================================================================" << endl << endl;
        cout << "individual function calls:" << endl;
        cout << "f:          " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, false, false, false, false) << " seconds" << endl;
        cout << "f+d1:       " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, true, false, false, false) << " seconds" << endl;
        cout << "f+d2:       " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, false, true, false, false) << " seconds" << endl;
        cout << "f+d3:       " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, false, false, true, false) << " seconds" << endl;
        cout << "f+d1+d2:    " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, true, true, false, false) << " seconds" << endl;
        cout << "f+d1+d3:    " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, true, false, true, false) << " seconds" << endl;
        cout << "f+d2+d3:    " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, false, true, true, false) << " seconds" << endl;
        cout << "f+d1+d2+d3: " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, true, true, true, false) << " seconds" << endl;
        cout << endl;
        cout << "fad function call:" << endl;
        cout << "f:          " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, false, false, false, true) << " seconds" << endl;
        cout << "f+d1:       " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, true, false, false, true) << " seconds" << endl;
        cout << "f+d2:       " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, false, true, false, true) << " seconds" << endl;
        cout << "f+d3:       " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, false, false, true, true) << " seconds" << endl;
        cout << "f+d1+d2:    " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, true, true, false, true) << " seconds" << endl;
        cout << "f+d1+d3:    " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, true, false, true, true) << " seconds" << endl;
        cout << "f+d2+d3:    " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, false, true, true, true) << " seconds" << endl;
        cout << "f+d1+d2+d3: " << benchmark_actf_derivs(std_actf::provideActivationFunction(actf_ids[iactf]), indata, neval, true, true, true, true) << " seconds" << endl;
        cout << "====================================================================================" << endl << endl;
        cout << endl;
    }

    delete [] indata;
    return 0;

}

