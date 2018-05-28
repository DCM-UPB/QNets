#include <iostream>
#include <random>

#include "ActivationFunctionManager.hpp"
#include "PrintUtilities.hpp"
#include "FeedForwardNeuralNetwork.hpp"

#include "FFNNBenchmarks.cpp"

int main (void) {
    using namespace std;

    FeedForwardNeuralNetwork * ffnn;

    const double xndim = 2, yndim = 1;
    const int nhl = 2;
    const int nhu[nhl] = {6,3};

    const int neval = 500000;

    const int nactfs = 7;
    const string actf_ids[nactfs] = {"lgs", "gss", "id_", "tans", "sin", "relu", "selu"};

    double ** const xdata = new double*[neval]; // xndim input data for propagate bench
    for (int i=0; i<neval; ++i) xdata[i] = new double[2];

    // generate some random input
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(18984687);
    rd = uniform_real_distribution<double>(-sqrt(3.), sqrt(3.)); // uniform with variance 1
    for (int i=0; i<neval; ++i){
        for (int j=0; j<xndim; ++j) xdata[i][j] = rd(rgen);
    }

    // FFPropagate benchmark
    for (int iactf=0; iactf<nactfs; ++iactf) {
        ffnn = new FeedForwardNeuralNetwork(xndim+1, nhu[0], yndim+1);
        for (int i=1; i<nhl; ++i) ffnn->pushHiddenLayer(nhu[i]);
        ffnn->connectFFNN();

        //Set ACTFs for hidden units
        for (int i=0; i<nhl; ++i) {
            for (int j=1; j<nhu[i]; ++j) {
                ffnn->getLayer(i+1)->getUnit(j)->setActivationFunction(std_actf::provideActivationFunction(actf_ids[iactf]));
            }
        }

        //Set ID ACTFs for output units
        for (int j=1; j<yndim+1; ++j) {
            ffnn->getLayer(nhl+1)->getUnit(j)->setActivationFunction(std_actf::provideActivationFunction("id_"));
        }

        cout << "FFPropagate benchmark with " << neval << " FF-Propagations for " << actf_ids[iactf] << " activation function." << endl;
        cout << "====================================================================================" << endl << endl;
        cout << "NN structure looks like:" << endl;
        printFFNNStructure(ffnn);
        cout << endl << endl;
        cout << "Benchmark results:" << endl;
        cout << "noderivs: " << benchmark_FFPropagate(ffnn, xdata, neval) << " seconds" << endl;
        ffnn->addFirstDerivativeSubstrate();
        ffnn->addSecondDerivativeSubstrate();
        ffnn->addVariationalFirstDerivativeSubstrate();
        cout << "stdderivs: " << benchmark_FFPropagate(ffnn, xdata, neval) << " seconds" << endl;
        ffnn->addCrossFirstDerivativeSubstrate();
        ffnn->addCrossSecondDerivativeSubstrate();
        cout << "allderivs: " << benchmark_FFPropagate(ffnn, xdata, neval) << " seconds" << endl;
        cout << "====================================================================================" << endl << endl;
        cout << endl;

        delete ffnn;
    }

    for (int i=0; i<neval; ++i) delete [] xdata[i];
    delete [] xdata;
    return 0;

}

