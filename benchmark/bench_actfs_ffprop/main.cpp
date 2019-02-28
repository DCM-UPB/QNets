#include <iomanip>
#include <iostream>
#include <random>

#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/io/PrintUtilities.hpp"
#include "ffnn/net/FeedForwardNeuralNetwork.hpp"

#include "FFNNBenchmarks.hpp"

using namespace std;

void run_single_benchmark(const string &label, FeedForwardNeuralNetwork * const ffnn, const double * const xdata, const int neval, const int nruns) {
    pair<double, double> result;
    const double time_scale = 1000000.; //microseconds

    result = sample_benchmark_FFPropagate(ffnn, xdata, neval, nruns);
    cout << label << ":" << setw(max(1, 20-static_cast<int>(label.length()))) << setfill(' ') << " " << result.first/neval*time_scale << " +- " << result.second/neval*time_scale << " microseconds" << endl;
}

int main () {
    const int neval = 1000;
    const int nruns = 5;

    const int xndim = 4, yndim = 1;
    const int nhl = 2;
    const int nhu[nhl] = {9,5};

    const int nactfs = 8;
    const string actf_ids[nactfs] = {"LGS", "GSS", "ID", "TANS", "SIN", "RELU", "SELU", "SRLU"};

    const int ndata = neval*xndim;
    auto * xdata = new double[ndata]; // xndim input data for propagate bench

    // generate some random input
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(18984687);
    rd = uniform_real_distribution<double>(-sqrt(3.), sqrt(3.)); // uniform with variance 1
    for (int i=0; i<ndata; ++i){
        xdata[i] = rd(rgen);
    }

    // FFPropagate benchmark
    for (const auto & actf_id : actf_ids) {
        FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(xndim+1, nhu[0], yndim+1);
        for (int i=1; i<nhl; ++i) { ffnn->pushHiddenLayer(nhu[i]);
}
        ffnn->connectFFNN();
        ffnn->assignVariationalParameters();

        //Set ACTFs for hidden units
        for (int i=0; i<nhl; ++i) {
            for (int j=1; j<nhu[i]; ++j) {
                ffnn->getNNLayer(i)->getNNUnit(j-1)->setActivationFunction(std_actf::provideActivationFunction(actf_id));
            }
        }

        //Set ID ACTFs for output units
        for (int j=1; j<yndim+1; ++j) {
            ffnn->getNNLayer(nhl)->getNNUnit(j-1)->setActivationFunction(std_actf::provideActivationFunction("ID"));
        }

        cout << "FFPropagate benchmark with " << nruns << " runs of " << neval << " FF-Propagations for " << actf_id << " activation function." << endl;
        cout << "=========================================================================================" << endl << endl;
        cout << "NN structure looks like:" << endl << endl;
        printFFNNStructure(ffnn, true, 0);
        cout << endl;
        cout << "Benchmark results (time per propagation):" << endl;

        run_single_benchmark("f", ffnn, xdata, neval, nruns);

        ffnn->addFirstDerivativeSubstrate();
        run_single_benchmark("f+d1", ffnn, xdata, neval, nruns);

        ffnn->addSecondDerivativeSubstrate();
        run_single_benchmark("f+d1+d2", ffnn, xdata, neval, nruns);

        ffnn->addVariationalFirstDerivativeSubstrate();
        run_single_benchmark("f+d1+d2+vd1", ffnn, xdata, neval, nruns);

        ffnn->addCrossFirstDerivativeSubstrate();
        run_single_benchmark("f+d1+d2+vd1+cd1", ffnn, xdata, neval, nruns);

        ffnn->addCrossSecondDerivativeSubstrate();
        run_single_benchmark("f+d1+d2+vd1+cd1+cd2", ffnn, xdata, neval, nruns);

        cout << "=========================================================================================" << endl << endl << endl;

        delete ffnn;
    }

    delete [] xdata;
    return 0;
}

