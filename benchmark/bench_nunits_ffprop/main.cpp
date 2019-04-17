#include <iomanip>
#include <iostream>
#include <random>

#include "qnets/io/PrintUtilities.hpp"

#include "FFNNBenchmarks.hpp"

using namespace std;

void run_single_benchmark(const string &label, FeedForwardNeuralNetwork * const ffnn, const double * const xdata, const int neval, const int nruns)
{
    pair<double, double> result;
    const double time_scale = 1000000.; //microseconds

    result = sample_benchmark_FFPropagate(ffnn, xdata, neval, nruns);
    cout << label << ":" << setw(max(1, 20 - static_cast<int>(label.length()))) << setfill(' ') << " " << result.first/neval*time_scale << " +- " << result.second/neval*time_scale << " microseconds" << endl;
}

int main()
{
    const int neval[3] = {50000, 1000, 20};
    const int nruns = 5;

    const int nhl = 2;
    const int yndim = 1;
    const int xndim[3] = {6, 24, 96}, nhu1[3] = {12, 48, 192}, nhu2[3] = {6, 24, 96};

    int ndata[3], ndata_full = 0;
    for (int i = 0; i < 3; ++i) {
        ndata[i] = neval[i]*xndim[i];
        ndata_full += ndata[i];
    }
    auto * xdata = new double[ndata_full]; // xndim input data for propagate bench

    // generate some random input
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(18984687);
    rd = uniform_real_distribution<double>(-sqrt(3.), sqrt(3.)); // uniform with variance 1
    for (int i = 0; i < ndata_full; ++i) {
        xdata[i] = rd(rgen);
    }

    // FFPropagate benchmark
    int xoffset = 0; // used to shift current xdata pointer
    for (int inet = 0; inet < 3; ++inet) {
        FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(xndim[inet] + 1, nhu1[inet] + 1, yndim + 1);
        for (int i = 1; i < nhl; ++i) {
            ffnn->pushHiddenLayer(nhu2[inet]);
        }
        ffnn->connectFFNN();
        ffnn->assignVariationalParameters();

        cout << "FFPropagate benchmark with " << nruns << " runs of " << neval[inet] << " FF-Propagations, for a FFNN of shape " << xndim[inet] << "x" << nhu1[inet] << "x" << nhu2[inet] << "x" << yndim << " ." << endl;
        cout << "=========================================================================================" << endl << endl;
        cout << "NN structure looks like:" << endl << endl;
        printFFNNStructure(ffnn, true, 0);
        cout << endl;
        cout << "Benchmark results (time per propagation):" << endl;

        run_single_benchmark("f", ffnn, xdata + xoffset, neval[inet], nruns);

        ffnn->addFirstDerivativeSubstrate();
        run_single_benchmark("f+d1", ffnn, xdata + xoffset, neval[inet], nruns);

        ffnn->addSecondDerivativeSubstrate();
        run_single_benchmark("f+d1+d2", ffnn, xdata + xoffset, neval[inet], nruns);

        ffnn->addVariationalFirstDerivativeSubstrate();
        run_single_benchmark("f+d1+d2+vd1", ffnn, xdata + xoffset, neval[inet], nruns);

        /* these currently kill 16GB+ of memory on the largest nets */
        //ffnn->addCrossFirstDerivativeSubstrate();
        //run_single_benchmark("f+d1+d2+vd1+cd1", ffnn, xdata+xoffset, neval[inet], nruns);

        //ffnn->addCrossSecondDerivativeSubstrate();
        //run_single_benchmark("f+d1+d2+vd1+cd1+cd2", ffnn, xdata+xoffset, neval[inet], nruns);

        cout << "=========================================================================================" << endl << endl << endl;

        delete ffnn;
        xoffset += ndata[inet];
    }

    delete[] xdata;
    return 0;
}

