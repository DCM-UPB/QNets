#include <iostream>
#include <random>
#include <iomanip>

#include "ActivationFunctionManager.hpp"
#include "PrintUtilities.hpp"

#include "FFNNBenchmarks.cpp"

using namespace std;

void run_single_benchmark(const string label, const string actf_id, const double * const indata, const int neval, const int nruns, const bool flag_d1, const bool flag_d2, const bool flag_d3, const bool flag_fad) {
    pair<double, double> result;
    const double time_scale = 1000000000.; //nanoseconds

    result = sample_benchmark_actf_derivs(std_actf::provideActivationFunction(actf_id), indata, neval, nruns, flag_d1, flag_d2, flag_d3, flag_fad);
    cout << label << ":" << setw(max(1, 11-(int)label.length())) << setfill(' ') << " " << result.first/neval*time_scale << " +- " << result.second/neval*time_scale << " nanoseconds" << endl;
}

int main (void) {

    const int neval = 100000;
    const int nruns = 50;

    const int nactfs = 7;
    const string actf_ids[nactfs] = {"LGS", "GSS", "ID", "TANS", "SIN", "RELU", "SELU"};

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
        cout << "ACTF derivative benchmark with " << nruns << " runs of " << neval << " evaluations for " << actf_ids[iactf] << " activation function." << endl;
        cout << "===========================================================================================" << endl << endl;
        for (bool flag_fad : { false, true }) {
            if (flag_fad) cout << "Time per evaluation using fad function call:" << endl;
            else cout << "Time per evaluation using individual function calls:" << endl;

            run_single_benchmark("f", actf_ids[iactf], indata, neval, nruns, false, false, false, flag_fad);
            run_single_benchmark("f+d1", actf_ids[iactf], indata, neval, nruns, true, false, false, flag_fad);
            run_single_benchmark("f+d2", actf_ids[iactf], indata, neval, nruns, false, true, false, flag_fad);
            run_single_benchmark("f+d3", actf_ids[iactf], indata, neval, nruns, false, false, true, flag_fad);
            run_single_benchmark("f+d1+d2", actf_ids[iactf], indata, neval, nruns, true, true, false, flag_fad);
            run_single_benchmark("f+d1+d3", actf_ids[iactf], indata, neval, nruns, true, false, true, flag_fad);
            run_single_benchmark("f+d2+d3", actf_ids[iactf], indata, neval, nruns, false, true, true, flag_fad);
            run_single_benchmark("f+d1+d2+d3", actf_ids[iactf], indata, neval, nruns, true, true, true, flag_fad);

            cout << endl;
        }
        cout << "===========================================================================================" << endl << endl << endl;
    }

    delete [] indata;
    return 0;

}

