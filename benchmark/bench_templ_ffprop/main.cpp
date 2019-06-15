#include <iomanip>
#include <iostream>
#include <random>
#include <memory>

#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/Sigmoid.hpp"

#include "FFNNBenchmarks.hpp"

using namespace std;

template <class TemplNet>
void run_single_benchmark(const string &label, TemplNet &tnet, const double xdata[], const int neval, const int nruns)
{
    pair<double, double> result;
    const double time_scale = 1000000.; //microseconds

    result = sample_benchmark(benchmark_TemplProp<TemplNet>, nruns, tnet, xdata, neval);
    cout << label << ":" << setw(max(1, 20 - static_cast<int>(label.length()))) << setfill(' ') << " " << result.first/neval*time_scale << " +- " << result.second/neval*time_scale << " microseconds" << endl;
}

template <int I>
void run_benchmark_netpack(const double xdata[], const int ndata[], const int xoffset, const int neval[], const int nruns) {}

template <int I, class TNet, class ... Args>
void run_benchmark_netpack(const double xdata[], const int ndata[], const int xoffset, const int neval[], const int nruns, TNet &tnet, Args& ... tnets)
{
    using namespace templ;
    cout << "FFPropagate benchmark with " << nruns << " runs of " << neval[I] << " FF-Propagations, for a FFNN of shape " << TNet::getNInput() << "x" << TNet::getNUnit(0) << "x" << TNet::getNUnit(1) << "x" << TNet::getNOutput() << " ." << endl;
    cout << "=========================================================================================" << endl << endl;
    cout << "Benchmark results (time per propagation):" << endl;

    tnet.dflags.set(DerivConfig::OFF);
    run_single_benchmark("f", tnet, xdata + xoffset, neval[I], nruns);

    tnet.dflags.set(DerivConfig::D1);
    run_single_benchmark("f+d1", tnet, xdata + xoffset, neval[I], nruns);

    tnet.dflags.set(DerivConfig::VD1);
    run_single_benchmark("f+vd1", tnet, xdata + xoffset, neval[I], nruns);

    tnet.dflags.set(DerivConfig::D1_VD1);
    run_single_benchmark("f+d1+vd1", tnet, xdata + xoffset, neval[I], nruns);

    tnet.dflags.set(DerivConfig::D12);
    run_single_benchmark("f+d1+d2", tnet, xdata + xoffset, neval[I], nruns);

    tnet.dflags.set(DerivConfig::D12_VD1);
    run_single_benchmark("f+d1+d2+vd1", tnet, xdata + xoffset, neval[I], nruns);

    tnet.dflags.set(DerivConfig::D12_VD12);
    run_single_benchmark("f+d1+d2+vd1+vd2", tnet, xdata + xoffset, neval[I], nruns);

    cout << "=========================================================================================" << endl << endl << endl;

    run_benchmark_netpack<I + 1, Args...>(xdata, ndata, xoffset + ndata[I], neval, nruns, tnets...);
}

int main()
{
    using namespace templ;

    const int neval[3] = {200000, 20000, 1000};
    const int nruns = 5;

    const int yndim = 1;
    constexpr int xndim[3] = {6, 24, 96}, nhu1[3] = {12, 48, 192}, nhu2[3] = {6, 24, 96};

    constexpr auto dconf = DerivConfig::D12_VD12; // "allocate" for all derivatives

    using RealT = double;

    // Small Net
    using L1Type_s = LayerConfig<nhu1[0], actf::Sigmoid>;
    using L2Type_s = LayerConfig<nhu2[0], actf::Sigmoid>;
    using L3Type_s = LayerConfig<yndim, actf::Sigmoid>;
    using NetType_s = TemplNet<RealT, dconf, xndim[0], L1Type_s, L2Type_s, L3Type_s>;
    auto tnet_s_ptr = std::make_unique<NetType_s>();
    auto &tnet_s = *tnet_s_ptr;

    // Medium Net
    using L1Type_m = LayerConfig<nhu1[1], actf::Sigmoid>;
    using L2Type_m = LayerConfig<nhu2[1], actf::Sigmoid>;
    using L3Type_m = LayerConfig<yndim, actf::Sigmoid>;
    using NetType_m = TemplNet<RealT, dconf, xndim[1], L1Type_m, L2Type_m, L3Type_m>;
    auto tnet_m_ptr = std::make_unique<NetType_m>();
    auto &tnet_m = *tnet_m_ptr;

    // Large Net
    using L1Type_l = LayerConfig<nhu1[2], actf::Sigmoid>;
    using L2Type_l = LayerConfig<nhu2[2], actf::Sigmoid>;
    using L3Type_l = LayerConfig<yndim, actf::Sigmoid>;
    using NetType_l = TemplNet<RealT, dconf, xndim[2], L1Type_l, L2Type_l, L3Type_l>;
    auto tnet_l_ptr = std::make_unique<NetType_l>();
    auto &tnet_l = *tnet_l_ptr;

    // Data
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

    for (int i=0; i<tnet_s.getNBeta(); ++i) {
        tnet_s.setBeta(i, rd(rgen));
    }
    for (int i=0; i<tnet_m.getNBeta(); ++i) {
        tnet_m.setBeta(i, rd(rgen));
    }
    for (int i=0; i<tnet_l.getNBeta(); ++i) {
        tnet_l.setBeta(i, rd(rgen));
    }

    // FFPropagate benchmark
    run_benchmark_netpack<0>(xdata, ndata, 0, neval, nruns, tnet_s, tnet_m, tnet_l);

    delete[] xdata;

    return 0;
}

