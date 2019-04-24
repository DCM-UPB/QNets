#include <iostream>
#include <random>
#include <cassert>
#include <memory>

#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/Sigmoid.hpp"
#include "qnets/poly/FeedForwardNeuralNetwork.hpp"

int main()
{
    using namespace std;
    using namespace templ;

    const double EXTRA_TINY = 1.e-16; // starts to fail below 1.e-18

    const int NU_IN = 2;
    using layer1 = LayerConfig<4, actf::Sigmoid>;
    using layer2 = LayerConfig<3, actf::Sigmoid>;
    using layer3 = LayerConfig<2, actf::Sigmoid>;
    const auto dopt = DerivConfig::D12_VD1; // test all derivs that are available for TemplNet
    using TestNet = TemplNet<double, dopt, NU_IN, layer1, layer2, layer3>;

    TestNet tmpl{};

    FeedForwardNeuralNetwork ffnn(3, 5, 3);
    ffnn.pushHiddenLayer(4);
    ffnn.connectFFNN();

    // random generator with fixed seed for generating the beta, in order to eliminate randomness of results in the unittest
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(18984687);
    rd = uniform_real_distribution<double>(-2., 2.);
    for (int i = 0; i < ffnn.getNBeta(); ++i) {
        ffnn.setBeta(i, rd(rgen));
        tmpl.setBeta(i, ffnn.getBeta(i));
    }
    for (int i = 0; i < ffnn.getNBeta(); ++i) {
        assert(ffnn.getBeta(i) == tmpl.getBeta(i));
    }

    ffnn.assignVariationalParameters();
    ffnn.addSecondDerivativeSubstrate();
    ffnn.addVariationalFirstDerivativeSubstrate();

    double x[2] = {1.7, -0.2};
    ffnn.setInput(x);
    tmpl.setInput(x, x + 2);

    ffnn.FFPropagate();
    tmpl.FFPropagate();

    for (int i = 0; i < ffnn.getNOutput(); ++i) {
        assert(ffnn.getOutput(i) == tmpl.getOutput()[i]);
        for (int j = 0; j < ffnn.getNInput(); ++j) {
            //std::cout << "d1_" << i << "_" << j << ": poly " << ffnn.getFirstDerivative(i, j) << " tmpl " << tmpl.getD1(i, j) << std::endl;
            //std::cout << "d2_" << i << "_" << j << ": poly " << ffnn.getSecondDerivative(i, j) << " tmpl " << tmpl.getD2(i, j) << std::endl;
            assert(fabs(ffnn.getFirstDerivative(i, j) - tmpl.getD1(i, j)) < EXTRA_TINY);
            assert(fabs(ffnn.getSecondDerivative(i, j) - tmpl.getD2(i, j)) < EXTRA_TINY);
        }
        for (int j = 0; j < ffnn.getNBeta(); ++j) {
            std::cout << "vd1_" << i << "_" << j << ": poly " << ffnn.getVariationalFirstDerivative(i, j) << " tmpl " << tmpl.getVD1(i, j) << std::endl;
            assert(fabs(ffnn.getVariationalFirstDerivative(i, j) - tmpl.getVD1(i, j)) < EXTRA_TINY); // currently unavailable due to backprop change
        }
    }

    return 0;
}
