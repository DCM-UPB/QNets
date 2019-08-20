#include <iostream>
#include <random>
#include <cassert>
#include <memory>

#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/Sigmoid.hpp"
#include "qnets/actf/SRLU.hpp"
#include "qnets/actf/Exp.hpp"
#include "qnets/poly/FeedForwardNeuralNetwork.hpp"

int main()
{
    using namespace std;
    using namespace templ;

    const double EXTRA_TINY = 1.e-16; // starts to fail below 1.e-18

    // Setup TemplNet
    const int NU_IN = 5;
    using layer1 = LayerConfig<9, actf::Sigmoid>;
    using layer2 = LayerConfig<7, actf::SRLU>;
    using layer3 = LayerConfig<5, actf::Exp>;
    const auto dopt = DerivConfig::D12_VD1; // test all derivs that are available for both networks
    using TestNet = TemplNet<double, dopt, NU_IN, layer1, layer2, layer3>;

    TestNet tmpl{};

    // Setup PolyNet
    FeedForwardNeuralNetwork ffnn(6, 10, 6);
    ffnn.pushHiddenLayer(8);

    for (int i = 0; i < ffnn.getNNLayer(0)->getNNeuralUnits(); ++i) {
        ffnn.getNNLayer(0)->getNNUnit(i)->setActivationFunction(std_actf::provideActivationFunction("LGS"));
    }
    for (int i = 0; i < ffnn.getNNLayer(1)->getNNeuralUnits(); ++i) {
        ffnn.getNNLayer(1)->getNNUnit(i)->setActivationFunction(std_actf::provideActivationFunction("SRLU"));
    }
    for (int i = 0; i < ffnn.getOutputLayer()->getNNeuralUnits(); ++i) {
        ffnn.getOutputLayer()->getNNUnit(i)->setActivationFunction(std_actf::provideActivationFunction("EXP"));
    }

    ffnn.connectFFNN();
    ffnn.assignVariationalParameters();
    ffnn.addSecondDerivativeSubstrate();
    ffnn.addVariationalFirstDerivativeSubstrate();


    // random generator with fixed seed for generating the beta, in order to eliminate randomness of results in the unittest
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(1337);
    rd = uniform_real_distribution<double>(-0.01, 0.01);
    for (int i = 0; i < ffnn.getNBeta(); ++i) {
        ffnn.setBeta(i, rd(rgen));
        tmpl.setBeta(i, ffnn.getBeta(i));
    }
    for (int i = 0; i < ffnn.getNBeta(); ++i) {
        assert(ffnn.getBeta(i) == tmpl.getBeta(i));
    }

    // now verify that both networks are equivalent
    double x[5] = {0.7, -0.2, -0.5, 0.1, 0.3};
    ffnn.setInput(x);
    tmpl.setInput(x, x + 5);

    ffnn.FFPropagate();
    tmpl.FFPropagate();

    for (int i = 0; i < ffnn.getNOutput(); ++i) {
        std::cout << "f_" << i << ": poly " << ffnn.getOutput(i) << " tmpl " << tmpl.getOutput(i) << std::endl;
        assert(ffnn.getOutput(i) == tmpl.getOutput(i));
        for (int j = 0; j < ffnn.getNInput(); ++j) {
            std::cout << "d1_" << i << "_" << j << ": poly " << ffnn.getFirstDerivative(i, j) << " tmpl " << tmpl.getD1(i, j) << std::endl;
            std::cout << "d2_" << i << "_" << j << ": poly " << ffnn.getSecondDerivative(i, j) << " tmpl " << tmpl.getD2(i, j) << std::endl;
            assert(fabs(ffnn.getFirstDerivative(i, j) - tmpl.getD1(i, j)) < EXTRA_TINY);
            assert(fabs(ffnn.getSecondDerivative(i, j) - tmpl.getD2(i, j)) < EXTRA_TINY);
        }
        for (int j = 0; j < ffnn.getNBeta(); ++j) {
            std::cout << "vd1_" << i << "_" << j << ": poly " << ffnn.getVariationalFirstDerivative(i, j) << " tmpl " << tmpl.getVD1(i, j) << std::endl;
            assert(fabs(ffnn.getVariationalFirstDerivative(i, j) - tmpl.getVD1(i, j)) < EXTRA_TINY);
        }
    }

    return 0;
}
