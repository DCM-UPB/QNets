#include <iostream>
#include <random>
#include <cassert>

#include "qnets/TemplNet.hpp"

int main()
{
    using namespace std;
    using namespace templ;

    //const double TINY = 0.0001;

    constexpr int NU_IN = 2;
    using layer1 = LayerConfig<int, NU_IN, 4, LogACTF>;
    using layer2 = LayerConfig<int, layer1::size(), 2, LogACTF>;
    using derivs = DerivConfig<true, true, true>; // we are going to check them all
    using TestNet = TemplNet<int, double, derivs, layer1, layer2>;

    // static type-based tests
    static_assert(TestNet::getNLayer() == 2, "nlayer != 2");
    static_assert(TestNet::getNUnit() == 6, "nunit != 6");
    static_assert(TestNet::getNInput() == 2, "ninput != 2");
    static_assert(TestNet::getNUnit(0) == 4, "nunit[0] != 4");
    static_assert(TestNet::getNOutput() == 2, "noutput != 2");

    static_assert(TestNet::getNBeta() == 22, "nbeta != 22");
    static_assert(TestNet::getNBeta(0) == 12, "nbeta[0] != 12");
    static_assert(TestNet::getNBeta(1) == 10, "nbeta[1] != 10");
    static_assert(TestNet::getNLink() == 16, "nlink != 16");
    static_assert(TestNet::getNLink(0) == 8, "nlink[0] != 8");
    static_assert(TestNet::getNLink(1) == 8, "nlink[1] != 8");

    static_assert(TestNet::allowsFirstDerivative(), "allow_d1 = false");
    static_assert(TestNet::allowsSecondDerivative(), "allow_d2 = false");
    static_assert(TestNet::allowsVariationalFirstDerivative(), "allow_vd1 = false");


    TestNet test; // create an instance

    constexpr std::array<int, 2> expected_shape{4, 2};
    constexpr std::array<int, 2> expected_betashape{12, 10};

    assert(TestNet::getShape() == expected_shape);
    assert(TestNet::getBetaShape() == expected_betashape);

    assert(test.input.size() == TestNet::getNInput());
    assert(test.output.size() == TestNet::getNOutput());
    assert(test.output.size() == test.getOutput().size());

    cout << "layers ";
    auto l0 = test.getLayer<0>();
    auto l1 = test.getLayer<1>();
    cout << "l0 " << l0.size() << endl;
    cout << "l1: " << l1.size() << endl;

    cout << "beta ";
    for (int i = 0; i < TestNet::getNBeta(); ++i) {
        cout << "b" << i << " " << test.getBeta(i) << "  ";
    }
    cout << endl;

    LogACTF actf{};
    std::array<double, 2> foo{};
    actf.f(test.getOutput().begin(), test.output.end(), foo.begin());
    cout << "actf ";
    for (double out : foo) { cout << out << " "; }
    cout << endl;

/*
       FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 5, 3);
       ffnn->pushHiddenLayer(4);

       ffnn->pushFeatureMapLayer(5);
       ffnn->getFeatureMapLayer(0)->setNMaps(0, 0, 1, 1, 2);

       ffnn->pushFeatureMapLayer(4);
       ffnn->getFeatureMapLayer(1)->setNMaps(1, 1, 0, 1, 0); // we specify only 3 units
       ffnn->getFeatureMapLayer(1)->setSize(6); // now the other 2 should be defaulted to IDMU (generates warning)
       ffnn->getFeatureMapLayer(1)->setNMaps(1, 1, 0, 1, 2); // to suppress further warning on copies

       //printFFNNStructure(ffnn);

       ffnn->connectFFNN();
*/

/*
    // random generator with fixed seed for generating the beta, in order to eliminate randomness of results in the unittest
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(18984687);
    rd = uniform_real_distribution<double>(-2., 2.);
    for (int i = 0; i < ffnn->getNBeta(); ++i) {
        ffnn->setBeta(i, rd(rgen));
    }
*/
    //printFFNNStructure(ffnn, false, 0);

    return 0;
}