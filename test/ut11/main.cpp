#include <iostream>
#include <random>

#include "qnets/TemplNet.hpp"

int main()
{
    using namespace std;
    using namespace templ;

    const double TINY = 0.0001;

    constexpr int NU_IN = 2;
    using layer1 = Layer<int, 4, LogACTF<>>;
    using layer2 = Layer<int, 2, LogACTF<>>;
    using derivs = DerivSetup<false, false, false>;
    using TestNet = TemplNet<int, double, derivs, NU_IN, layer1, layer2>;

    // compile time unit test (with possible run-time print)
    cout << "nlayer " << TestNet::nlayer << endl;
    static_assert(TestNet::nlayer == 3, "nlayer != 3");
    cout << "nunits " << TestNet::nunit_tot << endl;
    static_assert(TestNet::nunit_tot == 8, "nunit_tot != 8");
    cout << "nvp " << TestNet::nvp_tot << endl;
    static_assert(TestNet::nvp_tot == 22, "nvp_tot != 22");
    cout << "nlinks " << TestNet::nlink_tot << endl;
    static_assert(TestNet::nlink_tot == 16, "nlink_tot != 16");


    TestNet test;

    cout << "nu ";
    for (auto nu : test.nunits) {
        //cout << nu << " ";
    }
    cout << endl;

    cout << "in ";
    for (auto in : test.input) {
        cout << in << " ";
    }
    cout << endl;

    cout << "out ";
    for (auto out : test.output) {
        cout << out << " ";
    }
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