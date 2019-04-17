#include <iostream>
#include <random>

#include "qnets/net/TemplNet.hpp"

int main()
{
    using namespace std;

    const double TINY = 0.0001;

    using TestNet = TemplNet<int, double, LogACTF<>, LogACTF<>, 3, 3, 1, 5>;
    TestNet test;

    cout << TestNet::NLAYERS << endl;
    cout << TestNet::NUNITS << endl;


 /*   FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 5, 3);
    ffnn->pushHiddenLayer(4);

    ffnn->pushFeatureMapLayer(5);
    ffnn->getFeatureMapLayer(0)->setNMaps(0, 0, 1, 1, 2);

    ffnn->pushFeatureMapLayer(4);
    ffnn->getFeatureMapLayer(1)->setNMaps(1, 1, 0, 1, 0); // we specify only 3 units
    ffnn->getFeatureMapLayer(1)->setSize(6); // now the other 2 should be defaulted to IDMU (generates warning)
    ffnn->getFeatureMapLayer(1)->setNMaps(1, 1, 0, 1, 2); // to suppress further warning on copies

    //printFFNNStructure(ffnn);

    ffnn->connectFFNN();*/

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
