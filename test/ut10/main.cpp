#include <iostream>
#include <random>

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"
#include "../common/checkDerivatives.hpp"

int main()
{
    using namespace std;

    const double TINY = 0.0001;


    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 5, 3);
    ffnn->pushHiddenLayer(4);
    ffnn->pushFeatureMapLayer(4);
    ffnn->pushFeatureMapLayer(7);
    ffnn->getFeatureMapLayer(0)->setNMaps(1,2);
    ffnn->getFeatureMapLayer(1)->setNMaps(3,3);

    printFFNNStructure(ffnn);

    ffnn->connectFFNN();

    ffnn->getFeatureMapLayer(0)->getEDMapUnit(0)->getMap()->setParameters(1, 1, 2); // distance of both previous non-offset units
    ffnn->getFeatureMapLayer(0)->getIdMapUnit(0)->getMap()->setParameters(1); // set first identity to first non-offset unit
    ffnn->getFeatureMapLayer(0)->getIdMapUnit(1)->getMap()->setParameters(2); // set second identity to second non-offset unit

    // all useful distances from previous layer
    ffnn->getFeatureMapLayer(1)->getEDMapUnit(0)->getMap()->setParameters(1, 1, 2);
    ffnn->getFeatureMapLayer(1)->getEDMapUnit(1)->getMap()->setParameters(1, 1, 3);
    ffnn->getFeatureMapLayer(1)->getEDMapUnit(2)->getMap()->setParameters(1, 2, 3);

    // set identities to previous units
    ffnn->getFeatureMapLayer(1)->getIdMapUnit(0)->getMap()->setParameters(1);
    ffnn->getFeatureMapLayer(1)->getIdMapUnit(1)->getMap()->setParameters(2);
    ffnn->getFeatureMapLayer(1)->getIdMapUnit(2)->getMap()->setParameters(3);

    // random generator with fixed seed for generating the beta, in order to eliminate randomness of results in the unittest
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(18984687);
    rd = uniform_real_distribution<double>(-2., 2.);
    for (int i=0; i<ffnn->getNBeta(); ++i){
        ffnn->setBeta(i, rd(rgen));
    }

    ffnn->assignVariationalParameters();
    ffnn->addFirstDerivativeSubstrate();
    ffnn->addSecondDerivativeSubstrate();
    ffnn->addVariationalFirstDerivativeSubstrate();

    printFFNNStructure(ffnn, false, 0);

    check_derivatives(ffnn, TINY);

    delete ffnn;

    return 0;
}
