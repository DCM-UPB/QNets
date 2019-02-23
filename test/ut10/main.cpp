#include <iostream>
#include <random>

#include "ffnn/net/FeedForwardNeuralNetwork.hpp"
#include "ffnn/io/PrintUtilities.hpp"
#include "../common/checkDerivatives.hpp"
#include "../common/checkStoreOnFile.hpp"

int main()
{
    using namespace std;

    const double TINY = 0.0001;


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

    // first feature map layer
    ffnn->getFeatureMapLayer(0)->getEDMapUnit(0)->getMap()->setParameters(2, 1, vector<double> {-1., 1.}); // 2D distance of previous 1&2 vs fixed (-1,1) vector
    ffnn->getFeatureMapLayer(0)->getEPDMapUnit(0)->getMap()->setParameters(1, 1, 2); // 1D distance of previous units (1 vs. 2)
    ffnn->getFeatureMapLayer(0)->getIdMapUnit(0)->getMap()->setParameters(1); // set first identity to first non-offset unit
    ffnn->getFeatureMapLayer(0)->getIdMapUnit(1)->getMap()->setParameters(2); // set second identity to second non-offset unit

    // second feature map layer
    ffnn->getFeatureMapLayer(1)->getPSMapUnit(0)->getMap()->setParameters(1, 3); // pair sum of 1 and 3
    ffnn->getFeatureMapLayer(1)->getPDMapUnit(0)->getMap()->setParameters(2, 4); // pair difference of 2 and 4
    ffnn->getFeatureMapLayer(1)->getEPDMapUnit(0)->getMap()->setParameters(2, 1, 3); // 2D euclidean pair distance of 1&2 and 3&4
    ffnn->getFeatureMapLayer(1)->getIdMapUnit(0)->getMap()->setParameters(3); // pass-through identities
    ffnn->getFeatureMapLayer(1)->getIdMapUnit(1)->getMap()->setParameters(4);

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

    //printFFNNStructure(ffnn, false, 0);

    checkStoreOnFile(ffnn, true); // as a side effect the function also adds all substrates to ffnn

    checkDerivatives(ffnn, TINY);

    delete ffnn;

    return 0;
}
