#include "qnets/FeedForwardNeuralNetwork.hpp"

#include "../common/checkStoreOnFile.hpp"

int main()
{
    using namespace std;

    // make a check while the FFNN is not connected yet
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 5, 3);
    ffnn->pushHiddenLayer(4);

    ffnn->getNNLayer(0)->getNNUnit(2)->setActivationFunction(std_actf::provideActivationFunction("GSS"));
    ffnn->getNNLayer(1)->getNNUnit(0)->setActivationFunction(std_actf::provideActivationFunction("GSS"));
    ffnn->getNNLayer(2)->getNNUnit(1)->setActivationFunction(std_actf::provideActivationFunction("GSS"));

    assert(ffnn->getNNLayer(0)->getNNUnit(0)->getActivationFunction()->getIdCode() == "LGS");
    assert(ffnn->getNNLayer(0)->getNNUnit(1)->getActivationFunction()->getIdCode() == "LGS");
    assert(ffnn->getNNLayer(0)->getNNUnit(2)->getActivationFunction()->getIdCode() == "GSS");
    assert(ffnn->getNNLayer(0)->getNNUnit(3)->getActivationFunction()->getIdCode() == "LGS");
    assert(ffnn->getNNLayer(1)->getNNUnit(0)->getActivationFunction()->getIdCode() == "GSS");
    assert(ffnn->getNNLayer(1)->getNNUnit(1)->getActivationFunction()->getIdCode() == "LGS");
    assert(ffnn->getNNLayer(1)->getNNUnit(2)->getActivationFunction()->getIdCode() == "LGS");
    assert(ffnn->getNNLayer(2)->getNNUnit(0)->getActivationFunction()->getIdCode() == "LGS");
    assert(ffnn->getNNLayer(2)->getNNUnit(1)->getActivationFunction()->getIdCode() == "GSS");

    checkStoreOnFile(ffnn);

    delete ffnn;

    return 0;
}
