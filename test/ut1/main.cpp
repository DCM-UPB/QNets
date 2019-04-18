#include <iostream>
#include <random>

#include "qnets/FeedForwardNeuralNetwork.hpp"

#include "../common/checkDerivatives.hpp"

int main()
{
    using namespace std;

    const double TINY = 0.0001;


    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 5, 3);
    ffnn->pushHiddenLayer(4);
    ffnn->connectFFNN();

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

    ffnn->assignVariationalParameters();
    ffnn->addCrossSecondDerivativeSubstrate();

    checkDerivatives(ffnn, TINY);

    delete ffnn;

    return 0;
}
