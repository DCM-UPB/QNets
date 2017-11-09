#include "PrintUtilities.hpp"

#include <iostream>


void printFFANNStructure(FeedForwardNeuralNetwork * ffann)
{
    using namespace std;

    int maxLayerSize = 0;
    for (int l=0; l<ffann->getNLayers(); ++l)
    {
        if (ffann->getLayerSize(l) > maxLayerSize)
        {
            maxLayerSize = ffann->getLayerSize(l);
        }
    }

    for (int r=0; r<maxLayerSize; ++r)
    {
        for (int c=0; c<ffann->getNLayers(); ++c)
        {
            if (ffann->getLayerSize(c) > r)
            {
                cout << ffann->getLayer(c)->getUnit(r)->getActivationFunction()->getIdCode();
            }
            else
            {
                cout << "   ";
            }
            cout << "    ";
        }
        cout << endl;
    }
}
