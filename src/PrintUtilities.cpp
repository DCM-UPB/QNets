#include "PrintUtilities.hpp"

#include <iostream>
#include <iomanip>
#include <string>


void printFFNNStructure(FeedForwardNeuralNetwork * ffnn)
{
    using namespace std;

    int maxLayerSize = 0;
    for (int l=0; l<ffnn->getNLayers(); ++l)
    {
        if (ffnn->getLayerSize(l) > maxLayerSize)
        {
            maxLayerSize = ffnn->getLayerSize(l);
        }
    }

    for (int r=0; r<maxLayerSize; ++r)
    {
        for (int c=0; c<ffnn->getNLayers(); ++c)
        {
            if (ffnn->getLayerSize(c) > r)
            {
                cout << ffnn->getLayer(c)->getUnit(r)->getActivationFunction()->getIdCode();
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



void printFFNNStructureWithBeta(FeedForwardNeuralNetwork * ffnn)
{
    using namespace std;

    cout.precision(2);
    cout << fixed;
    std::string emptySpaceForBeta = "     ";
    std::string emptySapceForActivationFunctionId = "   ";
    std::string emptySpaceAfterBeta = "  ";
    std::string emptySpaceAfterActivationFunction = "     ";

    // max number of units over all layers
    int maxNUnits = 0;
    for (int l=0; l<ffnn->getNLayers(); ++l){
        if (ffnn->getLayer(l)->getNUnits() > maxNUnits){
            maxNUnits = ffnn->getLayer(l)->getNUnits();
        }
    }

    // variables used inside the loop
    int maxNBeta;
    NNUnitFeederInterface * feeder;

    for (int u=0; u<maxNUnits; ++u){
        // max number of beta (i.e. variational parameters) for the units u over all layers
        maxNBeta = 0;
        for (int l=0; l<ffnn->getNLayers(); ++l){
            if (u < ffnn->getLayerSize(l)) {
                feeder = ffnn->getLayer(l)->getUnit(u)->getFeeder();
                if (feeder){
                    if (feeder->getNVariationalParameters() > maxNBeta){
                        maxNBeta = feeder->getNVariationalParameters();
                    }
                }
            }
        }

        // here are considered the offest units, which are not connected to the previous layers
        if (maxNBeta==0){
            for (int l=0; l<ffnn->getNLayers(); ++l){
                cout << emptySpaceForBeta << emptySpaceAfterBeta;
                if (u < ffnn->getLayerSize(l)){
                    cout << ffnn->getLayer(l)->getUnit(u)->getActivationFunction()->getIdCode();
                } else {
                    cout << emptySapceForActivationFunctionId;
                }
                cout << emptySpaceAfterActivationFunction;
            }
            cout << endl;
        }

        // here are considered all the other units, (typically) connected to the previus layers
        for (int b=0; b<maxNBeta; ++b){
            for (int l=0; l<ffnn->getNLayers(); ++l){
                if (u < ffnn->getLayerSize(l)){
                    feeder = ffnn->getLayer(l)->getUnit(u)->getFeeder();
                    if (u < ffnn->getLayer(l)->getNUnits() && feeder){
                        if (b < feeder->getNVariationalParameters()){
                            if (feeder->getBeta(b)>0) cout << "+";
                            cout << feeder->getBeta(b);
                        } else {
                            cout << emptySpaceForBeta;
                        }
                        cout << emptySpaceAfterBeta;
                        if (b==0){
                            cout << ffnn->getLayer(l)->getUnit(u)->getActivationFunction()->getIdCode();
                        } else {
                            cout << emptySapceForActivationFunctionId;
                        }
                        cout << emptySpaceAfterActivationFunction;
                    } else {
                        cout << emptySpaceForBeta << emptySpaceAfterBeta;
                        if (b==0) {
                            cout << ffnn->getLayer(l)->getUnit(u)->getActivationFunction()->getIdCode();
                        } else {
                            cout << emptySapceForActivationFunctionId;
                        }
                        cout << emptySpaceAfterActivationFunction;
                    }
                } else {
                    cout << emptySpaceForBeta << emptySpaceAfterBeta << emptySapceForActivationFunctionId << emptySpaceAfterActivationFunction;
                }
            }
            cout << endl;
        }
        cout << endl;
    }
}
