#include "PrintUtilities.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <cmath>


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
    string emptySpaceForBeta = "     ";
    string emptySapceForActivationFunctionId = "   ";
    string emptySpaceAfterBeta = "  ";
    string emptySpaceAfterActivationFunction = "     ";

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
                            if (feeder->getBeta(b) >= 0.) cout << "+";
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



void printFFNNValues(FeedForwardNeuralNetwork * ffnn)
{
    using namespace std;

    cout.precision(2);
    cout << fixed;

    string emptySpaceForValue = "     ";
    string emptySpaceBetweenProtovalueAndValue = "    ";
    string emptySpaceAfterValue = "    ";

    int maxNUnits = 0;
    for (int l=0; l<ffnn->getNLayers(); ++l){
        if (ffnn->getLayer(l)->getNUnits() > maxNUnits){
            maxNUnits = ffnn->getLayer(l)->getNUnits();
        }
    }

    for (int u=0; u<maxNUnits; ++u){
        for (int l=0; l<ffnn->getNLayers(); ++l){
            if (u < ffnn->getLayerSize(l)){
                if (ffnn->getLayer(l)->getUnit(u)->getProtoValue() >= 0.) cout << "+";
                cout << ffnn->getLayer(l)->getUnit(u)->getProtoValue() << " -> ";
                if (ffnn->getLayer(l)->getUnit(u)->getValue() >= 0.) cout << "+";
                cout << ffnn->getLayer(l)->getUnit(u)->getValue() << "    ";
            } else {
                cout << emptySpaceForValue << emptySpaceBetweenProtovalueAndValue << emptySpaceForValue << emptySpaceAfterValue;
            }
        }
        cout << endl;
    }
}



void writePlotFile(FeedForwardNeuralNetwork * ffnn, const double * base_input, const int &input_i, const int &output_i, const double &min, const double &max, const int &npoints, std::string what, std::string filename, const double &xscale, const double &yscale, const double &xshift, const double &yshift){
    using namespace std;

    const double delta = (max-min)/(npoints-1);

    // compute the input points
    double * x = new double[npoints];
    x[0] = min;
    for (int i=1; i<npoints; ++i){
        x[i] = x[i-1] + delta;
    }

    // allocate the output variables
    double * v = new double[npoints];      // NN output value

    // compute the values
    const int ninput = ffnn->getNInput();
    double * input = new double[ninput];
    for (int i=0; i<ninput; ++i) input[i] = (base_input[i] + xshift) * xscale;
    for (int i=0; i<npoints; ++i){
        input[input_i] = (x[i] + xshift) * xscale;
        ffnn->setInput(input);
        ffnn->FFPropagate();

        if (what == "getOutput"){
            v[i] = ffnn->getOutput(output_i) / yscale - yshift;
        } else if (what == "getFirstDerivative"){
            v[i] = ffnn->getFirstDerivative(output_i, input_i) / yscale * xscale;
        } else if (what == "getSecondDerivative"){
            v[i] = ffnn->getSecondDerivative(output_i, input_i) / pow(yscale, 2) * pow(xscale, 2);
        } else if (what == "getVariationalFirstDerivative"){
            v[i] = ffnn->getVariationalFirstDerivative(output_i, input_i) / yscale;
        } else {
            throw std::invalid_argument( "writePlotFile(): the parameter 'what' was not valid" );
        }
    }

    // write the results on files
    ofstream vFile;
    vFile.open(filename);
    for (int i=0; i<npoints; ++i){
        vFile << x[i] << "    " << v[i] << endl;
    }
    vFile.close();

    delete[] x;
    delete[] v;
    delete[] input;
}
