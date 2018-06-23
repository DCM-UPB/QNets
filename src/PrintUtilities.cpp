#include "PrintUtilities.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <string>

void printFFNNStructure(FeedForwardNeuralNetwork * ffnn, const bool &drop_params, const int &drop_member_lvl)
{
    using namespace std;

    int maxLayerSize = 0;
    size_t maxStringLength[ffnn->getNLayers()];

    std::string stringCode = "";

    for (int l=0; l<ffnn->getNLayers(); ++l)
        {
            if (ffnn->getLayerSize(l) > maxLayerSize)
                {
                    maxLayerSize = ffnn->getLayerSize(l);
                }

            maxStringLength[l] = 0;
            for (int u = 0; u<ffnn->getLayerSize(l); ++u)
                {
                    stringCode = ffnn->getLayer(l)->getUnit(u)->getTreeCode();
                    if (drop_member_lvl > 0) stringCode = dropMembers(stringCode, drop_member_lvl);
                    if (drop_params) stringCode = dropParams(stringCode);

                    if (stringCode.length() > maxStringLength[l])
                        {
                            maxStringLength[l] = stringCode.length();
                        }
                }
            stringCode = ffnn->getLayer(l)->getIdCode() + " " + readParamValue(ffnn->getLayer(l)->getParams(), "nunits") + "U"; // print layer identifiers
            cout << stringCode << string(maxStringLength[l]-stringCode.length()+8, ' ');
        }
    cout << endl << endl;

    for (int u=0; u<maxLayerSize; ++u)
        {
            for (int l=0; l<ffnn->getNLayers(); ++l)
                {
                    if (ffnn->getLayerSize(l) > u)
                        {
                            stringCode = ffnn->getLayer(l)->getUnit(u)->getTreeCode();
                            if (drop_member_lvl > 0) stringCode = dropMembers(stringCode, drop_member_lvl);
                            if (drop_params) stringCode = dropParams(stringCode);

                            cout << stringCode;
                            cout << string(maxStringLength[l]-stringCode.length(), ' ');
                        }
                    else
                        {
                            cout << string(maxStringLength[l], ' ');
                        }
                    cout << "        ";
                }
            cout << endl;
        }
}



void printFFNNStructureWithBeta(FeedForwardNeuralNetwork * ffnn)
{
    using namespace std;

    // variables used inside the loops
    const int nlayers = ffnn->getNLayers();
    int maxNUnits, index;
    int * maxNBeta;
    size_t maxIdLength[ffnn->getNLayers()];
    NetworkUnitFeederInterface * feeder;
    NetworkUnitFeederInterface ** feeders;
    string * ids;

    const string emptySpaceForBeta = "     ";
    const string emptySpaceAfterBeta = "  ";
    const string emptySpaceAfterActivationFunction = "       ";

    cout.precision(2);
    cout << fixed;

    // max number of units over all layers
    maxNUnits = 0;
    for (int l=0; l<nlayers; ++l){
        if (ffnn->getLayer(l)->getNUnits() > maxNUnits){
            maxNUnits = ffnn->getLayer(l)->getNUnits();
        }
        maxIdLength[l] = 0;
    }

    // max number of beta (i.e. variational parameters) for the units with index u over all layers
    maxNBeta = new int[maxNUnits];
    feeders = new NetworkUnitFeederInterface*[nlayers*maxNUnits];
    ids = new string[nlayers*maxNUnits];
    for (int u=0; u<maxNUnits; ++u){
        maxNBeta[u] = 0;
        for (int l=0; l<nlayers; ++l){
            index = u*nlayers+l;
            feeder = NULL;

            if (u < ffnn->getLayer(l)->getNUnits()) {
                if (FedNetworkUnit * fnu = dynamic_cast<FedNetworkUnit *>(ffnn->getLayer(l)->getUnit(u))) {
                    feeder = fnu->getFeeder();
                    if (feeder){
                        if (feeder->getNVariationalParameters() > maxNBeta[u]){
                            maxNBeta[u] = feeder->getNVariationalParameters();
                        }
                    }
                }

                if (NNUnit * nnu = dynamic_cast<NNUnit *>(ffnn->getLayer(l)->getUnit(u))) {
                    ids[index] = " " + nnu->getActivationFunction()->getIdCode() + " ";
                }
                else {
                    ids[index] = "(" + ffnn->getLayer(l)->getUnit(u)->getIdCode() + ")"; // put placeholder unit identifiers in brackets
                }

                if (ids[index].length() > maxIdLength[l]){
                    maxIdLength[l] = ids[index].length();
                }
            }
            feeders[index] = feeder;
        }
    }

    // now we are ready to print
    for (int u=0; u<maxNUnits; ++u) {
        for (int b=0; b<max(1, maxNBeta[u]); ++b) {
            for (int l=0; l<nlayers; ++l) {
                index = u*nlayers+l;

                if (u < ffnn->getLayer(l)->getNUnits() && feeders[index]) {
                    if (b < feeders[index]->getNBeta()) {
                        if (feeders[index]->getBeta(b) >= 0.) cout << "+";
                        cout << feeders[index]->getBeta(b);
                    }
                    else {
                        cout << emptySpaceForBeta;
                    }
                    cout << emptySpaceAfterBeta;
                }
                else {
                    cout << emptySpaceForBeta << emptySpaceAfterBeta;
                }

                if (b==0) {
                    cout << ids[index];
                    cout << string(maxIdLength[l]-ids[index].length(), ' ');
                }
                else {
                    cout << string(maxIdLength[l], ' ');
                }
                cout << emptySpaceAfterActivationFunction;
            }

            cout << endl << endl << endl;
        }
    }


    delete[] maxNBeta;
    delete[] feeders;
    delete[] ids;
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
