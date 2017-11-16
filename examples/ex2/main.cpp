#include <iostream>
#include <cmath>
#include <fstream>

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"


int main() {
    using namespace std;



    cout << "Let's start by creating a Feed Forward Artificial Neural Netowrk (FFANN)" << endl;
    cout << "========================================================================" << endl;
    cin.ignore();

    cout << "We generate a FFANN with 4 layers and 3, 4, 5, 2 units respectively" << endl;
    cin.ignore();

    // NON I/O CODE
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 4, 2);
    ffnn->pushHiddenLayer(5);
    //

    cout << "Graphically it looks like this" << endl;
    cin.ignore();
    printFFNNStructure(ffnn);
    cin.ignore();

    ffnn->connectFFNN();

    cout << "If we look at the connections and variational parameters, it looks like this" << endl;
    cin.ignore();
    printFFNNStructureWithBeta(ffnn);
    cout << endl;
    cin.ignore();



    cout << "Reading a specific beta" << endl;
    cout << "=======================" << endl;
    cin.ignore();
    cout << "Suppose we want to access the beta that connects the 3rd unit of the 2nd layer with the 5th unit of the 3rd layer.";
    cin.ignore();

    cout << "Each unit connected to the previous layer units has a feeder attached, which is responsible for providing an input to the unit.";
    cin.ignore();
    cout << "The feeder, among other things that we will see in other examples, contains the variational parameters we are interested in (the beta).";
    cin.ignore();
    cout << "As first thing, let's get the feeder for the 5th unit of the 3rd layer." << endl;
    cin.ignore();

    // access the 5th unit of the 3rd layer
    NNUnit * nnu = ffnn->getLayer(2)->getUnit(4);
    // require its feeder
    NNUnitFeederInterface * feeder = nnu->getFeeder();
    cout << "Done! The feeder memory address is " << feeder << endl;
    cin.ignore();

    cout << "How many variational parameters (beta) does the feeder contains?";
    cin.ignore();
    cout << "NBeta = " << feeder->getNBeta() << endl;
    cin.ignore();

    cout << "These betas correspond to the units of the 2nd layer. We are interested in the beta related to the 3rd unit.";
    cin.ignore();
    cout << "Its value is " << feeder->getBeta(2) << endl << endl << endl;
    cin.ignore();



    cout << "Setting a specific beta" << endl;
    cout << "=======================" << endl;
    cin.ignore();
    cout << "Now we want to change the value of the beta we just read.";
    cin.ignore();

    cout << "We set its value to +8.88.";
    cin.ignore();
    feeder->setBeta(2, 8.88);

    cout << "If we now look at the NN we see that we set it correctly" << endl;
    cin.ignore();
    printFFNNStructureWithBeta(ffnn);
    cout << endl << endl;
    cin.ignore();



    cout << "Accessing all the variational parameters at once" << endl;
    cout << "================================================" << endl;
    cin.ignore();
    cout << "If we are not interested at manipulating the single connections of the NN but we want to see it from a more abstract point of view, we are just interested at all its variational parameters, without knowing their specific role.";
    cin.ignore();

    cout << "The NN has " << ffnn->getNBeta() << " variational parameters (beta).";
    cin.ignore();
    cout << "The betas are: ";
    cin.ignore();

    for (int i=0; i<ffnn->getNBeta(); ++i){
        cout << ffnn->getBeta(i) << "    ";
    }
    cin.ignore();

    cout << "We can change, for example, the 3rd beta into -4.44";
    cin.ignore();
    ffnn->setBeta(2, -4.44);
    cout << "Done! The betas are now:" << endl;
    for (int i=0; i<ffnn->getNBeta(); ++i){
        cout << ffnn->getBeta(i) << "    ";
    }



    cout << endl << endl;


    return 0;
}
