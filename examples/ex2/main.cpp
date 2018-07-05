#include <iostream>
#include <cmath>
#include <fstream>

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"


int main() {
    using namespace std;



    cout << "Let's start by creating a Feed Forward Artificial Neural Network (FFANN)" << endl;
    cout << "========================================================================" << endl;
    cin.ignore();

    cout << "We generate a FFANN with 4 layers and 3, 4, 5, 2 units respectively." << endl;
    cin.ignore();

    // NON I/O CODE
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 4, 2);
    ffnn->pushHiddenLayer(5);
    //

    cout << "With activation functions, but not yet connected, it looks like this:" << endl;
    cin.ignore();
    printFFNNStructure(ffnn, true, 0);
    cin.ignore();

    cout << "Let's connect the layers." << endl;
    cin.ignore();
    ffnn->connectFFNN();
    cout << "Now the network looks like this:" << endl;
    cin.ignore();
    printFFNNStructure(ffnn, true, 0);
    cin.ignore();
    cout << "Obviously the neural units now have a ray feeder, the default." << endl;
    cin.ignore();
    cout << "A ray multiplies the output values of the previous layers' units with a set of weights and sums up the results." << endl;
    cin.ignore();
    cout << "From this sum the activation function calculates the output value of the unit." << endl << endl;
    cin.ignore();
    cout << "If we only look at the weights (betas) and activation functions of neural (includes output) units, it looks like this:" << endl;
    cin.ignore();
    printFFNNStructureWithBeta(ffnn);
    cout << "Here offset units were included for illustration. The first beta of each neural unit corresponds to the previous offset unit.";
    cout << endl;
    cin.ignore();

    cout << "Reading a specific beta" << endl;
    cout << "=======================" << endl;
    cin.ignore();
    cout << "Suppose we want to access the beta that connects the 4th network unit (i.e. any type) of the 2nd layer with the 1st neural unit of the 3rd layer (or 2nd hidden neural layer).";
    cin.ignore();

    cout << "A we have seen above, a neural unit is connected to the previous layer's network units through a feeder. It provides the unit's input.";
    cin.ignore();
    cout << "The feeder, among other things that we will see in other examples, contains the variational parameters we are interested in (the beta).";
    cin.ignore();
    cout << "As first thing, let's get the feeder for the 1st neural unit of the 3rd layer." << endl;
    cin.ignore();

    // access the 1st neural unit of the 2nd hidden layer
    NNUnit * nnu = ffnn->getNNLayer(1)->getNNUnit(0);
    // require its feeder
    NetworkUnitFeederInterface * feeder = nnu->getFeeder();
    cout << "Done! The feeder memory address is " << feeder << endl;
    cin.ignore();

    cout << "How many variational parameters (beta) does the feeder contain?";
    cin.ignore();
    cout << "NBeta = " << feeder->getNBeta() << endl;
    cin.ignore();

    cout << "These betas correspond to the units of the 2nd layer. We are interested in the beta related to the 4th network unit.";
    cin.ignore();
    cout << "Its value is " << feeder->getBeta(3) << endl << endl << endl;
    cin.ignore();



    cout << "Setting a specific beta" << endl;
    cout << "=======================" << endl;
    cin.ignore();
    cout << "Now we want to change the value of the beta we just read.";
    cin.ignore();

    cout << "We set its value to +8.88.";
    cin.ignore();
    feeder->setBeta(3, 8.88);

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

    cout << "We can change, for example, the 4th beta into -4.44";
    cin.ignore();
    ffnn->setBeta(3, -4.44);
    cout << "Done! The betas are now:" << endl;
    for (int i=0; i<ffnn->getNBeta(); ++i){
        cout << ffnn->getBeta(i) << "    ";
    }



    cout << endl << endl;

    delete ffnn;

    return 0;
}
