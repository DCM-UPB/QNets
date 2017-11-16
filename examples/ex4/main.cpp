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
    cout << endl << endl;
    cin.ignore();



    cout << "Connect the FFNN" << endl;
    cout << "================" << endl;
    cin.ignore();

    cout << "Connecting a FFNN is a necessary step before making any computation.";
    cin.ignore();

    // NON I/O CODE
    ffnn->connectFFNN();
    //

    cin.ignore();
    cout << endl << endl;



    cout << "Set the input" << endl;
    cout << "=============" << endl;
    cin.ignore();

    cout << "We want to set the input.";
    cin.ignore();

    cout << "The value of the first units of each layer is set to 1, and cannot be modified, as it is an offset. Therefore the input is 2-dimensional, even though the input layer has 3 units.";
    cin.ignore();

    int ninput = 2;
    double * input = new double[ninput];
    input[0] = -3.;
    input[1] = 0.5;
    cout << "The input we want to set is: " << input[0] << "    " << input[1];
    cin.ignore();

    cout << "Before setting the input, the NN values look like this:";
    cin.ignore();

    printFFNNValues(ffnn);
    cin.ignore();

    cout << "For each unit there are two values connected by an arrow. The first is the protovalue, i.e. the value before the application of the activation function. The second one, the value, is the result of the application of the activation function to the protovalue.";
    cin.ignore();

    cout << "When we set the input, we set the protovalues of the input layer. The values are computed when the values are propagated, as we will see.";
    cin.ignore();

    cout << "Now we will set the input...";
    cin.ignore();

    // NON I/O CODE
    ffnn->setInput(ninput, input);
    //
    cout << "Done! Now the NN values look like this:";
    cin.ignore();

    printFFNNValues(ffnn);

    cin.ignore();
    cout << endl << endl;



    cout << "Propagate" << endl;
    cout << "=========" << endl;
    cin.ignore();

    cout << "Now that we have set the input, we can find the output. For doing that we will propagate the values.";
    cin.ignore();

    // NON I/O CODE
    ffnn->FFPropagate();
    //

    cout << "Done! Now the NN values look like this:";
    cin.ignore();

    printFFNNValues(ffnn);
    cout << endl;
    cin.ignore();

    cout << "The output is 1-dimensional, as the first unit of the output layer is an offset." << endl << "Its value is ";
    cout << ffnn->getOutput(1);
    cout << endl;




    cout << endl << endl;
    return 0;
}
