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

    cout << "We generate a FFANN with 4 layers and 3, 4, 5, 4 units respectively" << endl;
    cin.ignore();

    // NON I/O CODE
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 4, 4);
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

    cout << endl << endl << endl;



    cout << "Add derivatives substrates" << endl;
    cout << "==========================" << endl;
    cin.ignore();

    cout << "For computing the NN derivatives, we must add the derivatives substrates.";
    cin.ignore();

    // NON I/O CODE
    ffnn->addFirstDerivativeSubstrate();
    ffnn->addSecondDerivativeSubstrate();
    //

    cout << "Done! We informed all the units that they need to compute also first and second derivatives.";
    cin.ignore();
    cout << endl << endl;



    cout << "Set the input" << endl;
    cout << "=============" << endl;
    cin.ignore();

    int ninput = 2;
    double * input = new double[ninput];
    input[0] = -3.;
    input[1] = 0.5;
    cout << "The input we want to set is: " << input[0] << "    " << input[1];
    cin.ignore();

    // NON I/O CODE
    ffnn->setInput(input);
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

    cout << "The output values are ";
    cout << ffnn->getOutput(0) << "    " << ffnn->getOutput(1) << "    " << ffnn->getOutput(2) << endl;
    cin.ignore();

    cout << "The first derivatives in respect to the 1st, 2nd, and 3rd input value are:";
    cin.ignore();
    cout << "1st output (unit 2 of the output layer): " << ffnn->getFirstDerivative(0, 0) << "    " << ffnn->getFirstDerivative(0, 1) << "    " << ffnn->getFirstDerivative(0, 2);
    cin.ignore();
    cout << "2nd output (unit 3 of the output layer): " << ffnn->getFirstDerivative(1, 0) << "    " << ffnn->getFirstDerivative(1, 1) << "    " << ffnn->getFirstDerivative(1, 2);
    cin.ignore();
    cout << "3rd output (unit 4 of the output layer): " << ffnn->getFirstDerivative(2, 0) << "    " << ffnn->getFirstDerivative(2, 1) << "    " << ffnn->getFirstDerivative(2, 2) << endl;
    cin.ignore();

    cout << "The second derivatives in respect to the 1st, 2nd, and 3rd input value are:";
    cin.ignore();
    cout << "1st output (unit 2 of the output layer): " << ffnn->getSecondDerivative(0, 0) << "    " << ffnn->getSecondDerivative(0, 1) << "    " << ffnn->getSecondDerivative(0, 2);
    cin.ignore();
    cout << "2nd output (unit 3 of the output layer): " << ffnn->getSecondDerivative(1, 0) << "    " << ffnn->getSecondDerivative(1, 1) << "    " << ffnn->getSecondDerivative(1, 2);
    cin.ignore();
    cout << "3rd output (unit 4 of the output layer): " << ffnn->getSecondDerivative(2, 0) << "    " << ffnn->getSecondDerivative(2, 1) << "    " << ffnn->getSecondDerivative(2, 2);


    cout << endl << endl;
    return 0;
}
