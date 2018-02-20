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

    cout << "For computing the derivative in respect to the beta, we need to set the corresponding substrate.";
    cin.ignore();

    // NON I/O CODE
    ffnn->addVariationalFirstDerivativeSubstrate();
    //

    cout << "Done! We informed all the units that they need to compute the variational first derivatives.";
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

    cout << "There are " << ffnn->getNBeta() << " variational parameters, this means that each output will have such number of derivatives. These are:";
    cin.ignore();

    cout << "1st output (unit 2 of the output layer): " << endl;
    for (int i1=0; i1<ffnn->getNBeta()/10; ++i1){
        for (int i2=0; i2<10; ++i2){
            if (ffnn->getVariationalFirstDerivative(0, i1*10 + i2) >= 0.) cout << "+";
            cout << ffnn->getVariationalFirstDerivative(0, i1*10 + i2) << "    ";
        }
        cout << endl;
    }
    cin.ignore();

    cout << "2nd output (unit 3 of the output layer): " << endl;
    for (int i1=0; i1<ffnn->getNBeta()/10; ++i1){
        for (int i2=0; i2<10; ++i2){
            if (ffnn->getVariationalFirstDerivative(1, i1*10 + i2) >= 0.) cout << "+";
            cout << ffnn->getVariationalFirstDerivative(1, i1*10 + i2) << "    ";
        }
        cout << endl;
    }
    cin.ignore();

    cout << "3rd output (unit 4 of the output layer): " << endl;
    for (int i1=0; i1<ffnn->getNBeta()/10; ++i1){
        for (int i2=0; i2<10; ++i2){
            if (ffnn->getVariationalFirstDerivative(2, i1*10 + i2) >= 0.) cout << "+";
            cout << ffnn->getVariationalFirstDerivative(2, i1*10 + i2) << "    ";
        }
        cout << endl;
    }


    cout << endl << endl;
    return 0;
}
