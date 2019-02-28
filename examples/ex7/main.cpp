#include <cmath>
#include <fstream>
#include <iostream>

#include "ffnn/io/PrintUtilities.hpp"
#include "ffnn/net/FeedForwardNeuralNetwork.hpp"



int main() {
    using namespace std;



    cout << "Let's start by creating a Feed Forward Artificial Neural Network (FFANN)" << endl;
    cout << "========================================================================" << endl;
    cin.ignore();

    cout << "We generate a FFANN with 4 layers and 2, 4, 5, 2 units respectively. This means that will have only 1 input and 1 output." << endl;
    cin.ignore();

    // NON I/O CODE
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(2, 4, 2);
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



    cout << "Write plot files" << endl;
    cout << "================" << endl;

    auto * base_input = new double[ffnn->getNInput()]; // no need to set it, since it is 1-dim
    const int input_i = 0;
    const int output_i = 0;
    const double min = -5;
    const double max = 5;
    const int npoints = 200;

    cout << "We will compute the NN values and derivatives for input values varying in the range [" << min << ":" << max << "]. We will use a grid of " << npoints << " points." << endl;

    writePlotFile(ffnn, base_input, input_i, output_i, min, max, npoints, "getOutput", "v.txt");
    writePlotFile(ffnn, base_input, input_i, output_i, min, max, npoints, "getFirstDerivative", "v1d.txt");
    writePlotFile(ffnn, base_input, input_i, output_i, min, max, npoints, "getSecondDerivative", "v2d.txt");

    cout << "Done! In the files v.txt, v1d.txt, and v2d.txt we stored the values, and you can use any software you like to plot them (perhaps gnuplot?)." << endl << endl;

    cout << "Note that the executable was run within the build/examples/ directory, so you have to look there to find the mentioned output files." << endl;

    return 0;
}
