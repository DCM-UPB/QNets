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
    cin.ignore();
    
    const double minX = -1.;
    const double maxX = 4.;
    const int nPoints = 1000;
    const double delta = (maxX-minX)/nPoints;
    cout << "We will compute the NN values and derivatives for input values varying in the range [" << minX << ":" << maxX << "]. We will use a grid of " << nPoints << " points." << endl;
    cin.ignore();
    
    // compute the input points
    double * x = new double[nPoints];
    x[0] = minX;
    for (int i=1; i<nPoints; ++i){
       x[i] = x[i-1] + delta;
    }
    
    // allocate the output variables
    double * v = new double[nPoints];      // NN output value
    double * v1d = new double[nPoints];    // NN 1st derivative value
    double * v2d = new double[nPoints];    // NN 2nd derivative value
    
    // compute the values
    const int ninput = 1;
    double * input = new double[ninput];
    for (int i=0; i<nPoints; ++i){
       input[0] = x[i];
       ffnn->setInput(ninput, input);
       ffnn->FFPropagate();
       v[i] = ffnn->getOutput(1);
       v1d[i] = ffnn->getFirstDerivative(1, 0);
       v2d[i] = ffnn->getSecondDerivative(1, 0);
       //cout << x[i] << "    " << v[i] << "    " << v1d[i] << "    " << v2d[i] << endl;
    }
    
    // write the results on files
    ofstream vFile;
    ofstream v1dFile;
    ofstream v2dFile;
    vFile.open("v.txt");
    v1dFile.open("v1d.txt");
    v2dFile.open("v2d.txt");
    for (int i=0; i<nPoints; ++i){
       vFile << x[i] << "    " << v[i] << endl;
       v1dFile << x[i] << "    " << v1d[i] << endl;
       v2dFile << x[i] << "    " << v2d[i] << endl;
    }
    vFile.close();
    v1dFile.close();
    v2dFile.close();
    
    cout << "Done! In the files v.txt, v1d.txt, and v2d.txt we stored the values, and you can use any software you like to plot them (perhaps gnuplot?).";


    cout << endl << endl;
    return 0;
}
