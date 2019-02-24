#include <iostream>
#include <assert.h>
#include <cmath>

#include "ffnn/net/FeedForwardNeuralNetwork.hpp"
#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/io/PrintUtilities.hpp"



int main(){
    using namespace std;

    // --- create a FFNN that will look like this (lgs is a sigmoid, gss a gaussian):
    // lgs( b4 + b5*lgs( b0 + b1*x1 ) + b6*gss( b2 + b3*x2) )
    const double beta[7] = {-1.18, 0.37, -0.42, -0.86, 1.11, 2.018, -3.14};
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(2, 3, 2);
    ffnn->getNNLayer(0)->getNNUnit(1)->setActivationFunction(std_actf::provideActivationFunction("GSS"));
    ffnn->connectFFNN();
    ffnn->setBeta(beta);
    ffnn->assignVariationalParameters();


    // --- create two set of variables that will be compared.

    // the first variable set will be used with FFPropagate and then all the get...
    double * out_1 = new double[ffnn->getNOutput()];

    double ** d1_1 = new double*[ffnn->getNOutput()];
    for (int i=0; i<ffnn->getNOutput(); ++i){
        d1_1[i] = new double[ffnn->getNInput()];
        for (int j=0; j<ffnn->getNInput(); ++j) d1_1[i][j] = -6.66;
    }

    double ** d2_1 = new double*[ffnn->getNOutput()];
    for (int i=0; i<ffnn->getNOutput(); ++i){
        d2_1[i] = new double[ffnn->getNInput()];
        for (int j=0; j<ffnn->getNInput(); ++j) d2_1[i][j] = -6.66;
    }

    double ** vd1_1 = new double*[ffnn->getNOutput()];
    for (int i=0; i<ffnn->getNOutput(); ++i){
        vd1_1[i] = new double[ffnn->getNVariationalParameters()];
        for (int j=0; j<ffnn->getNVariationalParameters(); ++j) vd1_1[i][j] = -6.66;
    }

    // the second variable set will be used with the evaluate function
    double * out_2 = new double[ffnn->getNOutput()];

    double ** d1_2 = new double*[ffnn->getNOutput()];
    for (int i=0; i<ffnn->getNOutput(); ++i){
        d1_2[i] = new double[ffnn->getNInput()];
        for (int j=0; j<ffnn->getNInput(); ++j) d1_2[i][j] = -6.66;
    }

    double ** d2_2 = new double*[ffnn->getNOutput()];
    for (int i=0; i<ffnn->getNOutput(); ++i){
        d2_2[i] = new double[ffnn->getNInput()];
        for (int j=0; j<ffnn->getNInput(); ++j) d2_2[i][j] = -6.66;
    }

    double ** vd1_2 = new double*[ffnn->getNOutput()];
    for (int i=0; i<ffnn->getNOutput(); ++i){
        vd1_2[i] = new double[ffnn->getNVariationalParameters()];
        for (int j=0; j<ffnn->getNVariationalParameters(); ++j) vd1_2[i][j] = -6.66;
    }





    // --- make a first computation when there is no substrate

    double input[1] = {0.666};

    // FFPropagate
    ffnn->setInput(input);
    ffnn->FFPropagate();
    ffnn->getOutput(out_1);

    // evaluate
    ffnn->evaluate(input, out_2, d1_2, d2_2, vd1_2);

    // verify that the output is the same in the two cases
    for (int i=0; i<ffnn->getNOutput(); ++i){
        assert( out_1[i] == out_2[i] );
    }

    // verify that the derivatives have not been modified
    for (int i=0; i<ffnn->getNOutput(); ++i){
        for (int j=0; j<ffnn->getNInput(); ++j){
            assert( d1_2[i][j] == -6.66 );
            assert( d2_2[i][j] == -6.66 );
        }
        for (int j=0; j<ffnn->getNVariationalParameters(); ++j){
            assert( vd1_2[i][j] == -6.66 );
        }
    }





    // --- make a computation with all the derivative substrates

    input[0] = -0.666;
    ffnn->addFirstDerivativeSubstrate();
    ffnn->addSecondDerivativeSubstrate();
    ffnn->addVariationalFirstDerivativeSubstrate();

    // FFPropagate
    ffnn->setInput(input);
    ffnn->FFPropagate();
    ffnn->getOutput(out_1);
    ffnn->getFirstDerivative(d1_1);
    ffnn->getSecondDerivative(d2_1);
    ffnn->getVariationalFirstDerivative(vd1_1);

    // evaluate
    ffnn->evaluate(input, out_2, d1_2, d2_2, vd1_2);

    // verify that the output and all the derivatives are the same in the two cases
    for (int i=0; i<ffnn->getNOutput(); ++i){
        assert( out_1[i] == out_2[i] );
        for (int j=0; j<ffnn->getNInput(); ++j){
            assert( d1_1[i][j] == d1_2[i][j] );
            assert( d2_1[i][j] == d2_2[i][j] );
        }
        for (int j=0; j<ffnn->getNVariationalParameters(); ++j){
            assert( vd1_1[i][j] == vd1_2[i][j] );
        }
    }




    // --- free resources
    for (int i=0; i<ffnn->getNOutput(); ++i) delete[] vd1_2[i];
    delete[] vd1_2;

    for (int i=0; i<ffnn->getNOutput(); ++i) delete[] d2_2[i];
    delete[] d2_2;

    for (int i=0; i<ffnn->getNOutput(); ++i) delete[] d1_2[i];
    delete[] d1_2;

    delete[] out_2;

    for (int i=0; i<ffnn->getNOutput(); ++i) delete[] vd1_1[i];
    delete[] vd1_1;

    for (int i=0; i<ffnn->getNOutput(); ++i) delete[] d2_1[i];
    delete[] d2_1;

    for (int i=0; i<ffnn->getNOutput(); ++i) delete[] d1_1[i];
    delete[] d1_1;

    delete[] out_1;

    delete ffnn;

    return 0;
}
