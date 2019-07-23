#include <cmath>
#include <fstream>
#include <iostream>

#include "qnets/poly/io/PrintUtilities.hpp"


int main()
{
    using namespace std;

    auto * ffnn = new FeedForwardNeuralNetwork("stored_ffnn.txt");

    cout << "The FFNN read from the file looks like this:" << endl << endl;

    printFFNNStructure(ffnn, true, 0);

    cout << endl << "Note that the stored FFNN was not connected. Hence, there are no feeders or betas." << endl;
    cin.ignore();

    cout << "Now we connect the network and store it into a new file 'connected_ffnn.txt'." << endl;
    //NON I/O CODE
    ffnn->connectFFNN();
    ffnn->storeOnFile("connected_ffnn.txt");
    //
    cout << "Done." << endl;
    cin.ignore();

    cout << "Now we assign all betas to variational parameter indices and store it into a new file 'vpar_ffnn.txt'." << endl;
    //NON I/O CODE
    ffnn->assignVariationalParameters();
    ffnn->storeOnFile("vpar_ffnn.txt");
    //
    cout << "Done." << endl;
    cin.ignore();

    cout << "We may also choose to store the FFNN without beta weights into 'nobetas_ffnn.txt'. " << endl;
    //NON I/O CODE
    ffnn->storeOnFile("nobetas_ffnn.txt", false);
    //
    cout << "Done." << endl;
    cin.ignore();

    cout << "Finally we add 1st, 2nd and variational derivative substrates and store the ffnn without betas into a new file 'substrate_ffnn.txt' and with betas into 'substrate_ffnn_wbetas.txt'." << endl;
    //NON I/O CODE
    ffnn->addFirstDerivativeSubstrate();
    ffnn->addSecondDerivativeSubstrate();
    ffnn->addVariationalFirstDerivativeSubstrate();
    ffnn->storeOnFile("substrate_ffnn.txt", false);
    ffnn->storeOnFile("substrate_ffnn_wbetas.txt", true);
    //
    cout << "Done." << endl;
    cin.ignore();

    delete ffnn;

    cout << endl << endl;
    return 0;
}
