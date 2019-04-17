#include <cmath>
#include <fstream>
#include <iostream>

#include "qnets/io/PrintUtilities.hpp"


int main()
{
    using namespace std;

    int n0, n1, n2;

    cout << "Let's start by creating a Feed Forward Artificial Neural Network (FFANN)" << endl;
    cout << "========================================================================" << endl;
    cin.ignore();

    cout << "A newly generated FFANN has three layers:\n- the first one is the input layer\n- the second one is a hidden layer\n- the third one is an output layer." << endl;
    cin.ignore();

    cout << "How many units should the input layer have? ";
    cin >> n0;
    cout << "How many units should the hidden layer have? ";
    cin >> n1;
    cout << "How many units should the output layer have? ";
    cin >> n2;

    cout << endl << "We now create a FFANN with " << n0 << ", " << n1 << ", " << n2 << " units." << endl;
    cin.ignore();

    // NON I/O CODE
    auto * ffnn = new FeedForwardNeuralNetwork(n0, n1, n2);
    //

    cout << "Graphically it looks like this:" << endl;
    cin.ignore();
    printFFNNStructure(ffnn);

    cout << endl << "where it must be read from left to right, and:" << endl;
    cout << "OFF: Offset Unit" << endl;
    cout << "IN:  Input Unit" << endl;
    cout << "NNU: (Hidden) Neural Network Unit" << endl;
    cout << "OUT: Output Unit" << endl << endl;
    cin.ignore();

    cout << "At the moment the substructure of the units looks like this:" << endl;
    cin.ignore();
    printFFNNStructure(ffnn, true, 0);
    cout << endl << "which means that NNU and OUT units apply an logistic activation function (LGS)." << endl << endl << endl;

    cout << "Now let's add one more layer" << endl;
    cout << "========================" << endl;
    cin.ignore();
    cout << "How many units should this new layer have? ";
    cin.ignore();

    int nh;
    cin >> nh;

    cout << "We add a new hidden layer with " << nh << " units" << endl;
    cin.ignore();

    // NON I/O CODE
    ffnn->pushHiddenLayer(nh);
    //

    cout << "Graphically it looks like this:" << endl << endl;
    cin.ignore();
    printFFNNStructure(ffnn);
    cin.ignore();

    cout << "As you may have noticed, this new layer is inserted right before the output layer." << endl << endl << endl;
    cin.ignore();

    cout << "Let's remove one layer" << endl;
    cout << "==========================" << endl;
    cin.ignore();

    // NON I/O CODE
    ffnn->popHiddenLayer();
    //

    cout << "Graphically it looks like this:" << endl;
    cin.ignore();
    printFFNNStructure(ffnn);
    cin.ignore();

    cout << endl << "As you might have noticed, the layer that is removed is the hidden layer right before the output layer." << endl << endl << endl << endl;
    cin.ignore();

    cout << "This is the end of this example, we hope it has been useful for you." << endl << endl;

    return 0;
}
