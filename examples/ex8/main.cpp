#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "ReadUtilities.hpp"
#include "PrintUtilities.hpp"
#include "FeedForwardNeuralNetwork.hpp"



int main() {
    using namespace std;


    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork("ffnn.in");

    cout << "The FFNN read from the file looks like this:" << endl << endl;

    printFFNNStructure(ffnn);


    delete ffnn;

    cout << endl << endl;
    return 0;
}
