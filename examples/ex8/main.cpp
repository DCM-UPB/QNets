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

    vector<vector<string>> actf;
    readFFNNStructure("ffnn_geometry.in", actf);

    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(actf);

    cout << "The FFNN read from the file looks like this:" << endl << endl;

    printFFNNStructure(ffnn);


    delete ffnn;

    cout << endl << endl;
    return 0;
}
