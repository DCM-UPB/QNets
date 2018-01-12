#ifndef PRINT_UTILITIES
#define PRINT_UTILITIES


#include "FeedForwardNeuralNetwork.hpp"


#include <string>


void printFFNNStructure(FeedForwardNeuralNetwork * ffnn);

void printFFNNStructureWithBeta(FeedForwardNeuralNetwork * ffnn);

void printFFNNValues(FeedForwardNeuralNetwork * ffnn);

void writePlotFile(FeedForwardNeuralNetwork * ffnn, const double * base_input, const int &input_i, const int &output_i, const double &min, const double &max, const int &npoints, std::string what, std::string filename);
// the string "what" can be:
// 1. getOutput
// 2. 
// 3. 
// 4.  


#endif
