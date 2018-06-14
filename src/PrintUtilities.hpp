#ifndef PRINT_UTILITIES
#define PRINT_UTILITIES


#include "FeedForwardNeuralNetwork.hpp"


#include <string>


void printFFNNStructure(FeedForwardNeuralNetwork * ffnn, const bool &drop_params = true, const int &drop_member_lvl = 1); // will drop members from level drop_member_lvl onward (0 means drop is disabled)

void printFFNNStructureWithBeta(FeedForwardNeuralNetwork * ffnn);

void printFFNNValues(FeedForwardNeuralNetwork * ffnn);

void writePlotFile(FeedForwardNeuralNetwork * ffnn, const double * base_input, const int &input_i, const int &output_i, const double &min, const double &max, const int &npoints, std::string what, std::string filename, const double &xscale = 1.0, const double &yscale = 1.0, const double &xshift = 0.0, const double &yshift = 0.0);
// the string "what" can be:
// 1. getOutput
// 2. getFirstDerivative
// 3. getSecondDerivative
// 4. getVariationalFirstDerivative


#endif
