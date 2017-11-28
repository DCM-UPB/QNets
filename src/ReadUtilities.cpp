#include "ReadUtilities.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "ActivationFunctionManager.hpp"
#include "ActivationFunctionInterface.hpp"
#include "IdentityActivationFunction.hpp"
#include "LogisticActivationFunction.hpp"
#include "GaussianActivationFunction.hpp"


void readFFNNStructure(const char * filename, std::vector<std::vector<std::string>> &actf){
   using namespace std;
   
   const string null_actf = "xxx";
   
   ifstream file;
   file.open(filename);
   
   string line;
   string word;
   vector<vector<string>> actf_orig;
   
   // reading the file in the matrix actf_orig
   int nline = 0;
   int nwords;
   while (getline(file, line)){
      nline ++;
      istringstream iss(line);
      vector<string> line_actf_orig;
      nwords = 0;
      while (iss >> word){
         nwords ++;
         ActivationFunctionInterface * af = ActivationFunctionManager::provideActivationFunction(word);
         if (af != 0) {
            line_actf_orig.push_back(word);
         } else {line_actf_orig.push_back(null_actf);}
      }
      if (line_actf_orig.size() > 0 )
         actf_orig.push_back(line_actf_orig);
   }
   
   if (actf_orig.size() <= 0)
      throw invalid_argument( "file does not appear to be valid for describing a NN: not even one valid line found");
   
   if (actf_orig[0].size() <= 0)
      throw invalid_argument( "file does not appear to be valid for describing a NN: first line does not contain even one valid column");
   
   actf.clear();
   for (unsigned int i=0; i<actf_orig[0].size(); ++i){
      vector<string> layer_actf;
      for (unsigned int j=0; j<actf_orig.size(); ++j){
         if (actf_orig[j][i].compare(null_actf) != 0){
            layer_actf.push_back(actf_orig[j][i]);
         }
      }
      actf.push_back(layer_actf);
   }
   
}

