#include <iostream>
#include <cmath>
#include <fstream>

#include "FeedForwardNeuralNetwork.hpp"


void printNNStructure(FeedForwardNeuralNetwork &nn)
{
   using namespace std;

   for (int i=0; i<nn.getNLayers(); ++i)
   {
      cout << "Layer " << i << " has " << nn.getLayerSize(i) << " units" << endl;
   }
   cout << endl;
}

void printNNValues(FeedForwardNeuralNetwork &nn)
{
   using namespace std;

   for (int i=0; i<nn.getNLayers(); ++i)
   {
      cout << "--- Layer " << i << endl;
      cout << "Value:              ";
      for (int j=0; j<nn.getLayer(i)->getNUnits(); ++j)
      {
         cout << nn.getLayer(i)->getUnit(j)->getValue() << "   ";
      }
      cout << endl;
      cout << "First Derivative:   ";
      for (int j=0; j<nn.getLayer(i)->getNUnits(); ++j)
      {
         cout << nn.getLayer(i)->getUnit(j)->getFirstDerivativeValue(0) << "   ";
      }
      cout << endl << endl;
   }

   cout << "Variational parameters:   ";
   for (int j=0; j<nn.getNVariationalParameters(); ++j)
   {
      cout << nn.getVariationalParameter(j) << "   ";
   }
   cout << endl << endl;
}


int main(){
   using namespace std;

   cout << "Declare the FFANN" << endl;
   FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(4,6,2);
   printNNStructure(*ffnn);

   cout << "Change size of Layer 0 to 2" << endl;
   ffnn->setLayerSize(0,2);
   printNNStructure(*ffnn);

   cout << "Pop a hidden layer" << endl;
   ffnn->popHiddenLayer();
   printNNStructure(*ffnn);

   cout << "Add a hidden layer with size 5" << endl;
   ffnn->pushHiddenLayer(5);
   printNNStructure(*ffnn);

   cout << "Add a hidden layer with size 3" << endl;
   ffnn->pushHiddenLayer(3);
   printNNStructure(*ffnn);

   cout << "Connect the FFNN" << endl;
   ffnn->connectFFNN();

   ffnn->addFirstDerivativeSubstrate();
   cout << "Add the first derivative to the computations" << endl;
   ffnn->addSecondDerivativeSubstrate();
   cout << "Add the second derivative to the computations" << endl;
   ffnn->addVariationalFirstDerivativeSubstrate();
   cout << "Add the variational first derivative to the computations" << endl;
   cout << " -> Number of variational parameters: " << ffnn->getNVariationalParameters() << endl;
   cout << endl;

   double x=3.;
   ffnn->setInput(1, &x);
   ffnn->FFPropagate();
   printNNValues(*ffnn);

   // Plot of the 1-dimensional function
   const double L=10.;
   double xh;
   const int N=150;
   double dL=L/N;
   const double h=0.00001;
   double z, zh, zmh;
   double z1d;
   double z2d;
   const int ivar=2;
   double vp, vph;
   // nn value
   ofstream file;
   file.open("randomNN.txt");
   // first derivative
   ofstream file1d;
   file1d.open("randomNN-1d.txt");
   // second derivative
   ofstream file2d;
   file2d.open("randomNN-2d.txt");
   // variational first derivative
   ofstream filev1d;
   filev1d.open("randomNN-v1d.txt");
   // numerical first derivative
   ofstream filen1d;
   filen1d.open("randomNN-n1d.txt");
   // numerical second derivative
   ofstream filen2d;
   filen2d.open("randomNN-n2d.txt");
   // numerical variational first derivative
   ofstream filenv1d;
   filenv1d.open("randomNN-nv1d.txt");
   x=-L*0.5;
   for (int i=0; i<N; ++i)
   {
      x+=dL;
      ffnn->setInput(1, &x);
      ffnn->FFPropagate();
      z=ffnn->getOutput(1);
      file << x << "   " << z << endl;

      // first derivative
      z1d=ffnn->getFirstDerivative(1,0);
      file1d << x << "   " << z1d << endl;

      // second derivative
      z2d=ffnn->getSecondDerivative(1,0);
      file2d << x << "   " << z2d << endl;

      // variational first derivative
      z1d=ffnn->getVariationalFirstDerivative(1,ivar);
      filev1d << x << "   " << z1d << endl;

      // numerical first derivative
      xh=x+h;
      ffnn->setInput(1, &xh);
      ffnn->FFPropagate();
      zh=ffnn->getOutput(1);
      filen1d << x << "   " << (zh-z)/h << endl;
      ffnn->setInput(1, &x);

      // numerical second derivative
      xh=x-h;
      ffnn->setInput(1, &xh);
      ffnn->FFPropagate();
      zmh=ffnn->getOutput(1);
      filen2d << x << "   " << (zh-2.*z+zmh)/(h*h) << endl;
      ffnn->setInput(1, &x);

      // numerical variational first derivative
      vp=ffnn->getVariationalParameter(ivar);
      vph=vp+h;
      ffnn->setVariationalParameter(ivar,vph);
      ffnn->FFPropagate();
      zh=ffnn->getOutput(1);
      filenv1d << x << "   " << (zh-z)/h << endl;
      ffnn->setVariationalParameter(ivar,vp);

   }
   file.close();
   file1d.close();
   file2d.close();
   filev1d.close();
   filen1d.close();
   filen2d.close();
   filenv1d.close();

   // store NN in a file
   ffnn->storeOnFile("FFNN.txt");

   // create a new NN reading it from the file
   cout << endl << endl << "CREATE A COPY OF THE PREVIOUS FFNN USING THE FILE IN WHICH IT WAS STORED ..." << endl;
   FeedForwardNeuralNetwork * ffnn_copy =new FeedForwardNeuralNetwork("FFNN.txt");
   x=3.;
   ffnn->setInput(1, &x);
   ffnn->FFPropagate();
   printNNValues(*ffnn);

   // free memory
   delete ffnn_copy;
   delete ffnn;

   return 0;
}


