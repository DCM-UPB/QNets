#include "FeedForwardNeuralNetwork.hpp"

#include "NNUnit.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>



// --- Variational Parameters

int FeedForwardNeuralNetwork::getNBeta()
{
   using namespace std;
   int nbeta=0;
   for (vector<NNLayer *>::size_type i=0; i<_L.size(); ++i)
   {
      for (int j=0; j<_L[i]->getNUnits(); ++j)
      {
         if (_L[i]->getUnit(j)->getFeeder())
         {
            nbeta += _L[i]->getUnit(j)->getFeeder()->getNBeta();
         }
      }
   }
   return nbeta;
}


double FeedForwardNeuralNetwork::getBeta(const int &ib)
{
   using namespace std;
   if ( ib >= this->getNBeta() )
   {
      cout << endl << "ERROR FeedForwardNeuralNetwork::getBeta : index out of boundaries" << endl;
      cout << ib << " against the maximum allowed " << this->getNBeta() << endl << endl;
   }
   else
   {
      int idx=0;
      for (vector<NNLayer *>::size_type i=0; i<_L.size(); ++i)
      {
         for (int j=0; j<_L[i]->getNUnits(); ++j)
         {
            if (_L[i]->getUnit(j)->getFeeder())
            {
               for (int k=0; k<_L[i]->getUnit(j)->getFeeder()->getNBeta(); ++k)
               {
                  if (idx==ib) return _L[i]->getUnit(j)->getFeeder()->getBeta(k);
                  idx++;
               }
            }
         }
      }
   }
   cout << endl << "ERROR FeedForwardNeuralNetwork::getBeta : index not found" << endl;
   cout << ib << " against the maximum allowed " << this->getNBeta() << endl << endl;
   return -666.;
}


void FeedForwardNeuralNetwork::setBeta(const int &ib, const double &beta)
{
   using namespace std;
   if ( ib >= this->getNBeta() )
   {
      cout << endl << "ERROR FeedForwardNeuralNetwork::getBeta : index out of boundaries" << endl;
      cout << ib << " against the maximum allowed " << this->getNBeta() << endl << endl;
   }
   else
   {
      int idx=0;
      for (vector<NNLayer *>::size_type i=0; i<_L.size(); ++i)
      {
         for (int j=0; j<_L[i]->getNUnits(); ++j)
         {
            if (_L[i]->getUnit(j)->getFeeder())
            {
               for (int k=0; k<_L[i]->getUnit(j)->getFeeder()->getNBeta(); ++k)
               {
                  if (idx==ib) _L[i]->getUnit(j)->getFeeder()->setBeta(k,beta);
                  idx++;
               }
            }
         }
      }
   }

}


void FeedForwardNeuralNetwork::setVariationalParameter(const int &i, const double &vp)
{
   // variables
   std::vector<NNLayer *>::size_type il = 1;
   bool flag = false;
   // find where the variational parameter with the given index is stored, and set its value to vp
   while ( (!flag) && (il<_L.size()) )
   {
      flag = _L[il]->setVariationalParameter(i,vp);
      il++;
   }
   // check for errors
   if (!flag)
   {
      using namespace std;
      cout << endl << "ERROR FeedForwardNeuralNetwork::setVariationalParameter : variational parameter with the given id (" << i<< ") was not found." << endl << endl;
   }
}


double FeedForwardNeuralNetwork::getVariationalParameter(const int &i)
{
   // variables
   double vp;
   std::vector<NNLayer *>::size_type il = 1;
   bool flag = false;
   // find and store the variational parameter with the given index i
   while ( (!flag) && (il<_L.size()) )
   {
      flag = _L[il]->getVariationalParameter(i,vp);
      il++;
   }
   // check for errors
   if (!flag)
   {
      using namespace std;
      cout << endl << "ERROR FeedForwardNeuralNetwork::getVariationalParameter : variational parameter with the given id (" << i<< ") was not found." << endl << endl;
   }
   // return the result
   return vp;
}


// --- Computation

double FeedForwardNeuralNetwork::getVariationalFirstDerivative(const int &i, const int &iv1d)
{
   return ( _L[_L.size()-1]->getUnit(i)->getVariationalFirstDerivativeValue(iv1d) );
}


double FeedForwardNeuralNetwork::getSecondDerivative(const int &i, const int &i2d)
{
   return ( _L[_L.size()-1]->getUnit(i)->getSecondDerivativeValue(i2d) );
}


double FeedForwardNeuralNetwork::getFirstDerivative(const int &i, const int &i1d)
{
   return ( _L[_L.size()-1]->getUnit(i)->getFirstDerivativeValue(i1d) );
}


double FeedForwardNeuralNetwork::getOutput(const int &i)
{
   return ( _L[_L.size()-1]->getUnit(i)->getValue() );
}


void FeedForwardNeuralNetwork::FFPropagate()
{
   for (std::vector<NNLayer *>::size_type i=0; i<_L.size(); ++i)
   {
      _L[i]->computeValues();
   }
}


void FeedForwardNeuralNetwork::setInput(const int &n, const double *in)
{
   using namespace std;
   // check n
   if (n!=_L[0]->getNUnits()-1)
   {
      cout << "ERROR FeedForwardNeuralNetwor::setInput() : n is different from the number of units in the first layer = " << _L[0]->getNUnits()-1 << endl;
   }
   // set the protovalues of the first layer units
   for (int i=1; i<_L[0]->getNUnits(); ++i)
   {
      _L[0]->getUnit(i)->setProtoValue(in[i-1]);
   }
   // set the first derivatives
   if (_flag_1d)
   {
      for (int i=1; i<_L[0]->getNUnits(); ++i)
      {
         _L[0]->getUnit(i)->setFirstDerivativeValue(   i-1,
              _L[0]->getUnit(i)->getActivationFunction()->f1d( _L[0]->getUnit(i)->getProtoValue() )   );
      }
   }
}


// --- Substrates

void FeedForwardNeuralNetwork::addLastHiddenLayerVariationalFirstDerivativeSubstrate()
{
   // count the total number of variational parameters
   _nvp=0;
   for (std::vector<NNLayer *>::size_type i=_L.size()-2; i<_L.size(); ++i)
   {
      _nvp += _L[i]->getNVariationalParameters();
   }
   // set the substrate in the units
   for (std::vector<NNLayer *>::size_type i=0; i<_L.size(); ++i)
   {
      _L[i]->addVariationalFirstDerivativeSubstrate(_nvp);
   }
   // set the id of the variational parameters for all the feeders
   int id_vp=0;
   for (std::vector<NNLayer *>::size_type i=_L.size()-2; i<_L.size(); ++i)
   {
      id_vp = _L[i]->setVariationalParametersID(id_vp);
   }
}


void FeedForwardNeuralNetwork::addVariationalFirstDerivativeSubstrate()
{
   // count the total number of variational parameters
   _nvp=0;
   for (std::vector<NNLayer *>::size_type i=1; i<_L.size(); ++i)
   {
      _nvp += _L[i]->getNVariationalParameters();
   }
   // set the substrate in the units
   for (std::vector<NNLayer *>::size_type i=0; i<_L.size(); ++i)
   {
      _L[i]->addVariationalFirstDerivativeSubstrate(_nvp);
   }
   // set the id of the variational parameters for all the feeders
   int id_vp=0;
   for (std::vector<NNLayer *>::size_type i=1; i<_L.size(); ++i)
   {
      id_vp = _L[i]->setVariationalParametersID(id_vp);
   }

   _flag_v1d = true;
}


void FeedForwardNeuralNetwork::addSecondDerivativeSubstrate()
{
   // add the second derivative substrate to all the layers
   for (std::vector<NNLayer *>::size_type i=0; i<_L.size(); ++i)
   {
      _L[i]->addSecondDerivativeSubstrate(_L[0]->getNUnits()-1);
   }

   _flag_2d = true;
}


void FeedForwardNeuralNetwork::addFirstDerivativeSubstrate()
{
   // set and initialize the input layer
   _L[0]->addFirstDerivativeSubstrate(_L[0]->getNUnits()-1);
   for (int i=1; i<_L[0]->getNUnits(); ++i)
   {
      _L[0]->getUnit(i)->setFirstDerivativeValue(   i-1,
           _L[0]->getUnit(i)->getActivationFunction()->f1d( _L[0]->getUnit(i)->getProtoValue() )   );
   }
   // set all the other layers
   for (std::vector<NNLayer *>::size_type i=1; i<_L.size(); ++i)
   {
      _L[i]->addFirstDerivativeSubstrate(_L[0]->getNUnits()-1);
   }

   _flag_1d = true;
}


// --- Connect the neural network

void FeedForwardNeuralNetwork::connectFFNN()
{
   for (std::vector<NNLayer *>::size_type i=1; i<_L.size(); ++i)
   {
      _L[i]->connectOnTopOfLayer(_L[i-1]);
   }
   _flag_connected = true;
}


void FeedForwardNeuralNetwork::disconnectFFNN()
{
   if ( !_flag_connected )
   {
      using namespace std;
      cout << "ERROR: FeedForwardNeuralNetwork::disconnectFFNN() : trying to disconnect an already disconnected FFNN" << endl << endl;
   }

   for (std::vector<NNLayer *>::size_type i=1; i<_L.size(); ++i)
   {
      _L[i]->disconnect();
   }
   _flag_connected = false;
}


// --- Modify NN structure

void FeedForwardNeuralNetwork::setGlobalActivationFunctions(ActivationFunctionInterface * actf)
{
   for (std::vector<NNLayer *>::size_type i=1; i<_L.size(); ++i)
   {
      _L[i]->setActivationFunction(actf);
   }
}


void FeedForwardNeuralNetwork::setLayerSize(const int &li, const int &size)
{
   _L[li]->setSize(size);
}


void FeedForwardNeuralNetwork::setLayerActivationFunction(const int &li, ActivationFunctionInterface * actf)
{
   _L[li]->setActivationFunction(actf);
}


void FeedForwardNeuralNetwork::pushHiddenLayer(const int &size)
{
   NNLayer * newhidlay = new NNLayer(size, &_log_actf);

   std::vector<NNLayer *>::iterator it = _L.end()-1;

   if (_flag_connected)
   {
      using namespace std;
      // count the number of beta before the last (output) layer
      int nbeta = 0;
      for (vector<NNLayer *>::size_type i=0; i<_L.size()-1; ++i)
      {
         for (int j=0; j<_L[i]->getNUnits(); ++j)
         {
            if (_L[i]->getUnit(j)->getFeeder())
            {
               nbeta += _L[i]->getUnit(j)->getFeeder()->getNBeta();
            }
         }
      }
      int total_nbeta = this->getNBeta();
      // store the beta for the output
      double * old_beta = new double[total_nbeta-nbeta];
      for (int i=nbeta; i<total_nbeta; ++i)
      {
         old_beta[i-nbeta] = getBeta(i);
      }

      // disconnect last layer
      _L[_L.size()-1]->disconnect();  // disconnect the last (output) layer
      // insert new layer
      _L.insert(it, newhidlay);
      // reconnect the layers
      _L[_L.size()-2]->connectOnTopOfLayer(_L[_L.size()-3]);
      _L[_L.size()-1]->connectOnTopOfLayer(_L[_L.size()-2]);

      // restore the old beta
      for (int i=nbeta; i<total_nbeta; ++i)
      {
         this->setBeta(i,old_beta[i-nbeta]);
      }
      // set all the other beta to zero
      for (int i=total_nbeta; i<this->getNBeta(); ++i)
      {
         this->setBeta(i,0.);
      }
      // set the identity activation function for some units of the new hidden layer
      for (int i=1; i<_L[_L.size()-1]->getNUnits(); ++i) 
      {
         _L[_L.size()-2]->getUnit(i)->setActivationFunction(&_id_actf);
      }
      // set some beta to 1 for the output layer
      for (int i=1; i<_L[_L.size()-1]->getNUnits(); ++i)
      {
         _L[_L.size()-1]->getUnit(i)->getFeeder()->setBeta(i,1.);
      }
      // free memory
      delete[] old_beta;
   }
   else
   {
      _L.insert(it, newhidlay);
   }
}


void FeedForwardNeuralNetwork::popHiddenLayer()
{
   delete _L[_L.size()-2];

   std::vector<NNLayer *>::iterator it = _L.end()-2;
   _L.erase(it);
}


// --- Store FFNN on a file

void FeedForwardNeuralNetwork::storeOnFile(const char * filename)
{
   using namespace std;
   
   // open file
   ofstream file;
   file.open(filename);
   // store the number of layers
   file << this->getNLayers() << endl;
   // store the activaction function and size of each layer
   for (int i=0; i<this->getNLayers(); ++i)
   {
      if ( _L[i]->getActivationFunction() == &_id_actf ) file  << "id_actf"  << " ";
      if ( _L[i]->getActivationFunction() == &_log_actf ) file << "log_actf" << " ";
      file << this->getLayerSize(i) << " ";
   }
   file << endl;
   // store all the variational parameters
   for (int i=0; i<this->getNBeta(); ++i)
   {
      file << getBeta(i) << " ";
   }
   file << endl;
   file.close();
}


// --- Constructor

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(const char *filename)
{
   // open file
   using namespace std;

   ifstream file;
   file.open(filename);
   string line;
   stringstream sline;
   // read the number of layers
   int nlayers;
   file >> nlayers;
   // read and set the activation function and size of each layer
   string actf;
   int size;
   NNLayer * nnl;
   for (int i=0; i<nlayers; ++i)
   {
      file >> actf;
      file >> size;
      if (actf.compare("id_actf") == 0) {nnl = new NNLayer(size, &_id_actf);}
      else if (actf.compare("log_actf") == 0) {nnl = new NNLayer(size, &_log_actf);}
      else {cout << "ERROR FeedForwardNeuralNetwork(const char * filename) : activation function " << actf << " not known" << endl;}
      _L.push_back(nnl);
   }
   // set some other initial values
   _flag_1d = false;
   _flag_2d = false;
   _flag_v1d = false;
   _nvp=0;
   // connect the NN
   this->connectFFNN();
   // set the beta
   double beta;
   for (int i=0; i<this->getNBeta(); ++i)
   {
      file >> beta;
      this->setBeta(i,beta);
   }
   file.close();
}


FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(const int &insize, const int &hidlaysize, const int &outsize)
{
   NNLayer * in = new NNLayer(insize, &_id_actf);
   NNLayer * hidlay = new NNLayer(hidlaysize, &_log_actf);
   NNLayer * out = new NNLayer(outsize, &_log_actf);

   _L.push_back(in);
   _L.push_back(hidlay);
   _L.push_back(out);

   _flag_connected = false;
   _flag_1d = false;
   _flag_2d = false;
   _flag_v1d = false;

   _nvp=0;
}


// --- Destructor

FeedForwardNeuralNetwork::~FeedForwardNeuralNetwork()
{
   for (std::vector<NNLayer *>::size_type i=0; i<_L.size(); ++i)
   {
      delete _L[i];
   }
   _L.clear();

   _flag_connected = false;
   _flag_1d = false;
   _flag_2d = false;
   _flag_v1d = false;

   _nvp=0;
}
