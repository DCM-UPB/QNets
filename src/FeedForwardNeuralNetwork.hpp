#ifndef FEED_FORWARD_NEURAL_NETWORK
#define FEED_FORWARD_NEURAL_NETWORK

#include "ActivationFunctionInterface.hpp"
#include "IdentityActivationFunction.hpp"
#include "LogisticActivationFunction.hpp"
#include "GaussianActivationFunction.hpp"
#include "NNLayer.hpp"

#include <vector>
#include <string>


class FeedForwardNeuralNetwork
{
private:
   void construct(const int &insize, const int &hidlaysize, const int &outsize);

protected:
    std::vector<NNLayer *> _L;

    IdentityActivationFunction _id_actf;
    LogisticActivationFunction _log_actf;
    GaussianActivationFunction _gss_actf;

    bool _flag_connected;  // flag that tells if the FFNN has been connected or not
    bool _flag_1d, _flag_2d, _flag_v1d;  // flag that indicates if the substrates for the derivatives have been activated or not

    int _nvp;  // global number of variational parameters

public:
    FeedForwardNeuralNetwork(const int &insize, const int &hidlaysize, const int &outsize);
    FeedForwardNeuralNetwork(const char * filename);  // file must be formatted as with the method storeOnFile()
    FeedForwardNeuralNetwork(std::vector<std::vector<std::string>> &actf);
    ~FeedForwardNeuralNetwork();

    // Get information about the NN
    int getNHiddenLayers(){return _L.size()-2;}
    int getNLayers(){return _L.size();}
    int getNInput(){return _L.front()->getNUnits()-1;}
    int getNOutput(){return _L.back()->getNUnits()-1;}
    int getLayerSize(const int &li){return _L[li]->getNUnits();}
    ActivationFunctionInterface * getLayerActivationFunction(const int &li){return _L[li]->getActivationFunction();}
    NNLayer * getLayer(const int &li){return _L[li];}
    int getNBeta();
    double getBeta(const int &ib);

    // Modify NN structure
    void setGlobalActivationFunctions(ActivationFunctionInterface * actf);
    void setLayerSize(const int &li, const int &size);
    void setLayerActivationFunction(const int &li, ActivationFunctionInterface * actf);
    void pushHiddenLayer(const int &size);
    void popHiddenLayer();
    void setBeta(const int &ib, const double &beta);
    void randomizeBetas();

    // Substrates for the calculations required
    void addFirstDerivativeSubstrate();  // coordinates' first derivatives
    void addSecondDerivativeSubstrate();  // coordinates' second derivatives
    // Variational Derivative: either this ...
    void addVariationalFirstDerivativeSubstrate();  // variational first derivatives
    // ... or this
    void addLastHiddenLayerVariationalFirstDerivativeSubstrate();  // variational first derivative for and from the last hidden layer


    // Set initial parameters
    void setInput(const double * in);
    void setInput(const int &i, const double &in);

    // Connect the neural network
    void connectFFNN();
    void disconnectFFNN();

    // Computation
    void FFPropagate();

    // Get outputs
    void getOutput(double * out);
    double getOutput(const int &i);
    double getFirstDerivative(const int &i, const int &i1d); // i is the index of the output elemnet (i.e. unit=1, offset unit is meaningless), i1d the index of the input element
    double getSecondDerivative(const int &i, const int &i2d); // i is the index of the output element, i2d the index of the input element
    double getVariationalFirstDerivative(const int &i, const int &iv1d);  // i is the index of the output element, iv1d the index of the beta element

    // Store FFNN on file
    void storeOnFile(const char * filename);

};


#endif
