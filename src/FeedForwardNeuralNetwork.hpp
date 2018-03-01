#ifndef FEED_FORWARD_NEURAL_NETWORK
#define FEED_FORWARD_NEURAL_NETWORK

#include "ActivationFunctionInterface.hpp"
#include "NNLayer.hpp"

#include <vector>
#include <string>


class FeedForwardNeuralNetwork
{
private:
    void construct(const int &insize, const int &hidlaysize, const int &outsize);

protected:
    std::vector<NNLayer *> _L;

    bool _flag_connected;  // flag that tells if the FFNN has been connected or not
    bool _flag_1d, _flag_2d, _flag_v1d, _flag_c1d, _flag_c2d;  // flag that indicates if the substrates for the derivatives have been activated or not

    int _nvp;  // global number of variational parameters

public:
    FeedForwardNeuralNetwork(const int &insize, const int &hidlaysize, const int &outsize);
    FeedForwardNeuralNetwork(const char * filename);  // file must be formatted as with the method storeOnFile()
    FeedForwardNeuralNetwork(FeedForwardNeuralNetwork * ffnn);
    FeedForwardNeuralNetwork(std::vector<std::vector<std::string>> &actf);
    ~FeedForwardNeuralNetwork();



    // --- Get information about the NN structure
    int getNHiddenLayers(){return _L.size()-2;}
    int getNLayers(){return _L.size();}
    int getNInput(){return _L.front()->getNUnits()-1;}
    int getNOutput(){return _L.back()->getNUnits()-1;}
    int getLayerSize(const int &li){return _L[li]->getNUnits();}
    ActivationFunctionInterface * getLayerActivationFunction(const int &li){return _L[li]->getActivationFunction();}
    NNLayer * getLayer(const int &li){return _L[li];}
    bool isConnected(){return _flag_connected;}
    bool hasFirstDerivativeSubstrate(){return _flag_1d;}
    bool hasSecondDerivativeSubstrate(){return _flag_2d;}
    bool hasVariationalFirstDerivativeSubstrate(){return _flag_v1d;}
    bool hasCrossFirstDerivativeSubstrate(){return _flag_c1d;}
    bool hasCrossSecondDerivativeSubstrate(){return _flag_c2d;}


    // --- Modify NN structure
    void setGlobalActivationFunctions(ActivationFunctionInterface * actf);
    void setLayerSize(const int &li, const int &size);
    void setLayerActivationFunction(const int &li, ActivationFunctionInterface * actf);
    void pushHiddenLayer(const int &size);
    void popHiddenLayer();



    // --- Connect the neural network
    void connectFFNN();
    void disconnectFFNN();



    // --- Manage the betas, which exist only after that the FFNN has been connected
    int getNBeta();
    double getBeta(const int &ib);
    void getBeta(double * beta);
    void setBeta(const int &ib, const double &beta);
    void setBeta(const double * beta);
    void randomizeBetas();



    // --- Substrates for the calculations of derivatives
    void addFirstDerivativeSubstrate();  // coordinates first derivatives
    void addSecondDerivativeSubstrate();  // coordinates second derivatives

    // Variational Derivative: either this ...
    void addVariationalFirstDerivativeSubstrate();  // variational first derivatives
    // ... or this
    void addLastHiddenLayerVariationalFirstDerivativeSubstrate();  // variational first derivative for and from the last hidden layer

    // Substrate for the cross derivatives d/dx d/dbeta
    void addCrossFirstDerivativeSubstrate();  // cross first derivatives
    void addLastHiddenLayerCrossFirstDerivativeSubstrate();
    void addCrossSecondDerivativeSubstrate();  // cross second derivatives
    void addLastHiddenLayerCrossSecondDerivativeSubstrate();

    // shortcut for connecting and adding substrates
    void connectAndAddSubstrates(bool flag_d1 = false, bool flag_d2 = false, bool flag_vd1 = false, bool flag_c1d = false, bool flag_c2d = false);

    // Set initial parameters
    void setInput(const double * in);
    void setInput(const int &i, const double &in);



    // --- Computation
    void FFPropagate();

    // Shortcut for computation: set input and get all values and derivatives with one calculations.
    // If some derivatives are not supported (substrate missing) the values will be leaved unchanged.
    void evaluate(const double * in, double * out = NULL, double ** d1 = NULL, double ** d2 = NULL, double ** vd1 = NULL);



    // --- Get outputs
    void getOutput(double * out);
    double getOutput(const int &i);

    void getFirstDerivative(double ** d1);
    double getFirstDerivative(const int &i, const int &i1d); // i is the index of the output elemnet (i.e. unit=1, offset unit is meaningless), i1d the index of the input element

    void getSecondDerivative(double ** d2);
    double getSecondDerivative(const int &i, const int &i2d); // i is the index of the output element, i2d the index of the input element

    void getVariationalFirstDerivative(double ** vd1);
    double getVariationalFirstDerivative(const int &i, const int &iv1d);  // i is the index of the output element, iv1d the index of the beta element

    void getCrossFirstDerivative(double *** d1vd1);
    void getCrossFirstDerivative(const int &i, double ** d1vd1);
    double getCrossFirstDerivative(const int &i, const int &i1d, const int &iv1d);

    void getCrossSecondDerivative(double *** d1vd1);
    void getCrossSecondDerivative(const int &i, double ** d1vd1);
    double getCrossSecondDerivative(const int &i, const int &i1d, const int &iv1d);



    // --- Store FFNN on file
    void storeOnFile(const char * filename);

};


#endif
