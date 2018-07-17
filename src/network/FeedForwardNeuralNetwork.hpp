#ifndef FEED_FORWARD_NEURAL_NETWORK
#define FEED_FORWARD_NEURAL_NETWORK

#include "ActivationFunctionInterface.hpp"
#include "NetworkLayer.hpp"
#include "InputLayer.hpp"
#include "FedLayer.hpp"
#include "NNLayer.hpp"
#include "OutputNNLayer.hpp"
#include "FeatureMapLayer.hpp"
#include "NetworkUnit.hpp"

#include <vector>
#include <string>
#include <cstddef>

class FeedForwardNeuralNetwork
{
private:
    void _construct(const int &insize, const int &hidlaysize, const int &outsize); // construct from minimal set of unit numbers
    void _registerLayer(NetworkLayer * newLayer, const int &indexFromBack = 0); // register layers to correct vectors, position controlled by indexFromBack
    void _addNewLayer(const std::string &idCode, const int &nunits, const int &indexFromBack = 0, const std::string &params=""); // creates and registers a new layer according to idCode and nunits
    void _addNewLayer(const std::string &idCode, const std::string &params="", const int &indexFromBack = 0); // creates and registers a new layer according to idCode and params code (without it the layer will only have an offset unit)
    void _updateNVP(); // internal method to update _nvp member, call it after you changed/created variational parameter assignment
protected:
    std::vector<NetworkLayer *> _L; // contains all kinds of layers
    std::vector<FedLayer *> _L_fed; // contains layers with feeder
    std::vector<NNLayer *> _L_nn; // contains neural layers
    std::vector<FeatureMapLayer *> _L_fm; // contains feature map layers
    InputLayer * _L_in = NULL; // input layer
    OutputNNLayer * _L_out = NULL; // output layer

    bool _flag_connected = false;  // flag that tells if the FFNN has been connected or not
    bool _flag_1d = false, _flag_2d = false, _flag_v1d = false, _flag_c1d = false, _flag_c2d = false;  // flag that indicates if the substrates for the derivatives have been activated or not

    int _nvp = 0;  // global number of variational parameters

public:
    FeedForwardNeuralNetwork(const int &insize, const int &hidlaysize, const int &outsize);
    explicit FeedForwardNeuralNetwork(const char * filename);  // file must be formatted as with the method storeOnFile()
    explicit FeedForwardNeuralNetwork(FeedForwardNeuralNetwork * ffnn);
    ~FeedForwardNeuralNetwork();


    // --- Get information about the NN structure
    int getNLayers(){return _L.size();}
    int getNFedLayers(){return _L_fed.size();}
    int getNNeuralLayers(){return _L_nn.size();}
    int getNFeatureMapLayers(){return _L_fm.size();}
    int getNHiddenLayers(){return _L_nn.size()-1;}

    int getNInput(){return _L_in->getNInputUnits();}
    int getNOutput(){return _L_out->getNOutputNNUnits();}
    int getLayerSize(const int &li){return _L[li]->getNUnits();}

    NetworkLayer * getLayer(const int &li){return _L[li];}
    FedLayer * getFedLayer(const int &li){return _L_fed[li];}
    NNLayer * getNNLayer(const int &li){return _L_nn[li];}
    FeatureMapLayer * getFeatureMapLayer(const int &li){return _L_fm[li];}
    InputLayer * getInputLayer(){return _L_in;}
    OutputNNLayer * getOutputLayer(){return _L_out;}

    bool isConnected(){return _flag_connected;}
    bool hasFirstDerivativeSubstrate(){return _flag_1d;}
    bool hasSecondDerivativeSubstrate(){return _flag_2d;}
    bool hasVariationalFirstDerivativeSubstrate(){return _flag_v1d;}
    bool hasCrossFirstDerivativeSubstrate(){return _flag_c1d;}
    bool hasCrossSecondDerivativeSubstrate(){return _flag_c2d;}


    // --- Modify NN structure
    void setGlobalActivationFunctions(ActivationFunctionInterface * actf);
    void pushHiddenLayer(const int &size);
    void popHiddenLayer();
    void pushFeatureMapLayer(const int &size, const std::string &params="");


    // --- Connect the neural network
    void connectFFNN();
    void disconnectFFNN();


    // --- Manage the betas, which exist only after that the FFNN has been connected
    int getNBeta();
    double getBeta(const int &ib);
    void getBeta(double * beta);
    void setBeta(const int &ib, const double &beta);
    void setBeta(const double * beta);
    void randomizeBetas(); // has to be changed maybe if we add beta that are not "normal" weights

    // --- Manage the variational parameters (which may contain a subset of beta and/or non-beta parameters),
    //     which exist only after that they are assigned to actual parameters in the network (e.g. betas)
    void assignVariationalParameters(const int &starting_layer_index = 0); // make betas variational parameters, starting from starting_layer
    int getNVariationalParameters(){return _nvp;}
    double getVariationalParameter(const int &ivp);
    void getVariationalParameter(double * vp);
    void setVariationalParameter(const int &ivp, const double &vp);
    void setVariationalParameter(const double * vp);


    // --- Substrates for the calculations of derivatives
    void addFirstDerivativeSubstrate();  // coordinates first derivatives
    void addSecondDerivativeSubstrate();  // coordinates second derivatives

    // Substrate for the variational derivative d/dbeta:
    void addVariationalFirstDerivativeSubstrate();  // variational first derivatives

    // Substrate for the cross derivatives d/dx d/dbeta
    void addCrossFirstDerivativeSubstrate();  // cross first derivatives
    void addCrossSecondDerivativeSubstrate();  // cross second derivatives

    // shortcut for (connecting and) adding substrates
    void addSubstrates(const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_vd1 = false, const bool flag_c1d = false, const bool flag_c2d = false);
    void connectAndAddSubstrates(const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_vd1 = false, const bool flag_c1d = false, const bool flag_c2d = false);


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
    void getFirstDerivative(const int &iu, double * d1);  // iu is the unit index
    double getFirstDerivative(const int &iu, const int &i1d); // i is the index of the output elemnet (i.e. unit=1, offset unit is meaningless), i1d the index of the input element

    void getSecondDerivative(double ** d2);
    void getSecondDerivative(const int &i, double * d2);  // i is the output index
    double getSecondDerivative(const int &i, const int &i2d); // i is the index of the output element, i2d the index of the input element

    void getVariationalFirstDerivative(double ** vd1);
    void getVariationalFirstDerivative(const int &i, double * vd1);  // i is the output index
    double getVariationalFirstDerivative(const int &i, const int &iv1d);  // i is the index of the output element, iv1d the index of the beta element

    void getCrossFirstDerivative(double *** d1vd1);
    void getCrossFirstDerivative(const int &i, double ** d1vd1);  // i is the output index
    double getCrossFirstDerivative(const int &i, const int &i1d, const int &iv1d);  // i is the index of the output element, i1d, of the input element, iv1d the index of the beta element

    void getCrossSecondDerivative(double *** d1vd1);
    void getCrossSecondDerivative(const int &i, double ** d1vd1);  // i is the output index
    double getCrossSecondDerivative(const int &i, const int &i2d, const int &iv1d);  // i is the index of the output element, i2d, of the input element, iv1d the index of the beta element


    // --- Store FFNN on file
    void storeOnFile(const char * filename, const bool store_betas = true);
};


#endif
