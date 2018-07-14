#ifndef NN_LAYER
#define NN_LAYER

#include "FedLayer.hpp"
#include "NetworkLayer.hpp"
#include "NNUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"
#include "FeederInterface.hpp"
#include "NNRay.hpp"

#include <vector>
#include <string>

class NNLayer: public FedLayer
{
protected:
    std::vector<NNUnit *> _U_nn; // stores pointers to all neural units

    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is a/derived from NNUnit and register
public:
    // --- Constructor

    NNLayer(const int &nunits = 1, ActivationFunctionInterface * actf = std_actf::provideActivationFunction()){if (nunits>1) construct(nunits, actf);}
    virtual void construct(const int &nunits);
    virtual void construct(const int &nunits, ActivationFunctionInterface * actf);

    // --- Deconstructor

    virtual ~NNLayer(){_U_nn.clear();}
    virtual void deconstruct(){FedLayer::deconstruct(); _U_nn.clear();}

    // --- String Codes

    virtual std::string getIdCode(){return "NNL";}

    // --- Getters

    int getNNeuralUnits() {return _U_nn.size();}
    NNUnit * getNNUnit(const int &i) {return _U_nn[i];}

    // --- Modify structure

    void setActivationFunction(ActivationFunctionInterface * actf);

    // --- Connection

    virtual FeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i) {return new NNRay(nl);}
};


#endif
