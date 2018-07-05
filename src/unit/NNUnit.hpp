#ifndef NN_UNIT
#define NN_UNIT

#include "FedNetworkUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"
#include "NetworkUnitFeederInterface.hpp"

#include <stdexcept>
#include <string>

// Unit of an Artificial Neural Network
class NNUnit: public FedNetworkUnit
{
protected:
    // Activation Function of the unit
    // A function that calculates the output value from the input value (protovalue)
    ActivationFunctionInterface * _actf; // activation function

public:
    // Constructor and destructor
    NNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NetworkUnitFeederInterface * feeder = NULL) : FedNetworkUnit(feeder) {_actf = actf; if (!_actf) throw std::invalid_argument("NNUnit(): Passed pointer 'actf' was NULL.");}
    NNUnit(const std::string &actf_id, NetworkUnitFeederInterface * feeder = NULL) : NNUnit(std_actf::provideActivationFunction(actf_id), feeder) {}
    virtual ~NNUnit(){ delete _actf; }

    // return the ideal mean value (mu) and standard deviation (sigma) of the proto value (pv)
    // (here the ideal values are determined by the actfs active range)
    virtual double getIdealProtoMu(){return _actf->getIdealInputMu();}
    virtual double getIdealProtoSigma(){return _actf->getIdealInputSigma();}

    // return the output mean value (mu) and standard deviation (sigma)
    virtual double getOutputMu(){return _actf->getOutputMu(FedNetworkUnit::getOutputMu());}
    virtual double getOutputSigma(){return _actf->getOutputSigma(FedNetworkUnit::getOutputSigma());}

    // string code getters / setter
    virtual std::string getIdCode(){return "NNU";} // return identifier for unit type
    virtual std::string getMemberTreeCode(){return composeCodes(FedNetworkUnit::getMemberTreeCode(), _actf->getTreeCode());} // append actf treeCode
    virtual void setMemberParams(const std::string &memberTreeCode);

    // Setters
    void setActivationFunction(ActivationFunctionInterface * actf){delete _actf; if (actf) _actf=actf; else throw std::invalid_argument("NNUnit::setActivationFunction(): Passed pointer 'actf' was NULL.");}
    void setActivationFunction(const std::string &actf_id, const std::string &params = ""){this->setActivationFunction(std_actf::provideActivationFunction(actf_id, params));}

    // Getters
    ActivationFunctionInterface * getActivationFunction(){return _actf;}

    // Computation
    void computeOutput();
};


#endif
