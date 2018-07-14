#ifndef ACTIVATION_UNIT
#define ACTIVATION_UNIT

#include "NetworkUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"

#include <stdexcept>
#include <string>

// Network Unit with an activation function
class ActivationUnit: virtual public NetworkUnit
{
protected:
    // Activation Function of the unit
    // A function that calculates the output value from the input value (protovalue)
    ActivationFunctionInterface * _actf; // activation function

public:
    // Constructor and destructor
    ActivationUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction()) : _actf(actf) {if (!_actf) throw std::invalid_argument("ActivationUnit(): Passed pointer 'actf' was NULL.");}
    ActivationUnit(const std::string &actf_id) : ActivationUnit(std_actf::provideActivationFunction(actf_id)) {}
    virtual ~ActivationUnit(){ delete _actf; }

    // return the ideal mean value (mu) and standard deviation (sigma) of the proto value (pv)
    // (here the ideal values are determined by the actfs active range)
    virtual double getIdealProtoMu(){return _actf->getIdealInputMu();}
    virtual double getIdealProtoSigma(){return _actf->getIdealInputSigma();}

    // return the output mean value (mu) and standard deviation (sigma), assuming a constant input value
    virtual double getOutputMu(){return _actf->getOutputMu(_pv, 0);}
    virtual double getOutputSigma(){return _actf->getOutputMu(_pv, 0);}

    // string code getters / setter
    virtual std::string getMemberTreeCode(){return _actf->getTreeCode();}
    virtual void setMemberParams(const std::string &memberTreeCode);
    virtual std::string getIdCode() = 0; // virtual class

    // Setters
    void setActivationFunction(ActivationFunctionInterface * actf){delete _actf; if (actf) _actf=actf; else throw std::invalid_argument("ActivationUnit::setActivationFunction(): Passed pointer 'actf' was NULL.");}
    void setActivationFunction(const std::string &actf_id, const std::string &params = ""){this->setActivationFunction(std_actf::provideActivationFunction(actf_id, params));}

    // Getters
    ActivationFunctionInterface * getActivationFunction(){return _actf;}

    // Computation
    void computeOutput();
};


#endif
