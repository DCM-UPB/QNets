#ifndef FFNN_UNIT_ACTIVATIONUNIT_HPP
#define FFNN_UNIT_ACTIVATIONUNIT_HPP

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/unit/NetworkUnit.hpp"

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
    explicit ActivationUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction()): _actf(actf)
    {
        if (_actf == nullptr) {
            throw std::invalid_argument("ActivationUnit(): Passed pointer 'actf' was NULL.");
        }
    }
    explicit ActivationUnit(const std::string &actf_id): _actf(std_actf::provideActivationFunction(actf_id)) {}
    ~ActivationUnit() override { delete _actf; }

    // return the ideal mean value (mu) and standard deviation (sigma) of the proto value (pv)
    // (here the ideal values are determined by the actfs active range)
    double getIdealProtoMu() override { return _actf->getIdealInputMu(); }
    double getIdealProtoSigma() override { return _actf->getIdealInputSigma(); }

    // return the output mean value (mu) and standard deviation (sigma), assuming a constant input value
    double getOutputMu() override { return _actf->getOutputMu(_pv, 0); }
    double getOutputSigma() override { return _actf->getOutputMu(_pv, 0); }

    // string code getters / setter
    std::string getMemberTreeCode() override { return _actf->getTreeCode(); }
    void setMemberParams(const std::string &memberTreeCode) override;
    std::string getIdCode() override = 0; // virtual class

    // Setters
    void setActivationFunction(ActivationFunctionInterface * actf)
    {
        delete _actf;
        if (actf != nullptr) { _actf = actf; }
        else {
            throw std::invalid_argument("ActivationUnit::setActivationFunction(): Passed pointer 'actf' was NULL.");
        }
    }
    void setActivationFunction(const std::string &actf_id, const std::string &params = "") { this->setActivationFunction(std_actf::provideActivationFunction(actf_id, params)); }

    // Getters
    ActivationFunctionInterface * getActivationFunction() { return _actf; }

    // Computation
    void computeOutput() override;
};


#endif
