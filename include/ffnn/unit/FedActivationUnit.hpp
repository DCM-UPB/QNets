#ifndef FED_ACTIVATION_UNIT
#define FED_ACTIVATION_UNIT

#include "FedUnit.hpp"
#include "ActivationUnit.hpp"
#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"
#include "FeederInterface.hpp"

#include <string>

// Network Unit with feeder and activation function
class FedActivationUnit: public FedUnit, public ActivationUnit
{
public:
    // Constructor and destructor
    FedActivationUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), FeederInterface * feeder = NULL) : FedUnit(feeder), ActivationUnit(actf) {}
    FedActivationUnit(const std::string &actf_id, FeederInterface * feeder = NULL) : FedActivationUnit(std_actf::provideActivationFunction(actf_id), feeder) {}
    virtual ~FedActivationUnit(){}

    // return the output mean value (mu) and standard deviation (sigma)
    virtual double getOutputMu(){return _actf->getOutputMu(FedUnit::getOutputMu(), FedUnit::getOutputSigma());}
    virtual double getOutputSigma(){return _actf->getOutputSigma(FedUnit::getOutputMu(), FedUnit::getOutputSigma());}

    // string code getters / setter
    virtual std::string getMemberTreeCode(){return composeCodes(FedUnit::getMemberTreeCode(), ActivationUnit::getMemberTreeCode());} // append actf treeCode
    virtual void setMemberParams(const std::string &memberTreeCode){FedUnit::setMemberParams(memberTreeCode); ActivationUnit::setMemberParams(memberTreeCode);}
    virtual std::string getIdCode() = 0; // still meant as abstract
};


#endif
