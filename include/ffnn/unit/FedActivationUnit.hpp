#ifndef FFNN_UNIT_FEDACTIVATIONUNIT_HPP
#define FFNN_UNIT_FEDACTIVATIONUNIT_HPP

#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/feed/FeederInterface.hpp"
#include "ffnn/unit/ActivationUnit.hpp"
#include "ffnn/unit/FedUnit.hpp"

#include <string>

// Network Unit with feeder and activation function
class FedActivationUnit: public FedUnit, public ActivationUnit
{
public:
    // Constructor and destructor
    explicit FedActivationUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), FeederInterface * feeder = nullptr):
            FedUnit(feeder), ActivationUnit(actf) {}
    explicit FedActivationUnit(const std::string &actf_id, FeederInterface * feeder = nullptr):
            FedActivationUnit(std_actf::provideActivationFunction(actf_id), feeder) {}
    ~FedActivationUnit() override = default;

    // return the output mean value (mu) and standard deviation (sigma)
    double getOutputMu() override { return _actf->getOutputMu(FedUnit::getOutputMu(), FedUnit::getOutputSigma()); }
    double getOutputSigma() override { return _actf->getOutputSigma(FedUnit::getOutputMu(), FedUnit::getOutputSigma()); }

    // string code getters / setter
    std::string getMemberTreeCode() override { return composeCodes(FedUnit::getMemberTreeCode(), ActivationUnit::getMemberTreeCode()); } // append actf treeCode
    void setMemberParams(const std::string &memberTreeCode) override
    {
        FedUnit::setMemberParams(memberTreeCode);
        ActivationUnit::setMemberParams(memberTreeCode);
    }
    std::string getIdCode() override = 0; // still meant as abstract
};


#endif
