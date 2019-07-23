#ifndef FFNN_UNIT_FEDACTIVATIONUNIT_HPP
#define FFNN_UNIT_FEDACTIVATIONUNIT_HPP

#include "qnets/poly/actf/ActivationFunctionInterface.hpp"
#include "qnets/poly/actf/ActivationFunctionManager.hpp"
#include "qnets/poly/feed/FeederInterface.hpp"
#include "qnets/poly/unit/ActivationUnit.hpp"
#include "qnets/poly/unit/FedUnit.hpp"

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
