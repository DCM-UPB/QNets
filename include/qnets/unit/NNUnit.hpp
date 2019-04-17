#ifndef FFNN_UNIT_NNUNIT_HPP
#define FFNN_UNIT_NNUNIT_HPP

#include "qnets/actf/ActivationFunctionInterface.hpp"
#include "qnets/actf/ActivationFunctionManager.hpp"
#include "qnets/feed/FeederInterface.hpp"
#include "qnets/feed/NNRay.hpp"
#include "qnets/unit/FedActivationUnit.hpp"

#include <stdexcept>
#include <string>

// Unit of a fully connected Artificial Neural Network layer
class NNUnit: public FedActivationUnit
{
public:
    // Constructor and destructor
    explicit NNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NNRay * ray = nullptr):
            FedActivationUnit(actf, static_cast<FeederInterface *>(ray)) {}
    explicit NNUnit(const std::string &actf_id, NNRay * ray = nullptr):
            NNUnit(std_actf::provideActivationFunction(actf_id), ray) {}
    ~NNUnit() override = default;

    // string code id
    std::string getIdCode() override { return "NNU"; } // return identifier for unit type

    // restrict feeder to ray type
    void setFeeder(FeederInterface * feeder) final
    {
        if (auto * ray = dynamic_cast<NNRay *>(feeder)) {
            FedUnit::setFeeder(ray);
        }
        else {
            throw std::invalid_argument("[NNUnit::setFeeder] Passed feeder is not of class NNRay, but only NNRay or derived are allowed.");
        }
    }

    NNRay * getRay() { return dynamic_cast<NNRay *>(_feeder); }
};


#endif
