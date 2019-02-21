#ifndef NN_UNIT
#define NN_UNIT

#include "ffnn/unit/FedActivationUnit.hpp"
#include "ffnn/actf/ActivationFunctionInterface.hpp"
#include "ffnn/actf/ActivationFunctionManager.hpp"
#include "ffnn/feed/FeederInterface.hpp"
#include "ffnn/feed/NNRay.hpp"

#include <stdexcept>
#include <string>

// Unit of a fully connected Artificial Neural Network layer
class NNUnit: public FedActivationUnit
{
public:
    // Constructor and destructor
    NNUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), NNRay * ray = NULL) : FedActivationUnit(actf, static_cast<FeederInterface *>(ray)) {}
    NNUnit(const std::string &actf_id, NNRay * ray = NULL) : NNUnit(std_actf::provideActivationFunction(actf_id), ray) {}
    virtual ~NNUnit(){}

    // string code id
    virtual std::string getIdCode(){return "NNU";} // return identifier for unit type

    // restrict feeder to ray type
    void setFeeder(FeederInterface * feeder){
        if (NNRay * ray = dynamic_cast<NNRay *>(feeder)) {
            FedUnit::setFeeder(ray);
        }
        else {
            throw std::invalid_argument("[NNUnit::setFeeder] Passed feeder is not of class NNRay, but only NNRay or derived are allowed.");
        }
    }

    NNRay * getRay(){return static_cast<NNRay *>(_feeder);}
};


#endif
