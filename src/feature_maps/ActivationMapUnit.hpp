#ifndef ACTIVATION_MAP_UNIT
#define ACTIVATION_MAP_UNIT

#include "FedActivationUnit.hpp"
#include "FeederInterface.hpp"

#include <stdexcept>
#include <string>

// ActivationMapUnit template (just specify feature map type FM)
//
// Main functionality is to make sure the feeder
// is and stays of the correct feature map type.
//
// In contrast to the FeatureMapUnit, this
// type als has an activation function.
//
template <class FM> // for FM insert the specified kind of feature map
class ActivationMapUnit: public FedActivationUnit
{
public:
    // Constructor and destructor
    ActivationMapUnit(ActivationFunctionInterface * actf = std_actf::provideActivationFunction(), FM * fmap = NULL) : FedActivationUnit(actf, static_cast<FeederInterface *>(fmap)) {}
    ActivationMapUnit(const std::string &actf_id, FM * fmap = NULL) : ActivationMapUnit(std_actf::provideActivationFunction(actf_id), fmap) {}
    ~ActivationMapUnit(){}

    // restrict feeder to FM
    void setFeeder(FeederInterface * feeder){
        if (FM * fmap = dynamic_cast<FM *>(feeder)) {
            FedActivationUnit::setFeeder(fmap);
        }
        else {
            throw std::invalid_argument("[ActivationMapUnit::setFeeder] Passed feeder of type " + feeder->getIdCode() + " is not compatible to a unit of type " + this->getIdCode() + ".");
        }
    }

    FM * getMap(){return static_cast<FM *>(_feeder);}

    // devirtualize
    void computeValues(){FedActivationUnit::computeValues();}
};

#endif
