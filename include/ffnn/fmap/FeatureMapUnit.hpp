#ifndef FEATURE_MAP_UNIT
#define FEATURE_MAP_UNIT

#include "ffnn/unit/FedUnit.hpp"
#include "ffnn/feed/FeederInterface.hpp"

#include <stdexcept>
#include <string>

// FeatureMapUnit template (just specify feature map type FM)
//
// Main functionality is to make sure the feeder
// is and stays of the correct feature map type.
//
// In contrast to the ActivationMapUnit, this
// type has no activation function.
//
template <class FM> // for FM insert the specified kind of feature map
class FeatureMapUnit: public FedUnit
{
public:
    // Constructor and destructor
    explicit FeatureMapUnit(FM * fmap = NULL) : FedUnit(static_cast<FeederInterface *>(fmap)) {}
    ~FeatureMapUnit(){}

    // restrict feeder to FM
    void setFeeder(FeederInterface * feeder){
        if (FM * fmap = dynamic_cast<FM *>(feeder)) {
            FedUnit::setFeeder(fmap);
        }
        else {
            throw std::invalid_argument("[FeatureMapUnit::setFeeder] Passed feeder of type " + feeder->getIdCode() + " is not compatible to a unit of type " + this->getIdCode() + ".");
        }
    }

    FM * getMap(){return static_cast<FM *>(_feeder);}

    // devirtualize
    void computeOutput(){FedUnit::computeOutput();}
    void computeValues(){FedUnit::computeValues();}
};

#endif
