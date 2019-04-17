#ifndef FFNN_FMAP_FEATUREMAPUNIT_HPP
#define FFNN_FMAP_FEATUREMAPUNIT_HPP

#include "qnets/feed/FeederInterface.hpp"
#include "qnets/unit/FedUnit.hpp"

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
    explicit FeatureMapUnit(FM * fmap = nullptr): FedUnit(static_cast<FeederInterface *>(fmap)) {}
    ~FeatureMapUnit() override = default;

    // restrict feeder to FM
    void setFeeder(FeederInterface * feeder) override
    {
        if (FM * fmap = dynamic_cast<FM *>(feeder)) {
            FedUnit::setFeeder(fmap);
        }
        else {
            throw std::invalid_argument("[FeatureMapUnit::setFeeder] Passed feeder of type " + feeder->getIdCode() + " is not compatible to a unit of type " + this->getIdCode() + ".");
        }
    }

    FM * getMap() { return static_cast<FM *>(_feeder); }

    // devirtualize
    void computeOutput() override { FedUnit::computeOutput(); }
    void computeValues() override { FedUnit::computeValues(); }
};

#endif
