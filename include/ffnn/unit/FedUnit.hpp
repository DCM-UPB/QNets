#ifndef FFNN_UNIT_FEDUNIT_HPP
#define FFNN_UNIT_FEDUNIT_HPP

#include "ffnn/feed/FeederInterface.hpp"
#include "ffnn/unit/NetworkUnit.hpp"

#include <cstddef> // for NULL
#include <string>

// Network Unit with Feeder
class FedUnit: virtual public NetworkUnit
{
protected:
    // Feeder of the unit
    // The feeder of a unit is a class that takes care of providing the input to the unit, when called via: _feeder->getFeed()
    FeederInterface * _feeder;

public:
    // Constructor and destructor
    explicit FedUnit(FeederInterface * feeder = nullptr){_feeder = feeder;}
    ~FedUnit() override{ delete _feeder; _feeder=nullptr; }

    // return the output mean value (mu) and standard deviation (sigma)
    // (pretending a flat distribution and without feeder assuming normalized pv input (i.e. m = 0, s = 1) )
    double getOutputMu() override{return _feeder != nullptr ? _feeder->getFeedMu() : 0;}
    double getOutputSigma() override{return _feeder != nullptr ? _feeder->getFeedSigma() : 1;}

    // Setters and getters
    virtual void setFeeder(FeederInterface * feeder){ delete _feeder; _feeder = feeder;} // you may extend this to restrict type
    FeederInterface * getFeeder(){return _feeder;}

    // string code getters / setter
    std::string getMemberTreeCode() override{return _feeder != nullptr ? _feeder->getTreeCode() : "";} // return feeder's IdCodes + Params Tree
    void setMemberParams(const std::string &memberTreeCode) override
    {
        if (_feeder != nullptr) { _feeder->setTreeParams(readTreeCode(memberTreeCode, 0, _feeder->getIdCode())); }
    }
    std::string getIdCode() override = 0; // virtual class

    // Computation
    void computeFeed() override;
    void computeDerivatives() override;
};

#endif
