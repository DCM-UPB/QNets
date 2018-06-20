#ifndef FED_NETWORK_UNIT
#define FED_NETWORK_UNIT

#include "NetworkUnit.hpp"
#include "NetworkUnitFeederInterface.hpp"

#include <string>
#include <cstddef> // for NULL

// Network Unit with Feeder
class FedNetworkUnit: virtual public NetworkUnit
{
protected:
    // Feeder of the unit
    // The feeder of a unit is a class that takes care of providing the input to the unit, when called via: _feeder->getFeed()
    NetworkUnitFeederInterface * _feeder;

public:
    // Constructor and destructor
    explicit FedNetworkUnit(NetworkUnitFeederInterface * feeder = NULL){_feeder = feeder;}
    virtual ~FedNetworkUnit(){if (_feeder) delete _feeder; _feeder=NULL;}

    // return the output mean value (mu) and standard deviation (sigma)
    // (pretending a flat distribution and without feeder assuming normalized pv input (i.e. m = 0, s = 1) )
    virtual double getOutputMu(){return _feeder ? _feeder->getFeedMu() : 0;}
    virtual double getOutputSigma(){return _feeder ? _feeder->getFeedSigma() : 1;}

    // Setters and getters
    void setFeeder(NetworkUnitFeederInterface * feeder){if (_feeder) delete _feeder; _feeder = feeder;}
    NetworkUnitFeederInterface * getFeeder(){return _feeder;}

    // string code getters / setter
    virtual std::string getMemberTreeCode(){return _feeder ? _feeder->getTreeCode() : "";} // return feeder's IdCodes + Params Tree
    virtual void setMemberParams(const std::string &memberTreeCode) {if (_feeder) _feeder->setTreeParams(readTreeCode(memberTreeCode, 0, _feeder->getIdCode()));}

    // Computation
    void computeFeed();
    void computeDerivatives();
};

#endif
