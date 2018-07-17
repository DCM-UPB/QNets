#ifndef FEATURE_MAP_LAYER
#define FEATURE_MAP_LAYER

#include "NetworkLayer.hpp"
#include "FedLayer.hpp"
#include "FedUnit.hpp"

class FeatureMapLayer: public FedLayer
{
protected:
    // currently we control the feature map unit creation just by these integers
    int _nidmaps; // number of identity maps
    int _nedmaps; // number of euclidean distance maps

    FedUnit * _newFMU(const int &i); // create a new FeatureMapUnit for index i
    FeederInterface * _newFMF(NetworkLayer * nl, const int &i); // create a new FeatureMap feeder for index i
public:
    // --- Constructor

    FeatureMapLayer(const int &nidmaps, const int &nedmaps, const int &nunits = -1);
    explicit FeatureMapLayer(const std::string &params);

    virtual void construct(const int &nunits);

    // --- Deconstructor

    virtual ~FeatureMapLayer(){}
    virtual void deconstruct(){FedLayer::deconstruct();}

    // --- String Codes

    virtual std::string getIdCode(){return "FML";}

    // --- Modify structure

    void setNMaps(const int &nidmaps, const int &nedmaps);

    // --- Connection

    virtual FeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i);
};

#endif
