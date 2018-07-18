#ifndef FEATURE_MAP_LAYER
#define FEATURE_MAP_LAYER

#include "NetworkLayer.hpp"
#include "FedLayer.hpp"
#include "FedUnit.hpp"

#include "IdentityMapUnit.hpp"
#include "EuclideanDistanceMapUnit.hpp"

class FeatureMapLayer: public FedLayer
{
protected:
    // currently we control the feature map unit creation just by these integers (actually now obsolete since we use vectors of feature map units)
    int _nidmaps; // number of identity maps
    int _nedmaps; // number of euclidean distance maps

    std::vector<IdentityMapUnit *> _U_idm; // stores pointers to all identity map units
    std::vector<EuclideanDistanceMapUnit *> _U_edm; // stores pointers to all euclidean distance map units

    FedUnit * _newFMU(const int &i); // create a new FeatureMapUnit for index i
    FeederInterface * _newFMF(NetworkLayer * nl, const int &i); // create a new FeatureMap feeder for index i
    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is one of the known feature maps and register
public:
    // --- Constructor

    explicit FeatureMapLayer(const int &nunits); // "default" constructor with minimal information
    FeatureMapLayer(const int &nidmaps, const int &nedmaps, const int &nunits = -1);
    explicit FeatureMapLayer(const std::string &params);

    virtual void construct(const int &nunits);

    // --- Deconstructor

    virtual ~FeatureMapLayer(){}
    virtual void deconstruct(){FedLayer::deconstruct(); _U_idm.clear(); _U_edm.clear();}

    // --- String Codes

    virtual std::string getIdCode(){return "FML";}

    // --- Modify structure

    void setNMaps(const int &nedmaps, const int &nidmaps);

    // --- FeatureMapUnit getters
    int getNIdMapUnits(){return _U_idm.size();}
    int getNEDMapUnits(){return _U_edm.size();}

    IdentityMapUnit * getIdMapUnit(const int &i){return _U_idm[i];}
    EuclideanDistanceMapUnit * getEDMapUnit(const int &i){return _U_edm[i];}

    // --- Connection

    virtual FeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i);
};

#endif
