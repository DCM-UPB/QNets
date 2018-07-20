#ifndef FEATURE_MAP_LAYER
#define FEATURE_MAP_LAYER

#include "NetworkLayer.hpp"
#include "FedLayer.hpp"
#include "FedUnit.hpp"

#include "EuclideanDistanceMapUnit.hpp"
#include "IdentityMapUnit.hpp"

class FeatureMapLayer: public FedLayer
{
protected:
    // currently we control the feature map unit creation by these integers
    int _nedmaps; // DESIRED number of euclidean distance maps
    int _nidmaps; // DESIRED number of identity maps

    std::vector<EuclideanDistanceMapUnit *> _U_edm; // stores pointers to all euclidean distance map units
    std::vector<IdentityMapUnit *> _U_idm; // stores pointers to all identity map units

    FedUnit * _newFMU(const int &i); // create a new FeatureMapUnit for index i
    FeederInterface * _newFMF(NetworkLayer * nl, const int &i); // create a new FeatureMap feeder for index i
    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is one of the known feature maps and register
public:
    // --- Constructor / Destructor

    explicit FeatureMapLayer(const int &nunits); // "default" constructor with minimal information
    FeatureMapLayer(const int &nedmaps, const int &nidmaps, const int &nunits = -1);
    explicit FeatureMapLayer(const std::string &params);
    ~FeatureMapLayer();

    // --- construct / deconstruct methods

    void construct(const int &nunits);
    void deconstruct();

    // --- String Codes

    std::string getIdCode(){return "FML";}
    std::string getParams();
    void setParams(const std::string &params);

    // --- Modify structure

    void setNMaps(const int &nedmaps, const int &nidmaps);

    // --- FeatureMapUnit getters
    int getNIdMapUnits(){return _U_idm.size();}
    int getNEDMapUnits(){return _U_edm.size();}

    IdentityMapUnit * getIdMapUnit(const int &i){return _U_idm[i];}
    EuclideanDistanceMapUnit * getEDMapUnit(const int &i){return _U_edm[i];}

    // --- Connection

    FeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i);
};

#endif
