#include "FeatureMapLayer.hpp"
#include "NetworkLayer.hpp"
#include "FedUnit.hpp"
#include "FeederInterface.hpp"
#include "IdentityMapUnit.hpp"
#include "EuclideanDistanceMapUnit.hpp"


#include <iostream>

// --- Feature Map helpers

FedUnit * FeatureMapLayer::_newFMU(const int &i)
{
    if (i<=_nedmaps) {
        return new EuclideanDistanceMapUnit();
    }
    else {
        return new IdentityMapUnit();
    }
}

FeederInterface * FeatureMapLayer::_newFMF(NetworkLayer * nl, const int &i)
{
    if (i<=_nedmaps) {
        return new EuclideanDistanceMap(nl);
    }
    else {
        return new IdentityMap(nl);
    }
}


// --- Constructor

FeatureMapLayer::FeatureMapLayer(const int &nunits): _nidmaps(nunits-1), _nedmaps(0) // minimal initialization with ID maps
{
    if (nunits>1) construct(nunits);
}

FeatureMapLayer::FeatureMapLayer(const int &nidmaps, const int &nedmaps, const int &nunits): _nidmaps(nidmaps), _nedmaps(nedmaps)
{
    // if the user did specify nunits, don't calculate it
    int true_nunits = nunits < 0 ? 1 + _nidmaps + _nedmaps : nunits;
    if (true_nunits>1) construct(true_nunits);
}


FeatureMapLayer::FeatureMapLayer(const std::string &params)
{
    int nunits;
    setParamValue(readParamValue(params, "nunits"), nunits);
    int nidmaps;
    setParamValue(readParamValue(params, "nidmaps"), nidmaps);
    int nedmaps;
    setParamValue(readParamValue(params, "nedmaps"), nedmaps);

    FeatureMapLayer(nunits, nidmaps, nedmaps);
}


void FeatureMapLayer::construct(const int &nunits)
{
    if (nunits > 1 + _nidmaps + _nedmaps) {
        cout << endl << "[FeatureMapLayer::construct] Warning: Desired number of units is higher than 1 (offset) + number of IdMaps + number of EDMaps. The extra units will default to IDMaps." << endl << endl;
    }
    else if (nunits < 1 + _nidmaps + _nedmaps) {
        cout << endl << "[FeatureMapLayer::construct] Warning: Desired number of units is lower than 1 (offset) + number of IdMaps + number of EDMaps. This means desired maps beyond nunits will not be created." << endl << endl;
    }

    FedUnit * newUnit;
    for (int i=1; i<nunits; ++i) {
        newUnit = _newFMU(i);
        _registerUnit(newUnit);
    }
}


// --- Modify structure

void FeatureMapLayer::setNMaps(const int &nidmaps, const int &nedmaps)
{
    _nidmaps = nidmaps;
    _nedmaps = nedmaps;
    this->setSize(1 + nidmaps + nedmaps);
}

FeederInterface * FeatureMapLayer::connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i)
{
    return _newFMF(nl, i);
}
