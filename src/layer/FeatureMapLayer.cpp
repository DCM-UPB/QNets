#include "FeatureMapLayer.hpp"
#include "NetworkLayer.hpp"
#include "FedUnit.hpp"
#include "FeederInterface.hpp"
#include "EuclideanDistanceMapUnit.hpp"
#include "IdentityMapUnit.hpp"

#include <iostream>

// --- Register Unit

void FeatureMapLayer::_registerUnit(NetworkUnit * newUnit)
{
    FedLayer::_registerUnit(newUnit);
    if(EuclideanDistanceMapUnit * edmu = dynamic_cast<EuclideanDistanceMapUnit *>(newUnit)) {
        _U_edm.push_back(edmu);
    }
    if(IdentityMapUnit * idmu = dynamic_cast<IdentityMapUnit *>(newUnit)) {
        _U_idm.push_back(idmu);
    }
}


// --- Feature Map helpers

FedUnit * FeatureMapLayer::_newFMU(const int &i)
{
    if (i<_nedmaps) {
        return new EuclideanDistanceMapUnit();
    }
    else {
        return new IdentityMapUnit();
    }
}

FeederInterface * FeatureMapLayer::_newFMF(NetworkLayer * nl, const int &i)
{
    if (i<_nedmaps) {
        return new EuclideanDistanceMap(nl);
    }
    else {
        return new IdentityMap(nl);
    }
}


// --- Constructor / Destructor

FeatureMapLayer::FeatureMapLayer(const int &nunits): _nedmaps(0), _nidmaps(nunits-1) // minimal initialization with ID maps
{
    if (nunits>1) construct(nunits);
}

FeatureMapLayer::FeatureMapLayer(const int &nedmaps, const int &nidmaps, const int &nunits): _nedmaps(nedmaps), _nidmaps(nidmaps)
{
    // if the user did specify nunits, don't calculate it
    int true_nunits = nunits < 0 ? 1 + _nedmaps + _nidmaps : nunits;
    if (true_nunits>1) construct(true_nunits);
}


FeatureMapLayer::FeatureMapLayer(const std::string &params)
{
    int nunits;
    setParamValue(readParamValue(params, "nunits"), nunits);
    int nedmaps;
    setParamValue(readParamValue(params, "nedmaps"), nedmaps);
    int nidmaps;
    setParamValue(readParamValue(params, "nidmaps"), nidmaps);

    FeatureMapLayer(nedmaps, nidmaps, nunits);
}

FeatureMapLayer::~FeatureMapLayer()
{
    _U_edm.clear();
    _U_idm.clear();
}


// --- construct / deconstruct methods


void FeatureMapLayer::construct(const int &nunits)
{
    if (nunits > 1 + _nedmaps + _nidmaps) {
        cout << endl << "[FeatureMapLayer::construct] Warning: Desired number of units is higher than 1 (offset) + number of IdMaps + number of EDMaps. The extra units will default to IDMaps." << endl << endl;
    }
    else if (nunits < 1 + _nedmaps + _nidmaps) {
        cout << endl << "[FeatureMapLayer::construct] Warning: Desired number of units is lower than 1 (offset) + number of IdMaps + number of EDMaps. This means desired maps beyond nunits will not be created." << endl << endl;
    }

    FedUnit * newUnit;
    for (int i=1; i<nunits; ++i) {
        newUnit = _newFMU(i-1); // we need fedUnit indices here
        _registerUnit(newUnit);
    }
}

void FeatureMapLayer::deconstruct()
{
    FedLayer::deconstruct();
    _U_edm.clear();
    _U_idm.clear();
}


// --- String codes

std::string FeatureMapLayer::getParams()
{
    std::string str = composeCodes(NetworkLayer::getParams(), composeParamCode("nedmaps", _U_edm.size()));
    return composeCodes(str, composeParamCode("nidmaps", _U_idm.size()));
}

void FeatureMapLayer::setParams(const std::string &params)
{
    setParamValue(readParamValue(params, "nedmaps"), _nedmaps);
    setParamValue(readParamValue(params, "nidmaps"), _nidmaps);
    NetworkLayer::setParams(params);
}


// --- Modify structure

void FeatureMapLayer::setNMaps(const int &nedmaps, const int &nidmaps)
{
    _nedmaps = nedmaps;
    _nidmaps = nidmaps;
    this->setSize(1 + nedmaps + nidmaps);
}

FeederInterface * FeatureMapLayer::connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i)
{
    return _newFMF(nl, i);
}
