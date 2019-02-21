#include "ffnn/fmap/FeatureMapLayer.hpp"
#include "ffnn/layer/NetworkLayer.hpp"
#include "ffnn/unit/FedUnit.hpp"
#include "ffnn/feed/FeederInterface.hpp"

#include <iostream>

// --- Register Unit

void FeatureMapLayer::_registerUnit(NetworkUnit * newUnit)
{
    FedLayer::_registerUnit(newUnit);
    if(PairSumMapUnit * psmu = dynamic_cast<PairSumMapUnit *>(newUnit)) {
        _U_psm.push_back(psmu);
    }
    else if(PairDifferenceMapUnit * pdmu = dynamic_cast<PairDifferenceMapUnit *>(newUnit)) {
        _U_pdm.push_back(pdmu);
    }
    else if(EuclideanDistanceMapUnit * edmu = dynamic_cast<EuclideanDistanceMapUnit *>(newUnit)) {
        _U_edm.push_back(edmu);
    }
    else if(EuclideanPairDistanceMapUnit * epdmu = dynamic_cast<EuclideanPairDistanceMapUnit *>(newUnit)) {
        _U_epdm.push_back(epdmu);
    }
    else if(IdentityMapUnit * idmu = dynamic_cast<IdentityMapUnit *>(newUnit)) {
        _U_idm.push_back(idmu);
    }
}


// --- Feature Map helpers

FedUnit * FeatureMapLayer::_newFMU(const int &i)
{
    int ubound = _npsmaps;
    if (i<ubound) {
        return new PairSumMapUnit();
    }

    ubound += _npdmaps;
    if (i<ubound) {
        return new PairDifferenceMapUnit();
    }

    ubound += _nedmaps;
    if (i<ubound) {
        return new EuclideanDistanceMapUnit();
    }

    ubound += _nepdmaps;
    if (i<ubound) {
        return new EuclideanPairDistanceMapUnit();
    }

    return new IdentityMapUnit();
}

FeederInterface * FeatureMapLayer::_newFMF(NetworkLayer * nl, const int &i)
{
    int ubound = _npsmaps;
    if (i<ubound) {
        return new PairSumMap(nl);
    }

    ubound += _npdmaps;
    if (i<ubound) {
        return new PairDifferenceMap(nl);
    }

    ubound += _nedmaps;
    if (i<ubound) {
        return new EuclideanDistanceMap(nl);
    }

    ubound += _nepdmaps;
    if (i<ubound) {
        return new EuclideanPairDistanceMap(nl);
    }

    return new IdentityMap(nl);
}


// --- Constructor / Destructor

FeatureMapLayer::FeatureMapLayer(const int &nunits)
    : _npsmaps(0), _npdmaps(0), _nedmaps(0), _nepdmaps(0), _nidmaps(nunits-1) // minimal initialization with ID maps
{
    if (nunits>1) construct(nunits);
}

FeatureMapLayer::FeatureMapLayer(const int &npsmaps, const int &npdmaps, const int &nedmaps, const int &nepdmaps, const int &nidmaps, const int &nunits)
    : _npsmaps(npsmaps), _npdmaps(npdmaps), _nedmaps(nedmaps), _nepdmaps(nepdmaps), _nidmaps(nidmaps)
{
    // if the user did specify nunits, don't calculate it
    int true_nunits = nunits < 0 ? 1 + _npsmaps + _npdmaps +_nedmaps + _nepdmaps + _nidmaps : nunits;
    if (true_nunits>1) construct(true_nunits);
}


FeatureMapLayer::FeatureMapLayer(const std::string &params)
{
    int npsmaps;
    setParamValue(readParamValue(params, "npsmaps"), npsmaps);
    int npdmaps;
    setParamValue(readParamValue(params, "npdmaps"), npdmaps);
    int nedmaps;
    setParamValue(readParamValue(params, "nedmaps"), nedmaps);
    int nepdmaps;
    setParamValue(readParamValue(params, "nepdmaps"), nepdmaps);
    int nidmaps;
    setParamValue(readParamValue(params, "nidmaps"), nidmaps);
    int nunits;
    setParamValue(readParamValue(params, "nunits"), nunits);

    FeatureMapLayer(npsmaps, npdmaps, nedmaps, nepdmaps, nidmaps, nunits);
}

FeatureMapLayer::~FeatureMapLayer()
{
    _U_psm.clear();
    _U_pdm.clear();
    _U_edm.clear();
    _U_epdm.clear();
    _U_idm.clear();
}


// --- construct / deconstruct methods


void FeatureMapLayer::construct(const int &nunits)
{
    if (nunits > 1 + _npsmaps + _npdmaps + _nedmaps + _nepdmaps + _nidmaps) {
        cout << endl << "[FeatureMapLayer::construct] Warning: Desired number of units is higher than 1 (offset) + number of maps. The extra units will default to IDMaps." << endl << endl;
    }
    else if (nunits < 1 + _npsmaps + _npdmaps + _nedmaps + _nepdmaps +_nidmaps) {
        cout << endl << "[FeatureMapLayer::construct] Warning: Desired number of units is lower than 1 (offset) + number of maps. This means desired maps beyond nunits will not be created." << endl << endl;
    }

    for (int i=1; i<nunits; ++i) {
        FedUnit * newUnit = _newFMU(i-1); // we need fedUnit indices here
        _registerUnit(newUnit);
    }
}

void FeatureMapLayer::deconstruct()
{
    FedLayer::deconstruct();
    _U_psm.clear();
    _U_pdm.clear();
    _U_edm.clear();
    _U_epdm.clear();
    _U_idm.clear();
}


// --- String codes

std::string FeatureMapLayer::getParams()
{
    std::string str = composeCodes(NetworkLayer::getParams(), composeParamCode("npsmaps", _npsmaps));
    str = composeCodes(str, composeParamCode("npdmaps", _npdmaps));
    str = composeCodes(str, composeParamCode("nedmaps", _nedmaps));
    str = composeCodes(str, composeParamCode("nepdmaps", _nepdmaps));
    return composeCodes(str, composeParamCode("nidmaps", _nidmaps));
}

void FeatureMapLayer::setParams(const std::string &params)
{
    setParamValue(readParamValue(params, "npsmaps"), _npsmaps);
    setParamValue(readParamValue(params, "npdmaps"), _npdmaps);
    setParamValue(readParamValue(params, "nedmaps"), _nedmaps);
    setParamValue(readParamValue(params, "nepdmaps"), _nepdmaps);
    setParamValue(readParamValue(params, "nidmaps"), _nidmaps);
    NetworkLayer::setParams(params);
}


// --- Modify structure

void FeatureMapLayer::setNMaps(const int &npsmaps, const int &npdmaps, const int &nedmaps, const int &nepdmaps, const int &nidmaps)
{
    _npsmaps = npsmaps;
    _npdmaps = npdmaps;
    _nedmaps = nedmaps;
    _nepdmaps = nepdmaps;
    _nidmaps = nidmaps;
    this->setSize(1 + _npsmaps + _npdmaps + _nedmaps + _nepdmaps + _nidmaps);
}

FeederInterface * FeatureMapLayer::connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i)
{
    return _newFMF(nl, i);
}
