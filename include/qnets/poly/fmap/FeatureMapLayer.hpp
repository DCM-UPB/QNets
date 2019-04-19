#ifndef FFNN_FMAP_FEATUREMAPLAYER_HPP
#define FFNN_FMAP_FEATUREMAPLAYER_HPP

#include "qnets/poly/layer/FedLayer.hpp"
#include "qnets/poly/layer/NetworkLayer.hpp"
#include "qnets/poly/unit/FedUnit.hpp"

#include "qnets/poly/fmap/EuclideanDistanceMapUnit.hpp"
#include "qnets/poly/fmap/EuclideanPairDistanceMapUnit.hpp"
#include "qnets/poly/fmap/IdentityMapUnit.hpp"
#include "qnets/poly/fmap/PairDifferenceMapUnit.hpp"
#include "qnets/poly/fmap/PairSumMapUnit.hpp"

class FeatureMapLayer: public FedLayer
{
protected:
    // currently we control the feature map unit creation by these integers
    int _npsmaps{}; // DESIRED number of pair sum maps
    int _npdmaps{}; // DESIRED number of pair difference maps
    int _nedmaps{}; // DESIRED number of euclidean distance maps
    int _nepdmaps{}; // DESIRED number of euclidean pair distance maps
    int _nidmaps{}; // DESIRED number of identity maps

    std::vector<PairSumMapUnit *> _U_psm; // stores pointers to all pair sum map units
    std::vector<PairDifferenceMapUnit *> _U_pdm; // stores pointers to all pair difference map units
    std::vector<EuclideanDistanceMapUnit *> _U_edm; // stores pointers to all euclidean distance map units
    std::vector<EuclideanPairDistanceMapUnit *> _U_epdm; // stores pointers to all euclidean pair distance map units
    std::vector<IdentityMapUnit *> _U_idm; // stores pointers to all identity map units

    FedUnit * _newFMU(const int &i); // create a new FeatureMapUnit for index i
    FeederInterface * _newFMF(NetworkLayer * nl, const int &i); // create a new FeatureMap feeder for index i
    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is one of the known feature maps and register
public:
    // --- Constructor / Destructor

    explicit FeatureMapLayer(const int &nunits); // "default" constructor with minimal information
    FeatureMapLayer(const int &npsmaps, const int &npdmaps, const int &nedmaps, const int &nepdmaps, const int &nidmaps, const int &nunits = -1);
    explicit FeatureMapLayer(const std::string &params);
    ~FeatureMapLayer() override;

    // --- construct / deconstruct methods

    void construct(const int &nunits) override;
    void deconstruct() override;

    // --- String Codes

    std::string getIdCode() override { return "FML"; }
    std::string getParams() override;
    void setParams(const std::string &params) override;

    // --- Modify structure

    void setNMaps(const int &npsmaps, const int &npdmaps, const int &nedmaps, const int &nepdmaps, const int &nidmaps);

    // --- FeatureMapUnit getters
    int getNPSMapUnits() { return _U_psm.size(); }
    int getNPDMapUnits() { return _U_pdm.size(); }
    int getNEDMapUnits() { return _U_edm.size(); }
    int getNEPDMapUnits() { return _U_epdm.size(); }
    int getNIdMapUnits() { return _U_idm.size(); }

    PairSumMapUnit * getPSMapUnit(const int &i) { return _U_psm[i]; }
    PairDifferenceMapUnit * getPDMapUnit(const int &i) { return _U_pdm[i]; }
    EuclideanDistanceMapUnit * getEDMapUnit(const int &i) { return _U_edm[i]; }
    EuclideanPairDistanceMapUnit * getEPDMapUnit(const int &i) { return _U_epdm[i]; }
    IdentityMapUnit * getIdMapUnit(const int &i) { return _U_idm[i]; }

    // --- Connection

    FeederInterface * connectUnitOnTopOfLayer(NetworkLayer * nl, const int &i) override;
};

#endif
