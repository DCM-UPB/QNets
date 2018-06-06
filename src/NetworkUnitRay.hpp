#ifndef NETWORK_UNIT_RAY
#define NETWORK_UNIT_RAY

#include "NetworkUnitFeederInterface.hpp"
#include "NetworkUnit.hpp"
#include "NetworkLayerInterface.hpp"

#include <vector>
#include <set>
#include <random>
#include <algorithm>

template <typename UnitType>
class NetworkUnitRay: public NetworkUnitFeederInterface {
protected:
    // random number generator, used to initialize the intensities
    std::random_device _rdev;
    std::mt19937_64 _rgen;
    std::uniform_real_distribution<double> _rd;

    // key component of the ray: the source and their intensisities
    std::vector<UnitType *> _source;   // units from which the ray takes the values from
    std::vector<double> _intensity;   // intensity of each sorgent unit, i.e. its weight
    std::vector<int> _intensity_id;  // intensity identification id, useful for the NN
    int _intensity_id_shift;  // shift of the previous vector

    // store information about which beta are used in or for this ray
    // NB: 'in' this ray means that the beta is part of the ray
    //     'for' this ray means that the beta is either used in this ray or in another
    //           ray that genreates an output that is directly or indirectly used
    //           in this ray (sources)
    std::set<int> _betas_used_in_this_ray;
    std::set<int> _betas_used_for_this_ray;

public:

    /* Pseudo header
    NetworkUnitRay(NetworkLayerInterface<T> * nl);
    virtual ~NetworkUnitRay();

    // beta
    int getNBeta();
    double getBeta(const int &i);
    void setBeta(const int &i, const double &b);

    // Variational Parameters
    int getNVariationalParameters();
    int setVariationalParametersIndexes(const int &starting_index);
    bool getVariationalParameterValue(const int &id, double &value);
    bool setVariationalParameterValue(const int &id, const double &value);

    // Computation
    double getFeed();
    double getFirstDerivativeFeed(const int &i1d);
    double getSecondDerivativeFeed(const int &i2d);
    double getVariationalFirstDerivativeFeed(const int &iv1d);
    double getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d);
    double getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d);

    // Beta Index
    bool isBetaIndexUsedInThisRay(const int &id);   // beta is directly used?
    bool isBetaIndexUsedForThisRay(const int &id);   // beta is used directly or indirectly?
    */

    // Due to template usage this implementation code must be inside the header (or not all required versions will be compiled when including)


    // --- Betas

    int getNBeta(){return _intensity.size();}
    double getBeta(const int &i){return _intensity[i];}
    void setBeta(const int &i, const double &b){_intensity[i]=b;}

    // --- Variational Parameters

    int getNVariationalParameters(){return _intensity.size();}

    bool setVariationalParameterValue(const int &id, const double &value){
        if ( isBetaIndexUsedInThisRay(id) ){
            _intensity[ id - _intensity_id_shift ] = value;
            return true;
        }
        else{
            return false;
        }
    }


    bool getVariationalParameterValue(const int &id, double &value){
        if ( isBetaIndexUsedInThisRay(id) ){
            value = _intensity[ id - _intensity_id_shift ];
            return true;
        }
        else{
            value = 0.;
            return false;
        }
    }


    int setVariationalParametersIndexes(const int &starting_index){
        _intensity_id_shift=starting_index;

        int idx=starting_index;
        _intensity_id.clear();
        for (std::vector<double>::size_type i=0; i<_intensity.size(); ++i){
            _intensity_id.push_back(idx);
            idx++;
        }
        return idx;
    }


    // --- Computation


    double getFeed(){
        double feed = 0.;
        for (std::vector<NetworkUnit *>::size_type i=0; i<_source.size(); ++i){
            feed += _intensity[i]*_source[i]->getValue();
        }
        return feed;
    }


    double getFirstDerivativeFeed(const int &i1d){
        double feed = 0.;
        for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i){
            feed += _intensity[i]*_source[i]->getFirstDerivativeValue(i1d);
        }
        return feed;
    }


    double getSecondDerivativeFeed(const int &i2d){
        double feed = 0.;
        for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i){
            feed += _intensity[i]*_source[i]->getSecondDerivativeValue(i2d);
        }
        return feed;
    }


    double getVariationalFirstDerivativeFeed(const int &iv1d){
        double feed = 0.;

        // if the variational parameter with index iv1d is in the ray add the following element
        if ( isBetaIndexUsedInThisRay(iv1d) ){
            feed += _source[ iv1d - _intensity_id_shift ]->getValue();
        }
        // add all other components
        for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i){
            feed += _intensity[i] * _source[i]->getVariationalFirstDerivativeValue(iv1d);
        }

        return feed;
    }


    double getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d){
        double feed = 0.;

        // if the variational parameter with index iv1d is in the ray add the following element
        if ( isBetaIndexUsedInThisRay(iv1d) ){
            feed += _source[ iv1d - _intensity_id_shift ]->getFirstDerivativeValue(i1d);
        }
        // add all other components
        for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i){
            feed += _intensity[i] * _source[i]->getCrossFirstDerivativeValue(i1d, iv1d);
        }

        return feed;
    }


    double getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d){
        double feed = 0.;

        // if the variational parameter with index iv1d is in the ray add the following element
        if ( isBetaIndexUsedInThisRay(iv2d) ){
            feed += _source[ iv2d - _intensity_id_shift ]->getSecondDerivativeValue(i2d);
        }
        // add all other components
        for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i){
            feed += _intensity[i] * _source[i]->getCrossSecondDerivativeValue(i2d, iv2d);
        }

        return feed;
    }



    // --- Beta Index

    bool isBetaIndexUsedInThisRay(const int &id){
        std::vector<int>::iterator it_beta = std::find(_intensity_id.begin(), _intensity_id.end(), id);
        if ( it_beta != _intensity_id.end() ){
            return true;
        } else {
            return false;
        }
    }


    bool isBetaIndexUsedForThisRay(const int &id){
        if (isBetaIndexUsedInThisRay(id)){
            return true;
        }

        for (UnitType * u: _source){
            NetworkUnitFeederInterface * feeder = u->getFeeder();
            if (feeder != 0){
                if (feeder->isBetaIndexUsedForThisRay(id)){
                    return true;
                }
            }
        }

        return false;
    }


    // --- Constructor

    template <typename LayerType>
    NetworkUnitRay(LayerType * nl) { // nl){
        // target sigma to keep sum of weighted inputs in range [-4,4], assuming uniform distribution
        // sigma = 8/sqrt(12) = (b-a)/sqrt(12) * m^(1/2)
        const double bah = 4 * pow(nl->getNUnits(), -0.5); // (b-a)/2

        _rgen = std::mt19937_64(_rdev());
        _rd = std::uniform_real_distribution<double>(-bah,bah);

        for (int i=0; i<nl->getNUnits(); ++i){
            _source.push_back(nl->getUnit(i));
            _intensity.push_back(_rd(_rgen));
        }

        _intensity_id.clear();
    }

    // --- Destructor

    ~NetworkUnitRay(){
        _source.clear();
        _intensity.clear();
        _intensity_id.clear();
    }

};


#endif
