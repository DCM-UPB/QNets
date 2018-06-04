#include "NetworkUnitRay.hpp"
#include "NetworkUnit.hpp"

#include <algorithm>


// --- Variational Parameters

bool NetworkUnitRay::setVariationalParameterValue(const int &id, const double &value){
    if ( isBetaIndexUsedInThisRay(id) ){
        _intensity[ id - _intensity_id_shift ] = value;
        return true;
    }
    else{
        return false;
    }
}


bool NetworkUnitRay::getVariationalParameterValue(const int &id, double &value){
    if ( isBetaIndexUsedInThisRay(id) ){
        value = _intensity[ id - _intensity_id_shift ];
        return true;
    }
    else{
        value = 0.;
        return false;
    }
}


<<<<<<< HEAD:src/NNRay.cpp
int NNRay::setVariationalParametersIndexes(const int &starting_index){
    // Here we assign external vp indexes to internal indexes.
    // Also, we create two betas_used sets which are explained in the header.
    // NOTE: The current method assumes, that no index larger than max_id,
    //       max_id = starting_index + source.size() - 1 ,
    //       may be in use FOR (and trivially IN) this ray. 
    // NOTE2: betas_used sets are automatically sorted -> binary_search can be used later
    
    for (NNUnit * u: _source){
        NNUnitFeederInterface * feeder = u->getFeeder();
        if (feeder != 0){
            for (int i=0; i<starting_index; ++i) {
                if (feeder->isBetaIndexUsedForThisRay(i)){
                    _betas_used_for_this_ray.insert(i);
                }
            }
        }
    }
=======
int NetworkUnitRay::setVariationalParametersIndexes(const int &starting_index){
    _intensity_id_shift=starting_index;
>>>>>>> Introduced NetworkLayerInterface, NNLayer is now derived. Name changes: NNUnitFeederInterface -> NetworkUnitFeederInterface, NNRay -> NetworkUnitRay. Doesn't work correctly yet.:src/NetworkUnitRay.cpp

    _intensity_id_shift=starting_index;
    int idx=starting_index;
    _intensity_id.clear();
    for (std::vector<double>::size_type i=0; i<_intensity.size(); ++i){
        _intensity_id.push_back(idx);
        _betas_used_in_this_ray.insert(idx);
        _betas_used_for_this_ray.insert(idx);

        idx++;
    }

    return idx;
}


// --- Computation

double NetworkUnitRay::getFeed(){
    double feed = 0.;
    for (std::vector<NetworkUnit *>::size_type i=0; i<_source.size(); ++i){
        feed += _intensity[i]*_source[i]->getValue();
    }
    return feed;
}


double NetworkUnitRay::getFirstDerivativeFeed(const int &i1d){
    double feed = 0.;
    for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i){
        feed += _intensity[i]*_source[i]->getFirstDerivativeValue(i1d);
    }
    return feed;
}


double NetworkUnitRay::getSecondDerivativeFeed(const int &i2d){
    double feed = 0.;
    for (std::vector<NetworkUnit *>::size_type i=1; i<_source.size(); ++i){
        feed += _intensity[i]*_source[i]->getSecondDerivativeValue(i2d);
    }
    return feed;
}


double NetworkUnitRay::getVariationalFirstDerivativeFeed(const int &iv1d){
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


double NetworkUnitRay::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d){
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


double NetworkUnitRay::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d){
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

bool NetworkUnitRay::isBetaIndexUsedInThisRay(const int &id){
    std::vector<int>::iterator it_beta = std::find(_intensity_id.begin(), _intensity_id.end(), id);
    if ( it_beta != _intensity_id.end() ){
        return true;
    }
    else {
        return false;
    }
}



bool NetworkUnitRay::isBetaIndexUsedForThisRay(const int &id){
    if (isBetaIndexUsedInThisRay(id)){
        return true;
    }

    for (NetworkUnit * u: _source){
        NetworkUnitFeederInterface * feeder = u->getFeeder();
        if (feeder != 0){
            if (feeder->isBetaIndexUsedForThisRay(id)){
                return true;
            }
        }
    }
}


// --- Constructor

NetworkUnitRay::NetworkUnitRay(NetworkLayerInterface * nl){
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

NetworkUnitRay::~NetworkUnitRay(){
    _source.clear();
    _intensity.clear();
    _intensity_id.clear();
    _betas_used_in_this_ray.clear();
    _betas_used_for_this_ray.clear();
}
