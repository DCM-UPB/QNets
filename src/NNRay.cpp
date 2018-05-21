#include "NNRay.hpp"

#include <iostream>
#include <algorithm>




// --- Variational Parameters

bool NNRay::setVariationalParameterValue(const int &id, const double &value){
    if ( isBetaIndexUsedInThisRay(id) ){
        _intensity[ id - _intensity_id_shift ] = value;
        return true;
    }
    else{
        return false;
    }
}


bool NNRay::getVariationalParameterValue(const int &id, double &value){
    if ( isBetaIndexUsedInThisRay(id) ){
        value = _intensity[ id - _intensity_id_shift ];
        return true;
    }
    else{
        value = 0.;
        return false;
    }
}


int NNRay::setVariationalParametersIndexes(const int &starting_index){
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

double NNRay::getFeed(){
    double feed = 0.;
    for (std::vector<NNUnit *>::size_type i=0; i<_source.size(); ++i){
        feed += _intensity[i]*_source[i]->getValue();
    }
    return feed;
}


double NNRay::getFirstDerivativeFeed(const int &i1d){
    double feed = 0.;
    for (std::vector<NNUnit *>::size_type i=1; i<_source.size(); ++i){
        feed += _intensity[i]*_source[i]->getFirstDerivativeValue(i1d);
    }
    return feed;
}


double NNRay::getSecondDerivativeFeed(const int &i2d){
    double feed = 0.;
    for (std::vector<NNUnit *>::size_type i=1; i<_source.size(); ++i){
        feed += _intensity[i]*_source[i]->getSecondDerivativeValue(i2d);
    }
    return feed;
}


double NNRay::getVariationalFirstDerivativeFeed(const int &iv1d){
    double feed = 0.;

    // if the variational parameter with index iv1d is in the ray add the following element
    if ( isBetaIndexUsedInThisRay(iv1d) ){
        feed += _source[ iv1d - _intensity_id_shift ]->getValue();
    }
    // add all other components
    for (std::vector<NNUnit *>::size_type i=1; i<_source.size(); ++i){
        feed += _intensity[i] * _source[i]->getVariationalFirstDerivativeValue(iv1d);
    }

    return feed;
}


double NNRay::getCrossFirstDerivativeFeed(const int &i1d, const int &iv1d){
    double feed = 0.;

    // if the variational parameter with index iv1d is in the ray add the following element
    if ( isBetaIndexUsedInThisRay(iv1d) ){
        feed += _source[ iv1d - _intensity_id_shift ]->getFirstDerivativeValue(i1d);
    }
    // add all other components
    for (std::vector<NNUnit *>::size_type i=1; i<_source.size(); ++i){
        feed += _intensity[i] * _source[i]->getCrossFirstDerivativeValue(i1d, iv1d);
    }

    return feed;
}


double NNRay::getCrossSecondDerivativeFeed(const int &i2d, const int &iv2d){
    double feed = 0.;

    // if the variational parameter with index iv1d is in the ray add the following element
    if ( isBetaIndexUsedInThisRay(iv2d) ){
        feed += _source[ iv2d - _intensity_id_shift ]->getSecondDerivativeValue(i2d);
    }
    // add all other components
    for (std::vector<NNUnit *>::size_type i=1; i<_source.size(); ++i){
        feed += _intensity[i] * _source[i]->getCrossSecondDerivativeValue(i2d, iv2d);
    }

    return feed;
}



// --- Beta Index

bool NNRay::isBetaIndexUsedInThisRay(const int &id){
    boolean * answer = _beta_used_in_this_ray.find(id);
    if (answer != _beta_used_in_this_ray.end()){
        return *answer;
    } else {
        std::vector<int>::iterator it_beta = std::find(_intensity_id.begin(), _intensity_id.end(), id);
        if ( it_beta != _intensity_id.end() ){
            _beta_used_in_this_ray[id] = true;
            return true;
        } else {
            _beta_used_in_this_ray[id] = false;
            return false;
        }
    }
}



bool NNRay::isBetaIndexUsedForThisRay(const int &id){
    boolean * answer = _beta_used_for_this_ray.find(id);
    if (answer != _beta_used_for_this_ray.end()){
        return *answer;
    } else {
        if (isBetaIndexUsedInThisRay(id)){
            _beta_used_for_this_ray[id] = true;
            return true;
        }

        for (NNUnit * u: _source){
            NNUnitFeederInterface * feeder = u->getFeeder();
            if (feeder != 0){
                if (feeder->isBetaIndexUsedForThisRay(id)){
                    _beta_used_for_this_ray[id] = true;
                    return true;
                }
            }
        }

        _beta_used_for_this_ray[id] = false;
        return false;
    }
}



// --- Constructor

NNRay::NNRay(NNLayer * nnl){
    // target sigma to keep sum of weighted inputs in range [-4,4], assuming uniform distribution
    // sigma = 8/sqrt(12) = (b-a)/sqrt(12) * m^(1/2)
    const double bah = 4 * pow(nnl->getNUnits(), -0.5); // (b-a)/2

    _rgen = std::mt19937_64(_rdev());
    _rd = std::uniform_real_distribution<double>(-bah,bah);

    for (int i=0; i<nnl->getNUnits(); ++i){
        _source.push_back(nnl->getUnit(i));
        _intensity.push_back(_rd(_rgen));
    }

    _intensity_id.clear();
}


// --- Destructor

NNRay::~NNRay(){
    _source.clear();
    _intensity.clear();
    _intensity_id.clear();
}
