#include "NNRay.hpp"

#include <iostream>
#include <algorithm>


// --- Variational Parameters

bool NNRay::setVariationalParameterValue(const int &id, const double &value)
{
   std::vector<int>::iterator it_beta = std::find(_intensity_id.begin(), _intensity_id.end(), id);
   if ( it_beta != _intensity_id.end() )
   {
      _intensity[ *it_beta - _intensity_id_shift ] = value;
      return true;
   }
   else
   {
      return false;
   }
}


bool NNRay::getVariationalParameterValue(const int &id, double &value)
{
   std::vector<int>::iterator it_beta = std::find(_intensity_id.begin(), _intensity_id.end(), id);
   if ( it_beta != _intensity_id.end() )
   {
      value = _intensity[ *it_beta - _intensity_id_shift ];
      return true;
   }
   else
   {
      value = 0.;
      return false;
   }
}


int NNRay::setVariationalParametersIndexes(const int &starting_index)
{
   _intensity_id_shift=starting_index;

   int idx=starting_index;
   _intensity_id.clear();
   for (std::vector<double>::size_type i=0; i<_intensity.size(); ++i)
   {
      _intensity_id.push_back(idx);
      idx++;
   }
   return idx;
}


// --- Computation

double NNRay::getFeed()
{
   double feed = 0.;
   for (std::vector<NNUnit *>::size_type i=0; i<_source.size(); ++i)
   {
      feed += _intensity[i]*_source[i]->getValue();
   }
   return feed;
}


double NNRay::getFirstDerivativeFeed(const int &i1d)
{
   double feed = 0.;
   for (std::vector<NNUnit *>::size_type i=1; i<_source.size(); ++i)
   {
      feed += _intensity[i]*_source[i]->getFirstDerivativeValue(i1d);
   }
   return feed;
}


double NNRay::getSecondDerivativeFeed(const int &i2d)
{
   double feed = 0.;
   for (std::vector<NNUnit *>::size_type i=1; i<_source.size(); ++i)
   {
      feed += _intensity[i]*_source[i]->getSecondDerivativeValue(i2d);
   }
   return feed;
}


double NNRay::getVariationalFirstDerivativeFeed(const int &iv1d)
{
   double feed = 0.;

   // if the variational parameter with index iv1d is in the ray add the following element
   std::vector<int>::iterator it_beta = std::find(_intensity_id.begin(), _intensity_id.end(), iv1d);
   if ( it_beta != _intensity_id.end() )
   {
      feed += _source[ *it_beta - _intensity_id_shift ]->getValue();
   }
   // add all other components
   for (std::vector<NNUnit *>::size_type i=1; i<_source.size(); ++i)
   {
      feed += _intensity[i] * _source[i]->getVariationalFirstDerivativeValue(iv1d);
   }

   return feed;
}


// --- Constructor

NNRay::NNRay(NNLayer * nnl)
{

   // m^-(1/2) = sigma = (b-a)/sqrt(12)
   const double bah = sqrt(3) * pow(nnl->getNUnits(), -0.5);

   _rgen = std::mt19937_64(_rdev());
   _rd = std::uniform_real_distribution<double>(-bah,bah);

   for (int i=0; i<nnl->getNUnits(); ++i)
   {
      _source.push_back(nnl->getUnit(i));
      _intensity.push_back(_rd(_rgen));
   }

   _intensity_id.clear();
}


// --- Destructor

NNRay::~NNRay()
{
   _source.clear();
   _intensity.clear();
   _intensity_id.clear();
}
