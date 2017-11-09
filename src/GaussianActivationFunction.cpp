#include "GaussianActivationFunction.hpp"

#include <math.h>



std::string GaussianActivationFunction::getIdCode()
{
    return "gss";
}


// Activation Function Interface implementation

double GaussianActivationFunction::f(const double &in)
{
   return exp(-in*in);
}


double GaussianActivationFunction::f1d(const double &in)
{
   return -2.*in*exp(-in*in);
}


double GaussianActivationFunction::f2d(const double &in)
{
   return 2.*exp(-in*in)*( -1. + 2.*in*in );
}
