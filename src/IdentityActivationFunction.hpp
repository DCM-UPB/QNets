#ifndef IDENTITY_ACTIVATION_FUNCTION
#define IDENTITY_ACTIVATION_FUNCTION


#include <string>
#include "ActivationFunctionInterface.hpp"

class IdentityActivationFunction: public ActivationFunctionInterface
{
protected:

public:

    // getters
    ActivationFunctionInterface * getCopy(){return new IdentityActivationFunction();}
    std::string getIdCode(){return "ID";}

    // computation
    double f(const double &in){return in;}
    double f1d(const double &in){return 1.;}
    double f2d(const double &in){return 0.;}
    double f3d(const double &in){return 0.;}

    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false)
    {
        v = in;
        v1d = flag_d1 ? 1. : 0.;
        v2d = 0.;
        v3d = 0.;
    }
};


#endif
