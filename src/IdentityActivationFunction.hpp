#ifndef IDENTITY_ACTIVATION_FUNCTION
#define IDENTITY_ACTIVATION_FUNCTION


#include <string>


class IdentityActivationFunction: public ActivationFunctionInterface
{
protected:

public:
    std::string getIdCode(){return "id_";}
    double f(const double &in){return in;}
    double f1d(const double &in){return 1.;}
    double f2d(const double &in){return 0.;}
};


#endif
