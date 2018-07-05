#ifndef ACTIVATION_FUNCTION_INTERFACE
#define ACTIVATION_FUNCTION_INTERFACE

#include "SerializableComponent.hpp"

#include <string>


class ActivationFunctionInterface: public SerializableComponent
{
protected:

public:

    // Constructors: every child with hyper parameters should implement 3 constructors:

    // Construct from passed parameter values
    //ActivationFunctionInterface(){}

    // Construct from passed param string
    //ActivationFunctionInterface(const std::string &params){}

    // Construct from passed activation function
    //ActivationFunctionInterface(ActivationFunctionInterface * const actf){}

    // Destructor
    virtual ~ActivationFunctionInterface(){}

    // allocate a new copy of this to *actf
    virtual ActivationFunctionInterface * getCopy() = 0;

    // set class id code
    std::string getClassIdCode(){return "ACTF";}

    // compute the activation function value
    virtual double f(const double &) = 0;

    // first derivative of the activation function
    virtual double f1d(const double &) = 0;

    // second derivative of the activation function
    virtual double f2d(const double &) = 0;

    // third derivative of the activation function
    virtual double f3d(const double &) = 0;

    // function to calculate function value and all needed derivatives together, allowing for speedup over individual calls
    virtual void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false)
    {
        // Generic virtual implementation, overwriting it will generally yield faster function calls!
        // If possible, the fad function should at least break even in speed if at least 1 derivative is required and be faster for 2 or more.

        v = f(in);
        v1d = flag_d1 ? this->f1d(in) : 0.0;
        v2d = flag_d2 ? this->f2d(in) : 0.0;
        v3d = flag_d3 ? this->f3d(in) : 0.0;
    };
};


#endif
